#include "fusion4d_core/fusion4d_core.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "CMotionFieldEstimationF2F.h"
#include "CameraView.h"
#include "Fusion4DGPUKeyFrames.h"
#include "FusionConfig.h"
#include "Peabody.h"

namespace fusion4d_core {
namespace {

BoundingBox3D to_legacy_bbox(const BoundingBox& bbox) {
  return BoundingBox3D(
      bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax, bbox.zmin, bbox.zmax);
}

void validate_image(const ImageView& image, PixelFormat expected,
                    int expected_width, int expected_height,
                    const char* label) {
  if (image.data == nullptr) {
    throw std::invalid_argument(std::string(label) + " data is null");
  }
  if (image.format != expected) {
    throw std::invalid_argument(std::string(label) + " format mismatch");
  }
  if (image.width != expected_width || image.height != expected_height) {
    throw std::invalid_argument(std::string(label) + " dimensions mismatch");
  }
  if (image.stride_bytes <= 0) {
    throw std::invalid_argument(std::string(label) + " stride must be positive");
  }
}

cv::Mat wrap_depth_image(const ImageView& image) {
  return cv::Mat(
      image.height, image.width, CV_16UC1, const_cast<void*>(image.data),
      static_cast<size_t>(image.stride_bytes));
}

cv::Mat wrap_color_image(const ImageView& image, int expected_width,
                         int expected_height) {
  validate_image(image, PixelFormat::color_bgr8, expected_width, expected_height,
                 "color image");
  return cv::Mat(
      image.height, image.width, CV_8UC3, const_cast<void*>(image.data),
      static_cast<size_t>(image.stride_bytes));
}

void configure_global_fusion(const std::string& config_path,
                             int camera_count,
                             int frame_count_key_volume) {
  DEPTH_CAMERAS_NUM = camera_count;
  FrameCountKeyVolume = frame_count_key_volume;

  if (config_path.empty()) {
    return;
  }

  if (FusionConfig::current() == nullptr) {
    if (!FusionConfig::initialize_config(config_path)) {
      throw std::runtime_error("Failed to load config: " + config_path);
    }
  } else if (FusionConfig::get_config_path() != config_path) {
    if (!FusionConfig::reload(config_path)) {
      throw std::runtime_error("Failed to reload config: " + config_path);
    }
  }

  if (auto* conf = FusionConfig::current()) {
    DEPTH_CAMERAS_NUM = conf->GetValueWithDefault<int>(
        "DepthGeneration", "DepthCameraCount", DEPTH_CAMERAS_NUM);
    FrameCountKeyVolume = conf->GetValueWithDefault<int>(
        "Fusion", "FrameCountKeyVolume", FrameCountKeyVolume);
  }

  if (DEPTH_CAMERAS_NUM != camera_count) {
    throw std::runtime_error(
        "Config DepthCameraCount differs from SessionConfig.camera_count");
  }
}

template <typename FusionT>
void apply_common_config(FusionT& fusion, CConfig* conf, const char* prefix) {
  if (!conf) {
    return;
  }

  conf->SetErrorIfNameNotFound(false);
  fusion.align_mu = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_align_mu").c_str(), fusion.align_mu);
  fusion.ed_nodes_res = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_ed_nodes_res").c_str(),
      fusion.ed_nodes_res);
  fusion.vxl_res = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_vxl_res").c_str(), fusion.vxl_res);
  fusion.ed_nodes_res_low = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_ed_nodes_res_low").c_str(),
      fusion.ed_nodes_res_low);
  fusion.vxl_res_low = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_vxl_res_low").c_str(),
      fusion.vxl_res_low);
  fusion.fusion_mu = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_fusion_mu").c_str(), fusion.fusion_mu);
  fusion.iso_surface_level = conf->GetValueWithDefault<float>(
      "Fusion", (std::string(prefix) + "_iso_surface_level").c_str(),
      fusion.iso_surface_level);
  fusion.bBilateralFilter = conf->GetValueWithDefault<bool>(
      "Fusion", (std::string(prefix) + "_bBilateralFilter").c_str(),
      fusion.bBilateralFilter);
  fusion.bUseDepthTopBitAsSeg = conf->GetValueWithDefault<bool>(
      "Fusion", (std::string(prefix) + "_bUseDepthTopBitAsSeg").c_str(),
      fusion.bUseDepthTopBitAsSeg);
}

MeshSnapshot surface_to_snapshot(const CSurface<float>& surface,
                                 std::uint64_t frame_index) {
  MeshSnapshot snapshot;
  snapshot.frame_index = frame_index;
  snapshot.has_normals = surface.haveNormalInfo();
  snapshot.has_colors = surface.haveColorInfo();
  snapshot.positions.reserve(static_cast<size_t>(surface.vtNum) * 3U);
  if (snapshot.has_normals) {
    snapshot.normals.reserve(static_cast<size_t>(surface.vtNum) * 3U);
  }
  if (snapshot.has_colors) {
    snapshot.colors.reserve(static_cast<size_t>(surface.vtNum) * 3U);
  }
  if (surface.triNum > 0 && surface.triangles != nullptr) {
    snapshot.indices.assign(
        surface.triangles,
        surface.triangles + static_cast<size_t>(surface.triNum) * 3U);
  }

  for (int vertex_index = 0; vertex_index < surface.vtNum; ++vertex_index) {
    const float* position = surface.vt_data_block(vertex_index);
    snapshot.positions.insert(
        snapshot.positions.end(), position, position + 3);

    if (snapshot.has_normals) {
      const float* normal = surface.vt_normal(vertex_index);
      snapshot.normals.insert(snapshot.normals.end(), normal, normal + 3);
    }

    if (snapshot.has_colors) {
      const float* color = surface.vt_color(vertex_index);
      snapshot.colors.insert(snapshot.colors.end(), color, color + 3);
    }
  }

  return snapshot;
}

}  // namespace

struct Session::Impl {
  explicit Impl(const SessionConfig& session_config)
      : config(session_config),
        fusion(nullptr, config.camera_count, config.depth_width,
               config.depth_height, config.frame_count_key_volume),
        f2f(nullptr, config.camera_count, config.depth_width,
            config.depth_height, config.frame_count_key_volume) {
    if (config.camera_count < 1) {
      throw std::invalid_argument("camera_count must be >= 1");
    }
    if (config.depth_width < 1 || config.depth_height < 1) {
      throw std::invalid_argument("depth dimensions must be positive");
    }
    if (!config.calibration_path.empty() && !config.cameras.empty()) {
      throw std::invalid_argument(
          "Use either calibration_path or explicit cameras, not both");
    }
    if (config.calibration_path.empty() &&
        static_cast<int>(config.cameras.size()) != config.camera_count) {
      throw std::invalid_argument(
          "Explicit camera calibration count must match camera_count");
    }

    configure_global_fusion(
        config.config_path, config.camera_count, config.frame_count_key_volume);

    const cudaError_t status = cudaSetDevice(config.gpu);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
    }

    build_cameras();
    auto camera_ptrs = raw_camera_ptrs();
    fusion.setup_cameras(camera_ptrs);
    f2f.setup_cameras(camera_ptrs);

    auto* conf = FusionConfig::current();
    apply_common_config(fusion, conf, "fusion4d");
    apply_common_config(f2f, conf, "f2f_motion");
  }

  void build_cameras() {
    cameras.reserve(static_cast<size_t>(config.camera_count));
    for (int camera_index = 0; camera_index < config.camera_count; ++camera_index) {
      auto camera =
          std::make_unique<GCameraView>(config.depth_height, config.depth_width);
      if (!config.calibration_path.empty()) {
        if (!camera->LoadCalibrationFrom3DTMFormat(
                config.calibration_path.c_str(), camera_index, "depth", -1.0)) {
          throw std::runtime_error(
              "Cannot load depth calibration camera " +
              std::to_string(camera_index));
        }
      } else {
        const CameraCalibration& calibration =
            config.cameras[static_cast<size_t>(camera_index)];
        camera->SetCalibrationMatrix(
            calibration.intrinsics.data(),
            calibration.rotation.data(),
            calibration.translation.data(),
            nullptr);
      }
      cameras.push_back(std::move(camera));
    }
  }

  std::vector<GCameraView*> raw_camera_ptrs() {
    std::vector<GCameraView*> result;
    result.reserve(cameras.size());
    for (auto& camera : cameras) {
      result.push_back(camera.get());
    }
    return result;
  }

  std::vector<cv::Mat> wrap_depth_images(const FrameSetView& frame) const {
    if (frame.depth_images == nullptr ||
        frame.depth_image_count != config.camera_count) {
      throw std::invalid_argument(
          "depth_images must contain exactly camera_count entries");
    }

    std::vector<cv::Mat> result;
    result.reserve(static_cast<size_t>(frame.depth_image_count));
    for (int index = 0; index < frame.depth_image_count; ++index) {
      const ImageView& image = frame.depth_images[index];
      validate_image(
          image, PixelFormat::depth_u16, config.depth_width,
          config.depth_height, "depth image");
      result.push_back(wrap_depth_image(image));
    }
    return result;
  }

  void feed_optional_color(const FrameSetView& frame) {
    if (frame.color_images == nullptr || frame.color_image_count == 0) {
      return;
    }
    if (frame.color_image_count != config.camera_count) {
      throw std::invalid_argument(
          "color_images must contain exactly camera_count entries");
    }

    std::vector<cv::Mat> colors;
    colors.reserve(static_cast<size_t>(frame.color_image_count));
    for (int index = 0; index < frame.color_image_count; ++index) {
      colors.push_back(wrap_color_image(
          frame.color_images[index], config.depth_width, config.depth_height));
    }

    fusion.vol_fusion.feed_color_textures(colors);
    f2f.vol_fusion.feed_color_textures(colors);
  }

  SessionConfig config;
  Fusion4DKeyFrames fusion;
  CMotionFieldEstimationF2F f2f;
  std::vector<std::unique_ptr<GCameraView>> cameras;
  std::uint64_t last_frame_index = 0;
  SessionStats session_stats{};
};

Session::Session(const SessionConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

Session::~Session() = default;

Session::Session(Session&&) noexcept = default;

Session& Session::operator=(Session&&) noexcept = default;

void Session::submit_frame(const FrameSetView& frame) {
  auto depths = impl_->wrap_depth_images(frame);
  impl_->feed_optional_color(frame);

  if (!impl_->session_stats.initialized) {
    impl_->fusion.set_up_1st_frame(
        depths, to_legacy_bbox(impl_->config.bbox), nullptr,
        static_cast<int>(frame.frame_index));
    impl_->f2f.set_up_1st_frame(
        depths, to_legacy_bbox(impl_->config.bbox), nullptr,
        static_cast<int>(frame.frame_index), false);
    impl_->session_stats.initialized = true;
  } else {
    impl_->f2f.add_a_frame(
        depths, nullptr, static_cast<int>(frame.frame_index), false, false);
    EDNodesParasGPU ed_nodes = impl_->f2f.ed_nodes_for_init();
    impl_->fusion.add_a_frame(
        depths, &ed_nodes, nullptr, static_cast<int>(frame.frame_index), false);
    cudaDeviceSynchronize();
  }

  impl_->last_frame_index = frame.frame_index;
  ++impl_->session_stats.submitted_frames;
}

MeshSnapshot Session::read_surface(SurfaceKind kind) const {
  if (!impl_->session_stats.initialized) {
    throw std::runtime_error("Session has not processed any frames yet");
  }

  CSurface<float> surface;
  switch (kind) {
    case SurfaceKind::frame_to_frame:
      impl_->f2f.read_current_surface(surface);
      break;
    case SurfaceKind::fusion:
      impl_->fusion.read_current_surface(surface);
      break;
  }

  return surface_to_snapshot(surface, impl_->last_frame_index);
}

SessionStats Session::stats() const noexcept { return impl_->session_stats; }

}  // namespace fusion4d_core
