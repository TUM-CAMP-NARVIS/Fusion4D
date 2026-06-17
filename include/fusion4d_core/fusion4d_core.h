#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fusion4d_core {

enum class PixelFormat {
  depth_u16,
  color_bgr8,
};

enum class SurfaceKind {
  frame_to_frame,
  fusion,
};

struct ImageView {
  const void* data = nullptr;
  int width = 0;
  int height = 0;
  int stride_bytes = 0;
  PixelFormat format = PixelFormat::depth_u16;
};

struct CameraCalibration {
  std::array<double, 9> intrinsics{
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::array<double, 9> rotation{
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::array<double, 3> translation{0.0, 0.0, 0.0};
};

struct BoundingBox {
  double xmin = -100.0;
  double xmax = 100.0;
  double ymin = -200.0;
  double ymax = 0.0;
  double zmin = -100.0;
  double zmax = 100.0;
};

struct SessionConfig {
  int camera_count = 1;
  int depth_width = 0;
  int depth_height = 0;
  int gpu = 0;
  int frame_count_key_volume = 1;
  BoundingBox bbox{};
  std::vector<CameraCalibration> cameras;
  std::string calibration_path;
  std::string config_path;
};

struct FrameSetView {
  std::uint64_t frame_index = 0;
  const ImageView* depth_images = nullptr;
  int depth_image_count = 0;
  const ImageView* color_images = nullptr;
  int color_image_count = 0;
};

struct MeshSnapshot {
  std::uint64_t frame_index = 0;
  bool has_normals = false;
  bool has_colors = false;
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> colors;
  std::vector<int> indices;
};

struct SessionStats {
  std::uint64_t submitted_frames = 0;
  bool initialized = false;
};

class Session {
 public:
  explicit Session(const SessionConfig& config);
  ~Session();

  Session(Session&&) noexcept;
  Session& operator=(Session&&) noexcept;

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  void submit_frame(const FrameSetView& frame);
  MeshSnapshot read_surface(SurfaceKind kind) const;
  SessionStats stats() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fusion4d_core
