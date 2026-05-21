// Copyright (c) Davide Nardi
// Licensed under the MIT license.
#include "SurfacePreview.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_map>

namespace fs = std::filesystem;

namespace SurfacePreview {
namespace {
struct ProjectedPoint {
  cv::Point2f imagePt;
  cv::Scalar color;
  float z = 0.0f;
  int vertexIndex = -1;
};

cv::Scalar depth_color(float z, float zMin, float zMax) {
  const float t =
      zMax > zMin ? std::clamp((z - zMin) / (zMax - zMin), 0.0f, 1.0f) : 0.5f;
  return cv::Scalar(255.0f * (1.0f - t),
                    180.0f * (1.0f - std::abs(t - 0.5f) * 2.0f), 255.0f * t);
}

cv::Scalar surface_vertex_color(const float* p, const cv::Mat* colorImage,
                                const Options& opt, int width, int height) {
  if (!colorImage || colorImage->empty() || p[2] <= 1.0e-5f)
    return cv::Scalar(220, 220, 220);

  const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
  const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;
  const int u = static_cast<int>(std::round(opt.fx * p[0] / p[2] + cx));
  const int v = static_cast<int>(std::round(opt.fy * p[1] / p[2] + cy));
  if (u < 0 || u >= colorImage->cols || v < 0 || v >= colorImage->rows)
    return cv::Scalar(220, 220, 220);

  const cv::Vec3b bgr = colorImage->at<cv::Vec3b>(v, u);
  return cv::Scalar(bgr[0], bgr[1], bgr[2]);
}

std::vector<ProjectedPoint> project_points(const Surface& surface,
                                           const cv::Mat* colorImage,
                                           const Options& opt, int width,
                                           int height) {
  float zMin = std::numeric_limits<float>::max();
  float zMax = std::numeric_limits<float>::lowest();
  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    if (p[2] > 0.0f) {
      zMin = std::min(zMin, p[2]);
      zMax = std::max(zMax, p[2]);
    }
  }

  const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
  const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;
  std::vector<ProjectedPoint> points;
  points.reserve(surface.vtNum);
  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    if (p[2] <= 1.0e-5f) continue;

    const float u = static_cast<float>(opt.fx * p[0] / p[2] + cx);
    const float v = static_cast<float>(opt.fy * p[1] / p[2] + cy);
    if (u < 0.0f || u >= width || v < 0.0f || v >= height) continue;

    cv::Scalar color = depth_color(p[2], zMin, zMax);
    if (colorImage && !colorImage->empty()) {
      const cv::Vec3b bgr = colorImage->at<cv::Vec3b>(
          std::clamp(static_cast<int>(std::round(v)), 0, colorImage->rows - 1),
          std::clamp(static_cast<int>(std::round(u)), 0, colorImage->cols - 1));
      color = cv::Scalar(bgr[0], bgr[1], bgr[2]);
    }

    points.push_back({cv::Point2f(u, v), color, p[2], i});
  }
  return points;
}

float vertex_distance_cm(const Surface& surface, int a, int b) {
  const float* pa = &surface.data[static_cast<size_t>(a) * surface.vtDim];
  const float* pb = &surface.data[static_cast<size_t>(b) * surface.vtDim];
  const float dx = pa[0] - pb[0];
  const float dy = pa[1] - pb[1];
  const float dz = pa[2] - pb[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void add_triangle_if_valid(Surface& surface,
                           const std::vector<cv::Point2f>& projectedByVertex,
                           int a, int b, int c, float maxEdgePx,
                           float maxEdgeCm) {
  if (a < 0 || b < 0 || c < 0) return;
  if (a == b || b == c || a == c) return;

  const float abPx =
      static_cast<float>(cv::norm(projectedByVertex[a] - projectedByVertex[b]));
  const float bcPx =
      static_cast<float>(cv::norm(projectedByVertex[b] - projectedByVertex[c]));
  const float caPx =
      static_cast<float>(cv::norm(projectedByVertex[c] - projectedByVertex[a]));
  if (std::max({abPx, bcPx, caPx}) > maxEdgePx) return;

  const float abCm = vertex_distance_cm(surface, a, b);
  const float bcCm = vertex_distance_cm(surface, b, c);
  const float caCm = vertex_distance_cm(surface, c, a);
  if (std::max({abCm, bcCm, caCm}) > maxEdgeCm) return;

  surface.triangles.push_back(a);
  surface.triangles.push_back(b);
  surface.triangles.push_back(c);
}

int point_key(const cv::Point2f& p) {
  const int x = std::clamp(static_cast<int>(std::round(p.x)), 0, 65535);
  const int y = std::clamp(static_cast<int>(std::round(p.y)), 0, 65535);
  return (y << 16) ^ x;
}

void synthesize_image_space_triangles(Surface& surface, const Options& opt,
                                      int width, int height) {
  if (surface.triNum > 0 || surface.vtNum <= 0 || surface.vtDim < 3) return;

  constexpr int cellSizePx = 1;
  const int gridW = (width + cellSizePx - 1) / cellSizePx;
  const int gridH = (height + cellSizePx - 1) / cellSizePx;
  std::vector<int> grid(static_cast<size_t>(gridW) * gridH, -1);
  std::vector<float> gridZ(static_cast<size_t>(gridW) * gridH,
                           std::numeric_limits<float>::max());
  std::vector<cv::Point2f> projectedByVertex(surface.vtNum,
                                             cv::Point2f(-1.0f, -1.0f));

  const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
  const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;

  std::unordered_map<int, int> vertexByPixel;
  float zSum = 0.0f;
  int zCount = 0;
  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    if (p[2] <= 1.0e-5f) continue;

    const float u = static_cast<float>(opt.fx * p[0] / p[2] + cx);
    const float v = static_cast<float>(opt.fy * p[1] / p[2] + cy);
    if (u < 0.0f || u >= width || v < 0.0f || v >= height) continue;
    const cv::Point2f roundedPt(static_cast<float>(std::round(u)),
                                static_cast<float>(std::round(v)));
    projectedByVertex[i] = roundedPt;

    const int gx =
        std::clamp(static_cast<int>(std::round(u / cellSizePx)), 0, gridW - 1);
    const int gy =
        std::clamp(static_cast<int>(std::round(v / cellSizePx)), 0, gridH - 1);
    const size_t gridIdx = static_cast<size_t>(gy) * gridW + gx;
    if (p[2] < gridZ[gridIdx]) {
      gridZ[gridIdx] = p[2];
      grid[gridIdx] = i;
      vertexByPixel[point_key(roundedPt)] = i;
    }
    zSum += p[2];
    ++zCount;
  }

  if (zCount == 0) return;
  const float meanZ = zSum / zCount;
  const float metricPerPixel =
      static_cast<float>(meanZ / std::max(1.0, opt.fx));
  const float maxEdgePx = std::max(8.0f, opt.previewMaxEdge);
  const float maxEdgeCm = std::max(8.0f, maxEdgePx * metricPerPixel * 2.0f);

  surface.triangles.clear();
  surface.triangles.reserve(static_cast<size_t>(vertexByPixel.size()) * 3);

  cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
  for (const auto& item : vertexByPixel) {
    const int vertexIndex = item.second;
    const cv::Point2f& p = projectedByVertex[vertexIndex];
    if (p.x > 0.0f && p.x < width - 1.0f && p.y > 0.0f && p.y < height - 1.0f)
      subdiv.insert(p);
  }

  std::vector<cv::Vec6f> triangles;
  subdiv.getTriangleList(triangles);
  auto inside = [width, height](const cv::Point2f& p) {
    return p.x >= 0.0f && p.x < width && p.y >= 0.0f && p.y < height;
  };
  for (const auto& tri : triangles) {
    const cv::Point2f aPt(tri[0], tri[1]);
    const cv::Point2f bPt(tri[2], tri[3]);
    const cv::Point2f cPt(tri[4], tri[5]);
    if (!inside(aPt) || !inside(bPt) || !inside(cPt)) continue;

    const auto aIt = vertexByPixel.find(point_key(aPt));
    const auto bIt = vertexByPixel.find(point_key(bPt));
    const auto cIt = vertexByPixel.find(point_key(cPt));
    if (aIt == vertexByPixel.end() || bIt == vertexByPixel.end() ||
        cIt == vertexByPixel.end())
      continue;
    add_triangle_if_valid(surface, projectedByVertex, aIt->second, bIt->second,
                          cIt->second, maxEdgePx, maxEdgeCm);
  }
  surface.triNum = static_cast<int>(surface.triangles.size() / 3);
}

bool write_surface_ply(const fs::path& path, const Surface& surface,
                       const cv::Mat* colorImage, const Options& opt, int width,
                       int height) {
  std::ofstream out(path);
  if (!out) return false;

  out << "ply\nformat ascii 1.0\n";
  out << "element vertex " << surface.vtNum << "\n";
  out << "property float x\nproperty float y\nproperty float z\n";
  out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
  out << "element face " << surface.triNum << "\n";
  out << "property list uchar int vertex_indices\n";
  out << "end_header\n";

  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    const cv::Scalar bgr =
        surface_vertex_color(p, colorImage, opt, width, height);
    out << p[0] << " " << p[1] << " " << p[2] << " " << static_cast<int>(bgr[2])
        << " " << static_cast<int>(bgr[1]) << " " << static_cast<int>(bgr[0])
        << "\n";
  }

  for (int i = 0; i < surface.triNum; ++i) {
    out << "3 " << surface.triangles[static_cast<size_t>(i) * 3] << " "
        << surface.triangles[static_cast<size_t>(i) * 3 + 1] << " "
        << surface.triangles[static_cast<size_t>(i) * 3 + 2] << "\n";
  }

  return true;
}

bool write_surface_obj(const fs::path& path, const Surface& surface,
                       const cv::Mat* colorImage, const Options& opt, int width,
                       int height) {
  std::ofstream out(path);
  if (!out) return false;

  out << "# DeformableFusionDemo F2F surface export\n";
  out << "# Coordinates are in centimeters.\n";
  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    const cv::Scalar bgr =
        surface_vertex_color(p, colorImage, opt, width, height);
    out << "v " << p[0] << " " << p[1] << " " << p[2] << " " << bgr[2] / 255.0
        << " " << bgr[1] / 255.0 << " " << bgr[0] / 255.0 << "\n";
  }

  for (int i = 0; i < surface.triNum; ++i) {
    out << "f " << surface.triangles[static_cast<size_t>(i) * 3] + 1 << " "
        << surface.triangles[static_cast<size_t>(i) * 3 + 1] + 1 << " "
        << surface.triangles[static_cast<size_t>(i) * 3 + 2] + 1 << "\n";
  }

  return true;
}

void draw_camera_rasterized_mesh(cv::Mat& canvas,
                                 const std::vector<ProjectedPoint>& points,
                                 const cv::Mat* colorImage, int ox, int oy,
                                 int width, int height, float maxEdge) {
  cv::Mat view(height, width, CV_8UC3, cv::Scalar(8, 8, 8));
  if (colorImage && !colorImage->empty()) {
    cv::Mat background;
    if (colorImage->cols != width || colorImage->rows != height)
      cv::resize(*colorImage, background, cv::Size(width, height), 0.0, 0.0,
                 cv::INTER_LINEAR);
    else
      background = *colorImage;
    cv::addWeighted(background, 0.35, view, 0.65, 0.0, view);
  }
  auto key_for = [](const cv::Point2f& p) -> int {
    const int x = std::clamp(static_cast<int>(std::round(p.x)), 0, 65535);
    const int y = std::clamp(static_cast<int>(std::round(p.y)), 0, 65535);
    return (y << 16) ^ x;
  };

  if (points.size() >= 3) {
    cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
    const size_t maxPoints = 9000;
    const size_t stride = std::max<size_t>(1, points.size() / maxPoints);
    std::unordered_map<int, cv::Scalar> colorsByPixel;
    for (size_t i = 0; i < points.size(); i += stride) {
      const int key = key_for(points[i].imagePt);
      if (colorsByPixel.find(key) != colorsByPixel.end()) continue;
      colorsByPixel[key] = points[i].color;
      subdiv.insert(points[i].imagePt);
    }

    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);
    auto inside = [width, height](const cv::Point2f& p) {
      return p.x >= 0.0f && p.x < width && p.y >= 0.0f && p.y < height;
    };
    for (const auto& tri : triangles) {
      cv::Point2f a(tri[0], tri[1]);
      cv::Point2f b(tri[2], tri[3]);
      cv::Point2f c(tri[4], tri[5]);
      if (!inside(a) || !inside(b) || !inside(c)) continue;

      const float ab = static_cast<float>(cv::norm(a - b));
      const float bc = static_cast<float>(cv::norm(b - c));
      const float ca = static_cast<float>(cv::norm(c - a));
      if (std::max({ab, bc, ca}) > maxEdge) continue;

      const auto caIt = colorsByPixel.find(key_for(a));
      const auto cbIt = colorsByPixel.find(key_for(b));
      const auto ccIt = colorsByPixel.find(key_for(c));
      if (caIt == colorsByPixel.end() || cbIt == colorsByPixel.end() ||
          ccIt == colorsByPixel.end())
        continue;

      const cv::Scalar color =
          (caIt->second + cbIt->second + ccIt->second) * (1.0 / 3.0);
      std::vector<cv::Point> poly = {
          cv::Point(static_cast<int>(std::round(a.x)),
                    static_cast<int>(std::round(a.y))),
          cv::Point(static_cast<int>(std::round(b.x)),
                    static_cast<int>(std::round(b.y))),
          cv::Point(static_cast<int>(std::round(c.x)),
                    static_cast<int>(std::round(c.y)))};
      cv::fillConvexPoly(view, poly, color, cv::LINE_AA);
      cv::polylines(view, poly, true, cv::Scalar(25, 25, 25), 1, cv::LINE_AA);
    }
  }

  for (const auto& p : points)
    cv::circle(view, p.imagePt, 1, p.color, cv::FILLED, cv::LINE_AA);

  cv::rectangle(canvas, cv::Rect(ox, oy, width, height), cv::Scalar(55, 55, 55),
                1);
  view.copyTo(canvas(cv::Rect(ox, oy, width, height)));
  cv::putText(canvas, "camera rasterized mesh", cv::Point(ox + 8, oy + 24),
              cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(245, 245, 245), 1,
              cv::LINE_AA);
}

void draw_surface_projection(cv::Mat& canvas, const Surface& surface, int ox,
                             int oy, int size, int ax0, int ax1,
                             const char* label) {
  float minv[3] = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
  float maxv[3] = {std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};

  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    for (int a = 0; a < 3; ++a) {
      minv[a] = std::min(minv[a], p[a]);
      maxv[a] = std::max(maxv[a], p[a]);
    }
  }

  cv::rectangle(canvas, cv::Rect(ox, oy, size, size), cv::Scalar(55, 55, 55),
                1);
  cv::putText(canvas, label, cv::Point(ox + 8, oy + 22),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(230, 230, 230), 1,
              cv::LINE_AA);

  const float range0 = std::max(1.0e-5f, maxv[ax0] - minv[ax0]);
  const float range1 = std::max(1.0e-5f, maxv[ax1] - minv[ax1]);
  const float scale = 0.88f * size / std::max(range0, range1);
  const float center0 = 0.5f * (minv[ax0] + maxv[ax0]);
  const float center1 = 0.5f * (minv[ax1] + maxv[ax1]);

  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    const int x = ox + size / 2 + static_cast<int>((p[ax0] - center0) * scale);
    const int y = oy + size / 2 - static_cast<int>((p[ax1] - center1) * scale);
    if (x >= ox && x < ox + size && y >= oy && y < oy + size)
      cv::circle(canvas, cv::Point(x, y), 1,
                 depth_color(p[2], minv[2], maxv[2]), cv::FILLED, cv::LINE_AA);
  }
}
}  // namespace

Surface from_csurface(const CSurface<float>& surface) {
  Surface preview;
  preview.vtDim = surface.vtDim;
  preview.vtNum = surface.vtNum;
  preview.triNum = surface.triNum;
  const size_t valueCount =
      static_cast<size_t>(preview.vtNum) * static_cast<size_t>(preview.vtDim);
  if (surface.vtData && valueCount > 0)
    preview.data.assign(surface.vtData, surface.vtData + valueCount);

  const size_t triangleValueCount = static_cast<size_t>(preview.triNum) * 3;
  if (surface.triangles && triangleValueCount > 0)
    preview.triangles.assign(surface.triangles,
                             surface.triangles + triangleValueCount);
  return preview;
}

bool show_live_preview(const Surface& sourceSurface,
                       const std::string& surfaceLabel,
                       const cv::Mat* colorImage, const Options& opt, int width,
                       int height) {
  Surface surface = sourceSurface;
  synthesize_image_space_triangles(surface, opt, width, height);

  float minv[3] = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
  float maxv[3] = {std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    for (int a = 0; a < 3; ++a) {
      minv[a] = std::min(minv[a], p[a]);
      maxv[a] = std::max(maxv[a], p[a]);
    }
  }

  static bool windowCreated = false;
  const char* windowName = "DeformableFusionDemo live F2F preview";
  if (!windowCreated) {
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1280, 900);
    windowCreated = true;
  }

  const auto projected =
      project_points(surface, colorImage, opt, width, height);
  cv::Mat canvas(900, 1280, CV_8UC3, cv::Scalar(18, 18, 18));
  draw_camera_rasterized_mesh(canvas, projected, colorImage, 20, 60, width,
                              height, opt.previewMaxEdge);
  draw_surface_projection(canvas, surface, 700, 60, 260, 0, 1, "XY");
  draw_surface_projection(canvas, surface, 990, 60, 260, 0, 2, "XZ");
  draw_surface_projection(canvas, surface, 700, 350, 260, 2, 1, "ZY");
  cv::putText(canvas,
              surfaceLabel + "  vertices=" + std::to_string(surface.vtNum) +
                  " triangles=" + std::to_string(surface.triNum) +
                  " projected=" + std::to_string(projected.size()),
              cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.75,
              cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
  char boundsText[256];
  std::snprintf(boundsText, sizeof(boundsText),
                "surface cm x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]  input "
                "bbox x=[%.0f,%.0f] y=[%.0f,%.0f] z=[%.0f,%.0f]",
                minv[0], maxv[0], minv[1], maxv[1], minv[2], maxv[2],
                opt.bbox.x_s, opt.bbox.x_e, opt.bbox.y_s, opt.bbox.y_e,
                opt.bbox.z_s, opt.bbox.z_e);
  cv::putText(canvas, boundsText, cv::Point(20, 650), cv::FONT_HERSHEY_SIMPLEX,
              0.55, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
  cv::putText(
      canvas,
      "Dim RGB background shows camera crop; mesh edge filter is preview-only.",
      cv::Point(20, 675), cv::FONT_HERSHEY_SIMPLEX, 0.55,
      cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
  cv::putText(canvas, "Esc/Q: stop preview and finish process",
              cv::Point(20, 875), cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
  cv::imshow(windowName, canvas);

  const int key = cv::waitKey(1);
  return key != 27 && key != 'q' && key != 'Q';
}

void dump_surface_if_requested(const Surface& sourceSurface,
                               const fs::path& outputDir, int frame,
                               const cv::Mat* colorImage, const Options& opt,
                               int width, int height) {
  if (opt.dumpSurfaceEvery <= 0 || frame % opt.dumpSurfaceEvery != 0) return;

  Surface surface = sourceSurface;
  synthesize_image_space_triangles(surface, opt, width, height);

  fs::create_directories(outputDir);
  char filename[128];
  const std::string format = opt.dumpSurfaceFormat;
  std::snprintf(filename, sizeof(filename), "f2f_surface_%04d.%s", frame,
                format.c_str());
  const fs::path outPath = outputDir / filename;

  bool ok = false;
  if (format == "ply")
    ok = write_surface_ply(outPath, surface, colorImage, opt, width, height);
  else if (format == "obj")
    ok = write_surface_obj(outPath, surface, colorImage, opt, width, height);
  else {
    std::cerr << "Unsupported --dump-surface-format: " << opt.dumpSurfaceFormat
              << "\n";
    return;
  }

  if (ok) {
    std::cout << "Dumped " << outPath << " (" << surface.vtNum << " vertices, "
              << surface.triNum << " triangles)\n";
  }
}

Live3DViewer::MeshFrame make_viewer_mesh(const Surface& sourceSurface,
                                         const cv::Mat* colorImage,
                                         const Options& opt, int width,
                                         int height, int frame) {
  Surface surface = sourceSurface;
  synthesize_image_space_triangles(surface, opt, width, height);

  Live3DViewer::MeshFrame mesh;
  mesh.vertices.reserve(static_cast<size_t>(surface.vtNum) * 3);
  mesh.normals.reserve(static_cast<size_t>(surface.vtNum) * 3);
  mesh.colors.reserve(static_cast<size_t>(surface.vtNum) * 3);

  float minv[3] = {std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
  float maxv[3] = {std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};

  for (int i = 0; i < surface.vtNum; ++i) {
    const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
    mesh.vertices.push_back(p[0]);
    mesh.vertices.push_back(p[1]);
    mesh.vertices.push_back(p[2]);
    if (surface.vtDim >= 6) {
      mesh.normals.push_back(p[3]);
      mesh.normals.push_back(p[4]);
      mesh.normals.push_back(p[5]);
    } else {
      mesh.normals.push_back(0.0f);
      mesh.normals.push_back(0.0f);
      mesh.normals.push_back(1.0f);
    }
    for (int a = 0; a < 3; ++a) {
      minv[a] = std::min(minv[a], p[a]);
      maxv[a] = std::max(maxv[a], p[a]);
    }

    const cv::Scalar bgr =
        surface_vertex_color(p, colorImage, opt, width, height);
    mesh.colors.push_back(static_cast<unsigned char>(
        std::clamp(static_cast<int>(bgr[2]), 0, 255)));
    mesh.colors.push_back(static_cast<unsigned char>(
        std::clamp(static_cast<int>(bgr[1]), 0, 255)));
    mesh.colors.push_back(static_cast<unsigned char>(
        std::clamp(static_cast<int>(bgr[0]), 0, 255)));
  }

  mesh.center[0] = 0.5f * (minv[0] + maxv[0]);
  mesh.center[1] = 0.5f * (minv[1] + maxv[1]);
  mesh.center[2] = 0.5f * (minv[2] + maxv[2]);
  const float dx = maxv[0] - minv[0];
  const float dy = maxv[1] - minv[1];
  const float dz = maxv[2] - minv[2];
  const float radius =
      std::max(1.0f, 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz));
  mesh.triangles = surface.triangles;
  mesh.scale = 1.2f / radius;
  mesh.frame = frame;
  return mesh;
}
}  // namespace SurfacePreview
