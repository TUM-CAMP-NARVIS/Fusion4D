// Copyright (c) Davide Nardi// Licensed under the MIT license.
#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BoundingBox3D.h"
#include "CSurface.h"
#include "Live3DViewer.h"

namespace SurfacePreview {
struct Options {
  double fx = 525.0;
  double fy = 525.0;
  double cx = -1.0;
  double cy = -1.0;
  float previewMaxEdge = 80.0f;
  int dumpSurfaceEvery = 0;
  std::string dumpSurfaceFormat = "ply";
  BoundingBox3D bbox = BoundingBox3D(-100, 100, -200, 0, -100, 100);
};

struct Surface {
  int vtDim = 0;
  int vtNum = 0;
  int triNum = 0;
  std::vector<float> data;
  std::vector<int> triangles;
};

Surface from_csurface(const CSurface<float>& surface);
bool show_live_preview(const Surface& sourceSurface,
                       const std::string& surfaceLabel,
                       const cv::Mat* colorImage, const Options& opt, int width,
                       int height);
void dump_surface_if_requested(const Surface& sourceSurface,
                               const std::filesystem::path& outputDir,
                               int frame, const cv::Mat* colorImage,
                               const Options& opt, int width, int height);
Live3DViewer::MeshFrame make_viewer_mesh(const Surface& sourceSurface,
                                         const cv::Mat* colorImage,
                                         const Options& opt, int width,
                                         int height, int frame);
}  // namespace SurfacePreview
