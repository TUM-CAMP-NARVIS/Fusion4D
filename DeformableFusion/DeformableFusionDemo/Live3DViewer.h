// Copyright (c) Davide Nardi
// Licensed under the MIT license.
#pragma once

#include <vector>

namespace Live3DViewer {
struct MeshFrame {
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<unsigned char> colors;
  std::vector<int> triangles;
  float center[3] = {0.0f, 0.0f, 0.0f};
  float scale = 1.0f;
  int frame = 0;
};

struct GraphFrame {
  std::vector<float> restVertices;
  std::vector<float> deformedVertices;
  std::vector<int> edges;
};

void initialize(int argc, char** argv);
bool is_available();
void update_mesh(MeshFrame&& frame);
void update_graph(GraphFrame&& graph);
}  // namespace Live3DViewer
