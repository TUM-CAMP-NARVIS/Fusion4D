// Copyright (c) Davide Nardi
// Licensed under the MIT license.
#include "Live3DViewer.h"

#include <GL/freeglut.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <utility>

namespace Live3DViewer {
namespace {
enum class ColorMode {
  Camera,
  Normals,
  Phong,
};

struct State {
  bool initialized = false;
  bool closed = false;
  int window = 0;
  int width = 1000;
  int height = 800;
  bool showGraph = false;
  bool showWireframe = false;
  ColorMode colorMode = ColorMode::Camera;
  float rotX = -15.0f;
  float rotY = 25.0f;
  float panX = 0.0f;
  float panY = 0.0f;
  float zoom = 1.0f;
  int mouseButton = -1;
  int mouseX = 0;
  int mouseY = 0;
  MeshFrame mesh;
  GraphFrame graph;
  std::mutex mutex;
};

State g_state;

const char* color_mode_name(ColorMode mode) {
  switch (mode) {
    case ColorMode::Camera:
      return "camera";
    case ColorMode::Normals:
      return "normals";
    case ColorMode::Phong:
      return "phong";
  }
  return "unknown";
}

void normalize(float& x, float& y, float& z) {
  const float len = std::sqrt(x * x + y * y + z * z);
  if (len <= 1.0e-6f) {
    x = 0.0f;
    y = 0.0f;
    z = 1.0f;
    return;
  }
  x /= len;
  y /= len;
  z /= len;
}

void set_vertex_color(size_t vertexIndex, const MeshFrame& mesh,
                      ColorMode mode) {
  const size_t c = vertexIndex * 3;
  const bool hasColor = c + 2 < mesh.colors.size();
  const bool hasNormal = c + 2 < mesh.normals.size();

  if (mode == ColorMode::Normals && hasNormal) {
    float nx = mesh.normals[c];
    float ny = mesh.normals[c + 1];
    float nz = mesh.normals[c + 2];
    normalize(nx, ny, nz);
    glColor3f(0.5f * nx + 0.5f, 0.5f * ny + 0.5f, 0.5f * nz + 0.5f);
    return;
  }

  if (mode == ColorMode::Phong && hasNormal) {
    float nx = mesh.normals[c];
    float ny = mesh.normals[c + 1];
    float nz = mesh.normals[c + 2];
    normalize(nx, ny, nz);

    float lx = -0.35f;
    float ly = 0.45f;
    float lz = 0.82f;
    normalize(lx, ly, lz);
    const float ndotl = std::max(0.0f, nx * lx + ny * ly + nz * lz);
    const float rim =
        std::pow(std::max(0.0f, 1.0f - std::abs(nz)), 2.0f) * 0.18f;
    const float shade = std::min(1.0f, 0.24f + 0.72f * ndotl + rim);

    float r = 0.82f;
    float g = 0.84f;
    float b = 0.88f;
    if (hasColor) {
      r = mesh.colors[c] / 255.0f;
      g = mesh.colors[c + 1] / 255.0f;
      b = mesh.colors[c + 2] / 255.0f;
    }
    glColor3f(std::min(1.0f, r * shade + 0.08f),
              std::min(1.0f, g * shade + 0.08f),
              std::min(1.0f, b * shade + 0.08f));
    return;
  }

  if (hasColor) {
    glColor3ub(mesh.colors[c], mesh.colors[c + 1], mesh.colors[c + 2]);
  } else {
    glColor3f(0.82f, 0.84f, 0.88f);
  }
}

void draw_text(float x, float y, const std::string& text) {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluOrtho2D(0.0, g_state.width, 0.0, g_state.height);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glColor3f(0.9f, 0.9f, 0.9f);
  glRasterPos2f(x, y);
  glutBitmapString(GLUT_BITMAP_HELVETICA_18,
                   reinterpret_cast<const unsigned char*>(text.c_str()));
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void display() {
  MeshFrame mesh;
  GraphFrame graph;
  bool showGraph = true;
  bool showWireframe = false;
  ColorMode colorMode = ColorMode::Camera;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    mesh = g_state.mesh;
    graph = g_state.graph;
    showGraph = g_state.showGraph;
    showWireframe = g_state.showWireframe;
    colorMode = g_state.colorMode;
  }

  glViewport(0, 0, g_state.width, g_state.height);
  glClearColor(0.035f, 0.035f, 0.04f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(
      45.0, static_cast<double>(g_state.width) / std::max(1, g_state.height),
      0.01, 100.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glTranslatef(g_state.panX, g_state.panY, -3.0f);
  glScalef(g_state.zoom * mesh.scale, g_state.zoom * mesh.scale,
           g_state.zoom * mesh.scale);
  glRotatef(g_state.rotX, 1.0f, 0.0f, 0.0f);
  glRotatef(g_state.rotY, 0.0f, 1.0f, 0.0f);
  glTranslatef(-mesh.center[0], -mesh.center[1], -mesh.center[2]);

  glDisable(GL_LIGHTING);
  glLineWidth(2.0f);
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.2f, 0.2f);
  glVertex3f(mesh.center[0], mesh.center[1], mesh.center[2]);
  glVertex3f(mesh.center[0] + 25.0f, mesh.center[1], mesh.center[2]);
  glColor3f(0.2f, 1.0f, 0.2f);
  glVertex3f(mesh.center[0], mesh.center[1], mesh.center[2]);
  glVertex3f(mesh.center[0], mesh.center[1] + 25.0f, mesh.center[2]);
  glColor3f(0.2f, 0.5f, 1.0f);
  glVertex3f(mesh.center[0], mesh.center[1], mesh.center[2]);
  glVertex3f(mesh.center[0], mesh.center[1], mesh.center[2] + 25.0f);
  glEnd();

  if (!mesh.triangles.empty()) {
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i + 2 < mesh.triangles.size(); i += 3) {
      for (int k = 0; k < 3; ++k) {
        const int idx = mesh.triangles[i + k];
        if (idx < 0 || static_cast<size_t>(idx) * 3 + 2 >= mesh.vertices.size())
          continue;
        const size_t v = static_cast<size_t>(idx) * 3;
        set_vertex_color(static_cast<size_t>(idx), mesh, colorMode);
        glVertex3f(mesh.vertices[v], mesh.vertices[v + 1],
                   mesh.vertices[v + 2]);
      }
    }
    glEnd();

    if (showWireframe) {
      glEnable(GL_POLYGON_OFFSET_LINE);
      glPolygonOffset(-1.0f, -1.0f);
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glLineWidth(1.0f);
      glColor3f(0.0f, 0.0f, 0.0f);
      glBegin(GL_TRIANGLES);
      for (size_t i = 0; i + 2 < mesh.triangles.size(); i += 3) {
        for (int k = 0; k < 3; ++k) {
          const int idx = mesh.triangles[i + k];
          if (idx < 0 ||
              static_cast<size_t>(idx) * 3 + 2 >= mesh.vertices.size())
            continue;
          const size_t v = static_cast<size_t>(idx) * 3;
          glVertex3f(mesh.vertices[v], mesh.vertices[v + 1],
                     mesh.vertices[v + 2]);
        }
      }
      glEnd();
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      glDisable(GL_POLYGON_OFFSET_LINE);
    }
  } else {
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i + 2 < mesh.vertices.size(); i += 3) {
      set_vertex_color(i / 3, mesh, colorMode);
      glVertex3f(mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]);
    }
    glEnd();
  }

  if (showGraph && !graph.deformedVertices.empty()) {
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    glColor3f(0.1f, 0.95f, 1.0f);
    for (size_t i = 0; i + 1 < graph.edges.size(); i += 2) {
      const int a = graph.edges[i];
      const int b = graph.edges[i + 1];
      if (a < 0 || b < 0 ||
          static_cast<size_t>(a) * 3 + 2 >= graph.deformedVertices.size() ||
          static_cast<size_t>(b) * 3 + 2 >= graph.deformedVertices.size())
        continue;
      const size_t av = static_cast<size_t>(a) * 3;
      const size_t bv = static_cast<size_t>(b) * 3;
      glVertex3f(graph.deformedVertices[av], graph.deformedVertices[av + 1],
                 graph.deformedVertices[av + 2]);
      glVertex3f(graph.deformedVertices[bv], graph.deformedVertices[bv + 1],
                 graph.deformedVertices[bv + 2]);
    }
    glColor3f(1.0f, 0.55f, 0.05f);
    for (size_t i = 0; i + 2 < graph.restVertices.size() &&
                       i + 2 < graph.deformedVertices.size();
         i += 3) {
      glVertex3f(graph.restVertices[i], graph.restVertices[i + 1],
                 graph.restVertices[i + 2]);
      glVertex3f(graph.deformedVertices[i], graph.deformedVertices[i + 1],
                 graph.deformedVertices[i + 2]);
    }
    glEnd();

    glPointSize(7.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 0.92f, 0.1f);
    for (size_t i = 0; i + 2 < graph.deformedVertices.size(); i += 3)
      glVertex3f(graph.deformedVertices[i], graph.deformedVertices[i + 1],
                 graph.deformedVertices[i + 2]);
    glEnd();
    glEnable(GL_DEPTH_TEST);
  }

  draw_text(12.0f, static_cast<float>(g_state.height - 28),
            "frame " + std::to_string(mesh.frame) + "  vertices " +
                std::to_string(mesh.vertices.size() / 3) + "  triangles " +
                std::to_string(mesh.triangles.size() / 3) + "  graph nodes " +
                std::to_string(graph.deformedVertices.size() / 3) + "  graph " +
                (showGraph ? "on" : "off") + "  wire " +
                (showWireframe ? "on" : "off") + "  color " +
                color_mode_name(colorMode));
  draw_text(12.0f, 18.0f,
            "Left drag: rotate   Right drag: pan   Wheel: zoom   W: mesh wire  "
            " G: graph   C: color   R: reset   Esc/Q: close viewer");
  glutSwapBuffers();
}

void reshape(int width, int height) {
  g_state.width = std::max(1, width);
  g_state.height = std::max(1, height);
}

void close() { g_state.closed = true; }

void keyboard(unsigned char key, int, int) {
  if (key == 27 || key == 'q' || key == 'Q') {
    g_state.closed = true;
    glutHideWindow();
    return;
  }
  if (key == 'r' || key == 'R') {
    g_state.rotX = -15.0f;
    g_state.rotY = 25.0f;
    g_state.panX = 0.0f;
    g_state.panY = 0.0f;
    g_state.zoom = 1.0f;
    glutPostRedisplay();
  }
  if (key == 'w' || key == 'W') {
    g_state.showWireframe = !g_state.showWireframe;
    glutPostRedisplay();
  }
  if (key == 'g' || key == 'G') {
    g_state.showGraph = !g_state.showGraph;
    glutPostRedisplay();
  }
  if (key == 'c' || key == 'C') {
    if (g_state.colorMode == ColorMode::Camera)
      g_state.colorMode = ColorMode::Normals;
    else if (g_state.colorMode == ColorMode::Normals)
      g_state.colorMode = ColorMode::Phong;
    else
      g_state.colorMode = ColorMode::Camera;
    glutPostRedisplay();
  }
}

void mouse(int button, int state, int x, int y) {
  if (button == 3 && state == GLUT_DOWN)
    g_state.zoom *= 1.08f;
  else if (button == 4 && state == GLUT_DOWN)
    g_state.zoom /= 1.08f;
  else if (state == GLUT_DOWN) {
    g_state.mouseButton = button;
    g_state.mouseX = x;
    g_state.mouseY = y;
  } else {
    g_state.mouseButton = -1;
  }
  glutPostRedisplay();
}

void motion(int x, int y) {
  const int dx = x - g_state.mouseX;
  const int dy = y - g_state.mouseY;
  g_state.mouseX = x;
  g_state.mouseY = y;

  if (g_state.mouseButton == GLUT_LEFT_BUTTON) {
    g_state.rotY += dx * 0.4f;
    g_state.rotX += dy * 0.4f;
  } else if (g_state.mouseButton == GLUT_RIGHT_BUTTON) {
    g_state.panX += dx * 0.004f;
    g_state.panY -= dy * 0.004f;
  }
  glutPostRedisplay();
}
}  // namespace

void initialize(int argc, char** argv) {
  if (g_state.initialized) return;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInitWindowSize(g_state.width, g_state.height);
  g_state.window = glutCreateWindow("DeformableFusionDemo 3D live mesh");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutCloseFunc(close);
  g_state.initialized = true;
}

bool is_available() { return g_state.initialized && !g_state.closed; }

void update_mesh(MeshFrame&& frame) {
  if (!is_available()) return;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.mesh = std::move(frame);
  }
  glutSetWindow(g_state.window);
  glutPostRedisplay();
  glutMainLoopEvent();
}

void update_graph(GraphFrame&& graph) {
  if (!is_available()) return;
  {
    std::lock_guard<std::mutex> lock(g_state.mutex);
    g_state.graph = std::move(graph);
  }
  glutSetWindow(g_state.window);
  glutPostRedisplay();
  glutMainLoopEvent();
}
}  // namespace Live3DViewer
