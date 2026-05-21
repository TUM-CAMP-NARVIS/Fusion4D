// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "Peabody.h"

#include "CameraView.h"
#include "CMotionFieldEstimationF2F.h"
#include "Fusion4DGPUKeyFrames.h"
#include "FusionConfig.h"

#include <cuda_runtime.h>
#include <GL/freeglut.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

int DEPTH_CAMERAS_NUM = 1;
int FrameCountKeyVolume = 1;

namespace fs = std::filesystem;

namespace
{
struct Options
{
	std::string configPath;
	fs::path inputDir;
	fs::path depthDir;
	fs::path rgbDir;
	fs::path outputDir = "offline_fusion_output";
	std::string depthToken;
	std::string rgbToken;
	std::string calibrationPath;
	int cameraCount = 1;
	int gpu = 0;
	double fx = 525.0;
	double fy = 525.0;
	double cx = -1.0;
	double cy = -1.0;
	bool preview = false;
	int previewEvery = 1;
	float previewMaxEdge = 80.0f;
	bool viewer3d = false;
	int dumpSurfaceEvery = 0;
	std::string dumpSurfaceFormat = "ply";
	BoundingBox3D bbox = BoundingBox3D(-100, 100, -200, 0, -100, 100);
};

struct PreviewSurface
{
	int vtDim = 0;
	int vtNum = 0;
	int triNum = 0;
	std::vector<float> data;
	std::vector<int> triangles;
};

struct PreviewProjectedPoint
{
	cv::Point2f imagePt;
	cv::Scalar color;
	float z = 0.0f;
	int vertexIndex = -1;
};

struct Live3DViewer
{
	bool initialized = false;
	bool closed = false;
	int window = 0;
	int width = 1000;
	int height = 800;
	int frame = 0;
	std::vector<float> vertices;
	std::vector<unsigned char> colors;
	std::vector<int> triangles;
	std::vector<float> graphRestVertices;
	std::vector<float> graphDeformedVertices;
	std::vector<int> graphEdges;
	bool showGraph = true;
	float center[3] = { 0.0f, 0.0f, 0.0f };
	float scale = 1.0f;
	float rotX = -15.0f;
	float rotY = 25.0f;
	float panX = 0.0f;
	float panY = 0.0f;
	float zoom = 1.0f;
	int mouseButton = -1;
	int mouseX = 0;
	int mouseY = 0;
	std::mutex mutex;
};

Live3DViewer g_viewer3d;

bool has_image_extension(const fs::path& path)
{
	const auto ext = path.extension().string();
	std::string lower;
	lower.reserve(ext.size());
	for (char c : ext) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
	return lower == ".png" || lower == ".tif" || lower == ".tiff" || lower == ".bmp" ||
		lower == ".jpg" || lower == ".jpeg" || lower == ".exr";
}

std::string lower_string(std::string text)
{
	for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	return text;
}

bool natural_less(const fs::path& lhs, const fs::path& rhs)
{
	const auto a = lower_string(lhs.filename().string());
	const auto b = lower_string(rhs.filename().string());
	size_t ia = 0;
	size_t ib = 0;
	while (ia < a.size() && ib < b.size())
	{
		if (std::isdigit(static_cast<unsigned char>(a[ia])) &&
			std::isdigit(static_cast<unsigned char>(b[ib])))
		{
			size_t ja = ia;
			size_t jb = ib;
			while (ja < a.size() && std::isdigit(static_cast<unsigned char>(a[ja]))) ++ja;
			while (jb < b.size() && std::isdigit(static_cast<unsigned char>(b[jb]))) ++jb;
			const auto na = std::stoll(a.substr(ia, ja - ia));
			const auto nb = std::stoll(b.substr(ib, jb - ib));
			if (na != nb) return na < nb;
			ia = ja;
			ib = jb;
			continue;
		}
		if (a[ia] != b[ib]) return a[ia] < b[ib];
		++ia;
		++ib;
	}
	return a.size() < b.size();
}

bool read_preview_surface(const fs::path& path, PreviewSurface& surface)
{
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;

	char type = 0;
	in.read(&type, 1);
	if (type != '4' && type != '8') return false;

	int isNormal = 0;
	int isColor = 0;
	in.read(reinterpret_cast<char*>(&surface.vtDim), sizeof(int));
	in.read(reinterpret_cast<char*>(&isNormal), sizeof(int));
	in.read(reinterpret_cast<char*>(&isColor), sizeof(int));
	in.read(reinterpret_cast<char*>(&surface.vtNum), sizeof(int));
	if (!in || surface.vtDim < 3 || surface.vtNum <= 0) return false;

	const size_t valueCount = static_cast<size_t>(surface.vtNum) * static_cast<size_t>(surface.vtDim);
	surface.data.resize(valueCount);
	if (type == '4')
	{
		in.read(reinterpret_cast<char*>(surface.data.data()), valueCount * sizeof(float));
	}
	else
	{
		std::vector<double> tmp(valueCount);
		in.read(reinterpret_cast<char*>(tmp.data()), valueCount * sizeof(double));
		for (size_t i = 0; i < valueCount; ++i) surface.data[i] = static_cast<float>(tmp[i]);
	}
	if (!in) return false;

	in.read(reinterpret_cast<char*>(&surface.triNum), sizeof(int));
	if (!in || surface.triNum < 0) return false;
	surface.triangles.resize(static_cast<size_t>(surface.triNum) * 3);
	if (!surface.triangles.empty())
	{
		in.read(reinterpret_cast<char*>(surface.triangles.data()), surface.triangles.size() * sizeof(int));
	}
	return static_cast<bool>(in);
}

cv::Scalar depth_color(float z, float zMin, float zMax)
{
	const float t = zMax > zMin ? std::clamp((z - zMin) / (zMax - zMin), 0.0f, 1.0f) : 0.5f;
	return cv::Scalar(255.0f * (1.0f - t), 180.0f * (1.0f - std::abs(t - 0.5f) * 2.0f), 255.0f * t);
}

std::vector<PreviewProjectedPoint> project_preview_points(
	const PreviewSurface& surface,
	const cv::Mat* colorImage,
	const Options& opt,
	int width,
	int height)
{
	float zMin = std::numeric_limits<float>::max();
	float zMax = std::numeric_limits<float>::lowest();
	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		if (p[2] > 0.0f)
		{
			zMin = std::min(zMin, p[2]);
			zMax = std::max(zMax, p[2]);
		}
	}

	const double fx = opt.fx;
	const double fy = opt.fy;
	const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
	const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;
	std::vector<PreviewProjectedPoint> points;
	points.reserve(surface.vtNum);
	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		if (p[2] <= 1.0e-5f) continue;

		const float u = static_cast<float>(fx * p[0] / p[2] + cx);
		const float v = static_cast<float>(fy * p[1] / p[2] + cy);
		if (u < 0.0f || u >= width || v < 0.0f || v >= height) continue;

		cv::Scalar color = depth_color(p[2], zMin, zMax);
		if (colorImage && !colorImage->empty())
		{
			const cv::Vec3b bgr = colorImage->at<cv::Vec3b>(
				std::clamp(static_cast<int>(std::round(v)), 0, colorImage->rows - 1),
				std::clamp(static_cast<int>(std::round(u)), 0, colorImage->cols - 1));
			color = cv::Scalar(bgr[0], bgr[1], bgr[2]);
		}

		points.push_back({ cv::Point2f(u, v), color, p[2], i });
	}
	return points;
}

float vertex_distance_cm(const PreviewSurface& surface, int a, int b)
{
	const float* pa = &surface.data[static_cast<size_t>(a) * surface.vtDim];
	const float* pb = &surface.data[static_cast<size_t>(b) * surface.vtDim];
	const float dx = pa[0] - pb[0];
	const float dy = pa[1] - pb[1];
	const float dz = pa[2] - pb[2];
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void add_triangle_if_valid(
	PreviewSurface& surface,
	const std::vector<cv::Point2f>& projectedByVertex,
	int a,
	int b,
	int c,
	float maxEdgePx,
	float maxEdgeCm)
{
	if (a < 0 || b < 0 || c < 0) return;
	if (a == b || b == c || a == c) return;

	const float abPx = static_cast<float>(cv::norm(projectedByVertex[a] - projectedByVertex[b]));
	const float bcPx = static_cast<float>(cv::norm(projectedByVertex[b] - projectedByVertex[c]));
	const float caPx = static_cast<float>(cv::norm(projectedByVertex[c] - projectedByVertex[a]));
	if (std::max({ abPx, bcPx, caPx }) > maxEdgePx) return;

	const float abCm = vertex_distance_cm(surface, a, b);
	const float bcCm = vertex_distance_cm(surface, b, c);
	const float caCm = vertex_distance_cm(surface, c, a);
	if (std::max({ abCm, bcCm, caCm }) > maxEdgeCm) return;

	surface.triangles.push_back(a);
	surface.triangles.push_back(b);
	surface.triangles.push_back(c);
}

int point_key(const cv::Point2f& p)
{
	const int x = std::clamp(static_cast<int>(std::round(p.x)), 0, 65535);
	const int y = std::clamp(static_cast<int>(std::round(p.y)), 0, 65535);
	return (y << 16) ^ x;
}

void synthesize_image_space_triangles(PreviewSurface& surface, const Options& opt, int width, int height)
{
	if (surface.triNum > 0 || surface.vtNum <= 0 || surface.vtDim < 3) return;

	constexpr int cellSizePx = 1;
	const int gridW = (width + cellSizePx - 1) / cellSizePx;
	const int gridH = (height + cellSizePx - 1) / cellSizePx;
	std::vector<int> grid(static_cast<size_t>(gridW) * gridH, -1);
	std::vector<float> gridZ(static_cast<size_t>(gridW) * gridH, std::numeric_limits<float>::max());
	std::vector<cv::Point2f> projectedByVertex(surface.vtNum, cv::Point2f(-1.0f, -1.0f));

	const double fx = opt.fx;
	const double fy = opt.fy;
	const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
	const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;

	std::unordered_map<int, int> vertexByPixel;
	float zSum = 0.0f;
	int zCount = 0;
	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		if (p[2] <= 1.0e-5f) continue;

		const float u = static_cast<float>(fx * p[0] / p[2] + cx);
		const float v = static_cast<float>(fy * p[1] / p[2] + cy);
		if (u < 0.0f || u >= width || v < 0.0f || v >= height) continue;
		const cv::Point2f roundedPt(
			static_cast<float>(std::round(u)),
			static_cast<float>(std::round(v)));
		projectedByVertex[i] = roundedPt;

		const int gx = std::clamp(static_cast<int>(std::round(u / cellSizePx)), 0, gridW - 1);
		const int gy = std::clamp(static_cast<int>(std::round(v / cellSizePx)), 0, gridH - 1);
		const size_t gridIdx = static_cast<size_t>(gy) * gridW + gx;
		if (p[2] < gridZ[gridIdx])
		{
			gridZ[gridIdx] = p[2];
			grid[gridIdx] = i;
			vertexByPixel[point_key(roundedPt)] = i;
		}
		zSum += p[2];
		++zCount;
	}

	if (zCount == 0) return;
	const float meanZ = zSum / zCount;
	const float metricPerPixel = static_cast<float>(meanZ / std::max(1.0, fx));
	const float maxEdgePx = std::max(8.0f, opt.previewMaxEdge);
	const float maxEdgeCm = std::max(8.0f, maxEdgePx * metricPerPixel * 2.0f);

	surface.triangles.clear();
	surface.triangles.reserve(static_cast<size_t>(vertexByPixel.size()) * 3);

	cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
	for (const auto& item : vertexByPixel)
	{
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
	for (const auto& tri : triangles)
	{
		const cv::Point2f aPt(tri[0], tri[1]);
		const cv::Point2f bPt(tri[2], tri[3]);
		const cv::Point2f cPt(tri[4], tri[5]);
		if (!inside(aPt) || !inside(bPt) || !inside(cPt)) continue;

		const auto aIt = vertexByPixel.find(point_key(aPt));
		const auto bIt = vertexByPixel.find(point_key(bPt));
		const auto cIt = vertexByPixel.find(point_key(cPt));
		if (aIt == vertexByPixel.end() || bIt == vertexByPixel.end() || cIt == vertexByPixel.end()) continue;
		add_triangle_if_valid(surface, projectedByVertex, aIt->second, bIt->second, cIt->second, maxEdgePx, maxEdgeCm);
	}
	surface.triNum = static_cast<int>(surface.triangles.size() / 3);
}

cv::Scalar surface_vertex_color(const float* p, const cv::Mat* colorImage, const Options& opt, int width, int height)
{
	if (!colorImage || colorImage->empty() || p[2] <= 1.0e-5f) return cv::Scalar(220, 220, 220);

	const double fx = opt.fx;
	const double fy = opt.fy;
	const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
	const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;
	const int u = static_cast<int>(std::round(fx * p[0] / p[2] + cx));
	const int v = static_cast<int>(std::round(fy * p[1] / p[2] + cy));
	if (u < 0 || u >= colorImage->cols || v < 0 || v >= colorImage->rows) return cv::Scalar(220, 220, 220);

	const cv::Vec3b bgr = colorImage->at<cv::Vec3b>(v, u);
	return cv::Scalar(bgr[0], bgr[1], bgr[2]);
}

bool write_surface_ply(const fs::path& path, const PreviewSurface& surface, const cv::Mat* colorImage, const Options& opt, int width, int height)
{
	std::ofstream out(path);
	if (!out) return false;

	out << "ply\nformat ascii 1.0\n";
	out << "element vertex " << surface.vtNum << "\n";
	out << "property float x\nproperty float y\nproperty float z\n";
	out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
	out << "element face " << surface.triNum << "\n";
	out << "property list uchar int vertex_indices\n";
	out << "end_header\n";

	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		const cv::Scalar bgr = surface_vertex_color(p, colorImage, opt, width, height);
		out << p[0] << " " << p[1] << " " << p[2] << " "
			<< static_cast<int>(bgr[2]) << " " << static_cast<int>(bgr[1]) << " " << static_cast<int>(bgr[0]) << "\n";
	}

	for (int i = 0; i < surface.triNum; ++i)
	{
		out << "3 " << surface.triangles[static_cast<size_t>(i) * 3] << " "
			<< surface.triangles[static_cast<size_t>(i) * 3 + 1] << " "
			<< surface.triangles[static_cast<size_t>(i) * 3 + 2] << "\n";
	}

	return true;
}

bool write_surface_obj(const fs::path& path, const PreviewSurface& surface, const cv::Mat* colorImage, const Options& opt, int width, int height)
{
	std::ofstream out(path);
	if (!out) return false;

	out << "# DeformableFusionDemo F2F surface export\n";
	out << "# Coordinates are in centimeters.\n";
	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		const cv::Scalar bgr = surface_vertex_color(p, colorImage, opt, width, height);
		out << "v " << p[0] << " " << p[1] << " " << p[2] << " "
			<< bgr[2] / 255.0 << " " << bgr[1] / 255.0 << " " << bgr[0] / 255.0 << "\n";
	}

	for (int i = 0; i < surface.triNum; ++i)
	{
		out << "f " << surface.triangles[static_cast<size_t>(i) * 3] + 1 << " "
			<< surface.triangles[static_cast<size_t>(i) * 3 + 1] + 1 << " "
			<< surface.triangles[static_cast<size_t>(i) * 3 + 2] + 1 << "\n";
	}

	return true;
}

void dump_surface_if_requested(const fs::path& surfacePath, const fs::path& outputDir, int frame, const cv::Mat* colorImage, const Options& opt, int width, int height)
{
	if (opt.dumpSurfaceEvery <= 0 || frame % opt.dumpSurfaceEvery != 0) return;

	PreviewSurface surface;
	if (!read_preview_surface(surfacePath, surface)) return;
	synthesize_image_space_triangles(surface, opt, width, height);

	fs::create_directories(outputDir);
	char filename[128];
	const std::string format = lower_string(opt.dumpSurfaceFormat);
	std::snprintf(filename, sizeof(filename), "f2f_surface_%04d.%s", frame, format.c_str());
	const fs::path outPath = outputDir / filename;

	bool ok = false;
	if (format == "ply")
	{
		ok = write_surface_ply(outPath, surface, colorImage, opt, width, height);
	}
	else if (format == "obj")
	{
		ok = write_surface_obj(outPath, surface, colorImage, opt, width, height);
	}
	else
	{
		std::cerr << "Unsupported --dump-surface-format: " << opt.dumpSurfaceFormat << "\n";
		return;
	}

	if (ok)
	{
		std::cout << "Dumped " << outPath << " (" << surface.vtNum << " vertices, " << surface.triNum << " triangles)\n";
	}
}

void draw_camera_rasterized_mesh(
	cv::Mat& canvas,
	const std::vector<PreviewProjectedPoint>& points,
	const cv::Mat* colorImage,
	int ox,
	int oy,
	int width,
	int height,
	float maxEdge)
{
	cv::Mat view(height, width, CV_8UC3, cv::Scalar(8, 8, 8));
	if (colorImage && !colorImage->empty())
	{
		cv::Mat background;
		if (colorImage->cols != width || colorImage->rows != height)
		{
			cv::resize(*colorImage, background, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
		}
		else
		{
			background = *colorImage;
		}
		cv::addWeighted(background, 0.35, view, 0.65, 0.0, view);
	}
	auto key_for = [](const cv::Point2f& p) -> int {
		const int x = std::clamp(static_cast<int>(std::round(p.x)), 0, 65535);
		const int y = std::clamp(static_cast<int>(std::round(p.y)), 0, 65535);
		return (y << 16) ^ x;
	};

	if (points.size() >= 3)
	{
		cv::Subdiv2D subdiv(cv::Rect(0, 0, width, height));
		const size_t maxPoints = 9000;
		const size_t stride = std::max<size_t>(1, points.size() / maxPoints);
		std::unordered_map<int, cv::Scalar> colorsByPixel;
		for (size_t i = 0; i < points.size(); i += stride)
		{
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
		for (const auto& tri : triangles)
		{
			cv::Point2f a(tri[0], tri[1]);
			cv::Point2f b(tri[2], tri[3]);
			cv::Point2f c(tri[4], tri[5]);
			if (!inside(a) || !inside(b) || !inside(c)) continue;

			const float ab = static_cast<float>(cv::norm(a - b));
			const float bc = static_cast<float>(cv::norm(b - c));
			const float ca = static_cast<float>(cv::norm(c - a));
			if (std::max({ ab, bc, ca }) > maxEdge) continue;

			const auto caIt = colorsByPixel.find(key_for(a));
			const auto cbIt = colorsByPixel.find(key_for(b));
			const auto ccIt = colorsByPixel.find(key_for(c));
			if (caIt == colorsByPixel.end() || cbIt == colorsByPixel.end() || ccIt == colorsByPixel.end()) continue;

			const cv::Scalar color = (caIt->second + cbIt->second + ccIt->second) * (1.0 / 3.0);
			std::vector<cv::Point> poly = {
				cv::Point(static_cast<int>(std::round(a.x)), static_cast<int>(std::round(a.y))),
				cv::Point(static_cast<int>(std::round(b.x)), static_cast<int>(std::round(b.y))),
				cv::Point(static_cast<int>(std::round(c.x)), static_cast<int>(std::round(c.y)))
			};
			cv::fillConvexPoly(view, poly, color, cv::LINE_AA);
			cv::polylines(view, poly, true, cv::Scalar(25, 25, 25), 1, cv::LINE_AA);
		}
	}

	for (const auto& p : points)
	{
		cv::circle(view, p.imagePt, 1, p.color, cv::FILLED, cv::LINE_AA);
	}

	cv::rectangle(canvas, cv::Rect(ox, oy, width, height), cv::Scalar(55, 55, 55), 1);
	view.copyTo(canvas(cv::Rect(ox, oy, width, height)));
	cv::putText(canvas, "camera rasterized mesh", cv::Point(ox + 8, oy + 24), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
}

void draw_surface_projection(cv::Mat& canvas, const PreviewSurface& surface, int ox, int oy, int size, int ax0, int ax1, const char* label)
{
	float minv[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
	float maxv[3] = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		for (int a = 0; a < 3; ++a)
		{
			minv[a] = std::min(minv[a], p[a]);
			maxv[a] = std::max(maxv[a], p[a]);
		}
	}

	cv::rectangle(canvas, cv::Rect(ox, oy, size, size), cv::Scalar(55, 55, 55), 1);
	cv::putText(canvas, label, cv::Point(ox + 8, oy + 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(230, 230, 230), 1, cv::LINE_AA);

	const float range0 = std::max(1.0e-5f, maxv[ax0] - minv[ax0]);
	const float range1 = std::max(1.0e-5f, maxv[ax1] - minv[ax1]);
	const float scale = 0.88f * size / std::max(range0, range1);
	const float center0 = 0.5f * (minv[ax0] + maxv[ax0]);
	const float center1 = 0.5f * (minv[ax1] + maxv[ax1]);

	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		const int x = ox + size / 2 + static_cast<int>((p[ax0] - center0) * scale);
		const int y = oy + size / 2 - static_cast<int>((p[ax1] - center1) * scale);
		if (x >= ox && x < ox + size && y >= oy && y < oy + size)
		{
			cv::circle(canvas, cv::Point(x, y), 1, depth_color(p[2], minv[2], maxv[2]), cv::FILLED, cv::LINE_AA);
		}
	}
}

bool show_live_preview(const fs::path& surfacePath, const cv::Mat* colorImage, const Options& opt, int width, int height)
{
	PreviewSurface surface;
	if (!read_preview_surface(surfacePath, surface)) return true;
	synthesize_image_space_triangles(surface, opt, width, height);

	float minv[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
	float maxv[3] = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };
	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		for (int a = 0; a < 3; ++a)
		{
			minv[a] = std::min(minv[a], p[a]);
			maxv[a] = std::max(maxv[a], p[a]);
		}
	}

	static bool windowCreated = false;
	const char* windowName = "DeformableFusionDemo live F2F preview";
	if (!windowCreated)
	{
		cv::namedWindow(windowName, cv::WINDOW_NORMAL);
		cv::resizeWindow(windowName, 1280, 900);
		windowCreated = true;
	}

	const auto projected = project_preview_points(surface, colorImage, opt, width, height);
	cv::Mat canvas(900, 1280, CV_8UC3, cv::Scalar(18, 18, 18));
	draw_camera_rasterized_mesh(canvas, projected, colorImage, 20, 60, width, height, opt.previewMaxEdge);
	draw_surface_projection(canvas, surface, 700, 60, 260, 0, 1, "XY");
	draw_surface_projection(canvas, surface, 990, 60, 260, 0, 2, "XZ");
	draw_surface_projection(canvas, surface, 700, 350, 260, 2, 1, "ZY");
	cv::putText(canvas, surfacePath.filename().string() + "  vertices=" + std::to_string(surface.vtNum) +
		" triangles=" + std::to_string(surface.triNum) + " projected=" + std::to_string(projected.size()),
		cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(245, 245, 245), 1, cv::LINE_AA);
	char boundsText[256];
	std::snprintf(boundsText, sizeof(boundsText), "surface cm x=[%.1f,%.1f] y=[%.1f,%.1f] z=[%.1f,%.1f]  input bbox x=[%.0f,%.0f] y=[%.0f,%.0f] z=[%.0f,%.0f]",
		minv[0], maxv[0], minv[1], maxv[1], minv[2], maxv[2],
		opt.bbox.x_s, opt.bbox.x_e, opt.bbox.y_s, opt.bbox.y_e, opt.bbox.z_s, opt.bbox.z_e);
	cv::putText(canvas, boundsText, cv::Point(20, 650), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
	cv::putText(canvas, "Dim RGB background shows camera crop; mesh edge filter is preview-only.",
		cv::Point(20, 675), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
	cv::putText(canvas, "Esc/Q: stop preview and finish process",
		cv::Point(20, 875), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
	cv::imshow(windowName, canvas);

	const int key = cv::waitKey(1);
	return key != 27 && key != 'q' && key != 'Q';
}

void draw_viewer_text(float x, float y, const std::string& text)
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, g_viewer3d.width, 0.0, g_viewer3d.height);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(0.9f, 0.9f, 0.9f);
	glRasterPos2f(x, y);
	glutBitmapString(GLUT_BITMAP_HELVETICA_18, reinterpret_cast<const unsigned char*>(text.c_str()));
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

void display_live_3d_viewer()
{
	std::vector<float> vertices;
	std::vector<unsigned char> colors;
	std::vector<int> triangles;
	std::vector<float> graphRestVertices;
	std::vector<float> graphDeformedVertices;
	std::vector<int> graphEdges;
	bool showGraph = true;
	float center[3];
	float scale = 1.0f;
	int frame = 0;
	{
		std::lock_guard<std::mutex> lock(g_viewer3d.mutex);
		vertices = g_viewer3d.vertices;
		colors = g_viewer3d.colors;
		triangles = g_viewer3d.triangles;
		graphRestVertices = g_viewer3d.graphRestVertices;
		graphDeformedVertices = g_viewer3d.graphDeformedVertices;
		graphEdges = g_viewer3d.graphEdges;
		showGraph = g_viewer3d.showGraph;
		center[0] = g_viewer3d.center[0];
		center[1] = g_viewer3d.center[1];
		center[2] = g_viewer3d.center[2];
		scale = g_viewer3d.scale;
		frame = g_viewer3d.frame;
	}

	glViewport(0, 0, g_viewer3d.width, g_viewer3d.height);
	glClearColor(0.035f, 0.035f, 0.04f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, static_cast<double>(g_viewer3d.width) / std::max(1, g_viewer3d.height), 0.01, 100.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(g_viewer3d.panX, g_viewer3d.panY, -3.0f);
	glScalef(g_viewer3d.zoom * scale, g_viewer3d.zoom * scale, g_viewer3d.zoom * scale);
	glRotatef(g_viewer3d.rotX, 1.0f, 0.0f, 0.0f);
	glRotatef(g_viewer3d.rotY, 0.0f, 1.0f, 0.0f);
	glTranslatef(-center[0], -center[1], -center[2]);

	glDisable(GL_LIGHTING);
	glLineWidth(2.0f);
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.2f, 0.2f); glVertex3f(center[0], center[1], center[2]); glVertex3f(center[0] + 25.0f, center[1], center[2]);
	glColor3f(0.2f, 1.0f, 0.2f); glVertex3f(center[0], center[1], center[2]); glVertex3f(center[0], center[1] + 25.0f, center[2]);
	glColor3f(0.2f, 0.5f, 1.0f); glVertex3f(center[0], center[1], center[2]); glVertex3f(center[0], center[1], center[2] + 25.0f);
	glEnd();

	if (!triangles.empty())
	{
		glBegin(GL_TRIANGLES);
		for (size_t i = 0; i + 2 < triangles.size(); i += 3)
		{
			for (int k = 0; k < 3; ++k)
			{
				const int idx = triangles[i + k];
				if (idx < 0 || static_cast<size_t>(idx) * 3 + 2 >= vertices.size()) continue;
				const size_t c = static_cast<size_t>(idx) * 3;
				glColor3ub(colors[c], colors[c + 1], colors[c + 2]);
				glVertex3f(vertices[c], vertices[c + 1], vertices[c + 2]);
			}
		}
		glEnd();

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3f(0.02f, 0.02f, 0.02f);
		glBegin(GL_TRIANGLES);
		for (size_t i = 0; i + 2 < triangles.size(); i += 3)
		{
			for (int k = 0; k < 3; ++k)
			{
				const int idx = triangles[i + k];
				if (idx < 0 || static_cast<size_t>(idx) * 3 + 2 >= vertices.size()) continue;
				const size_t v = static_cast<size_t>(idx) * 3;
				glVertex3f(vertices[v], vertices[v + 1], vertices[v + 2]);
			}
		}
		glEnd();
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else
	{
		glPointSize(2.0f);
		glBegin(GL_POINTS);
		for (size_t i = 0; i + 2 < vertices.size(); i += 3)
		{
			glColor3ub(colors[i], colors[i + 1], colors[i + 2]);
			glVertex3f(vertices[i], vertices[i + 1], vertices[i + 2]);
		}
		glEnd();
	}

	if (showGraph && !graphDeformedVertices.empty())
	{
		glDisable(GL_DEPTH_TEST);
		glLineWidth(2.0f);
		glBegin(GL_LINES);
		glColor3f(0.1f, 0.95f, 1.0f);
		for (size_t i = 0; i + 1 < graphEdges.size(); i += 2)
		{
			const int a = graphEdges[i];
			const int b = graphEdges[i + 1];
			if (a < 0 || b < 0 ||
				static_cast<size_t>(a) * 3 + 2 >= graphDeformedVertices.size() ||
				static_cast<size_t>(b) * 3 + 2 >= graphDeformedVertices.size()) continue;
			const size_t av = static_cast<size_t>(a) * 3;
			const size_t bv = static_cast<size_t>(b) * 3;
			glVertex3f(graphDeformedVertices[av], graphDeformedVertices[av + 1], graphDeformedVertices[av + 2]);
			glVertex3f(graphDeformedVertices[bv], graphDeformedVertices[bv + 1], graphDeformedVertices[bv + 2]);
		}
		glColor3f(1.0f, 0.55f, 0.05f);
		for (size_t i = 0; i + 2 < graphRestVertices.size() && i + 2 < graphDeformedVertices.size(); i += 3)
		{
			glVertex3f(graphRestVertices[i], graphRestVertices[i + 1], graphRestVertices[i + 2]);
			glVertex3f(graphDeformedVertices[i], graphDeformedVertices[i + 1], graphDeformedVertices[i + 2]);
		}
		glEnd();

		glPointSize(7.0f);
		glBegin(GL_POINTS);
		glColor3f(1.0f, 0.92f, 0.1f);
		for (size_t i = 0; i + 2 < graphDeformedVertices.size(); i += 3)
			glVertex3f(graphDeformedVertices[i], graphDeformedVertices[i + 1], graphDeformedVertices[i + 2]);
		glEnd();
		glEnable(GL_DEPTH_TEST);
	}

	draw_viewer_text(12.0f, static_cast<float>(g_viewer3d.height - 28),
		"frame " + std::to_string(frame) + "  vertices " + std::to_string(vertices.size() / 3) +
		"  triangles " + std::to_string(triangles.size() / 3) +
		"  graph nodes " + std::to_string(graphDeformedVertices.size() / 3));
	draw_viewer_text(12.0f, 18.0f, "Left drag: rotate   Right drag: pan   Wheel: zoom   G: graph   R: reset   Esc/Q: close viewer");
	glutSwapBuffers();
}

void reshape_live_3d_viewer(int width, int height)
{
	g_viewer3d.width = std::max(1, width);
	g_viewer3d.height = std::max(1, height);
}

void close_live_3d_viewer()
{
	g_viewer3d.closed = true;
}

void keyboard_live_3d_viewer(unsigned char key, int, int)
{
	if (key == 27 || key == 'q' || key == 'Q')
	{
		g_viewer3d.closed = true;
		glutHideWindow();
		return;
	}
	if (key == 'r' || key == 'R')
	{
		g_viewer3d.rotX = -15.0f;
		g_viewer3d.rotY = 25.0f;
		g_viewer3d.panX = 0.0f;
		g_viewer3d.panY = 0.0f;
		g_viewer3d.zoom = 1.0f;
		glutPostRedisplay();
	}
	if (key == 'g' || key == 'G')
	{
		g_viewer3d.showGraph = !g_viewer3d.showGraph;
		glutPostRedisplay();
	}
}

void mouse_live_3d_viewer(int button, int state, int x, int y)
{
	if (button == 3 && state == GLUT_DOWN) g_viewer3d.zoom *= 1.08f;
	else if (button == 4 && state == GLUT_DOWN) g_viewer3d.zoom /= 1.08f;
	else if (state == GLUT_DOWN)
	{
		g_viewer3d.mouseButton = button;
		g_viewer3d.mouseX = x;
		g_viewer3d.mouseY = y;
	}
	else
	{
		g_viewer3d.mouseButton = -1;
	}
	glutPostRedisplay();
}

void motion_live_3d_viewer(int x, int y)
{
	const int dx = x - g_viewer3d.mouseX;
	const int dy = y - g_viewer3d.mouseY;
	g_viewer3d.mouseX = x;
	g_viewer3d.mouseY = y;

	if (g_viewer3d.mouseButton == GLUT_LEFT_BUTTON)
	{
		g_viewer3d.rotY += dx * 0.4f;
		g_viewer3d.rotX += dy * 0.4f;
	}
	else if (g_viewer3d.mouseButton == GLUT_RIGHT_BUTTON)
	{
		g_viewer3d.panX += dx * 0.004f;
		g_viewer3d.panY -= dy * 0.004f;
	}
	glutPostRedisplay();
}

void initialize_live_3d_viewer(int argc, char** argv)
{
	if (g_viewer3d.initialized) return;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(g_viewer3d.width, g_viewer3d.height);
	g_viewer3d.window = glutCreateWindow("DeformableFusionDemo 3D live mesh");
	glutDisplayFunc(display_live_3d_viewer);
	glutReshapeFunc(reshape_live_3d_viewer);
	glutKeyboardFunc(keyboard_live_3d_viewer);
	glutMouseFunc(mouse_live_3d_viewer);
	glutMotionFunc(motion_live_3d_viewer);
	glutCloseFunc(close_live_3d_viewer);
	g_viewer3d.initialized = true;
}

void update_live_3d_viewer(const fs::path& surfacePath, const cv::Mat* colorImage, const Options& opt, int width, int height, int frame)
{
	if (!opt.viewer3d || !g_viewer3d.initialized || g_viewer3d.closed) return;

	PreviewSurface surface;
	if (!read_preview_surface(surfacePath, surface)) return;
	synthesize_image_space_triangles(surface, opt, width, height);

	std::vector<float> vertices;
	std::vector<unsigned char> colors;
	vertices.reserve(static_cast<size_t>(surface.vtNum) * 3);
	colors.reserve(static_cast<size_t>(surface.vtNum) * 3);

	float minv[3] = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
	float maxv[3] = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

	for (int i = 0; i < surface.vtNum; ++i)
	{
		const float* p = &surface.data[static_cast<size_t>(i) * surface.vtDim];
		vertices.push_back(p[0]);
		vertices.push_back(p[1]);
		vertices.push_back(p[2]);
		for (int a = 0; a < 3; ++a)
		{
			minv[a] = std::min(minv[a], p[a]);
			maxv[a] = std::max(maxv[a], p[a]);
		}

		const cv::Scalar bgr = surface_vertex_color(p, colorImage, opt, width, height);
		colors.push_back(static_cast<unsigned char>(std::clamp(static_cast<int>(bgr[2]), 0, 255)));
		colors.push_back(static_cast<unsigned char>(std::clamp(static_cast<int>(bgr[1]), 0, 255)));
		colors.push_back(static_cast<unsigned char>(std::clamp(static_cast<int>(bgr[0]), 0, 255)));
	}

	float center[3] = {
		0.5f * (minv[0] + maxv[0]),
		0.5f * (minv[1] + maxv[1]),
		0.5f * (minv[2] + maxv[2])
	};
	const float dx = maxv[0] - minv[0];
	const float dy = maxv[1] - minv[1];
	const float dz = maxv[2] - minv[2];
	const float radius = std::max(1.0f, 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz));

	{
		std::lock_guard<std::mutex> lock(g_viewer3d.mutex);
		g_viewer3d.vertices.swap(vertices);
		g_viewer3d.colors.swap(colors);
		g_viewer3d.triangles = surface.triangles;
		g_viewer3d.center[0] = center[0];
		g_viewer3d.center[1] = center[1];
		g_viewer3d.center[2] = center[2];
		g_viewer3d.scale = 1.2f / radius;
		g_viewer3d.frame = frame;
	}

	glutSetWindow(g_viewer3d.window);
	glutPostRedisplay();
	glutMainLoopEvent();
}

void update_live_3d_graph(EDNodesParasGPU edNodes, const Options& opt)
{
	if (!opt.viewer3d || !g_viewer3d.initialized || g_viewer3d.closed) return;

	int nodesNum = 0;
	checkCudaErrors(cudaMemcpy(&nodesNum, edNodes.ed_nodes_num_gpu.dev_ptr, sizeof(int), cudaMemcpyDeviceToHost));
	nodesNum = std::clamp(nodesNum, 0, edNodes.ed_nodes_num_gpu.max_size);
	if (nodesNum <= 0) return;

	std::vector<DeformGraphNodeCuda> nodes(nodesNum);
	checkCudaErrors(cudaMemcpy(nodes.data(), edNodes.dev_ed_nodes, sizeof(DeformGraphNodeCuda) * nodesNum, cudaMemcpyDeviceToHost));

	std::vector<float> restVertices;
	std::vector<float> deformedVertices;
	restVertices.reserve(static_cast<size_t>(nodesNum) * 3);
	deformedVertices.reserve(static_cast<size_t>(nodesNum) * 3);
	for (const auto& node : nodes)
	{
		const float gx = node.g[0];
		const float gy = node.g[1];
		const float gz = node.g[2];
		restVertices.push_back(gx);
		restVertices.push_back(gy);
		restVertices.push_back(gz);
		deformedVertices.push_back(gx + node.t[0]);
		deformedVertices.push_back(gy + node.t[1]);
		deformedVertices.push_back(gz + node.t[2]);
	}

	std::vector<int> edges;
	std::unordered_set<unsigned long long> seen;
	for (int i = 0; i < nodesNum; ++i)
	{
		for (int k = 0; k < EDNODE_NN_MAX; ++k)
		{
			const int j = nodes[i].neighbors[k];
			if (j < 0 || j >= nodesNum || j == i) continue;
			const int a = std::min(i, j);
			const int b = std::max(i, j);
			const unsigned long long key = (static_cast<unsigned long long>(a) << 32) | static_cast<unsigned int>(b);
			if (seen.insert(key).second)
			{
				edges.push_back(a);
				edges.push_back(b);
			}
		}
	}

	{
		std::lock_guard<std::mutex> lock(g_viewer3d.mutex);
		g_viewer3d.graphRestVertices.swap(restVertices);
		g_viewer3d.graphDeformedVertices.swap(deformedVertices);
		g_viewer3d.graphEdges.swap(edges);
	}

	glutSetWindow(g_viewer3d.window);
	glutPostRedisplay();
	glutMainLoopEvent();
}

std::vector<fs::path> list_images(const fs::path& dir, const std::string& token)
{
	if (dir.empty() || !fs::is_directory(dir))
	{
		throw std::runtime_error("Directory not found: " + dir.string());
	}

	const auto tokenLower = lower_string(token);
	std::vector<fs::path> files;
	for (const auto& entry : fs::directory_iterator(dir))
	{
		if (!entry.is_regular_file() || !has_image_extension(entry.path())) continue;
		const auto filename = lower_string(entry.path().filename().string());
		if (!tokenLower.empty() && filename.find(tokenLower) == std::string::npos) continue;
		files.push_back(entry.path());
	}
	std::sort(files.begin(), files.end(), natural_less);
	return files;
}

cv::Mat load_depth(const fs::path& path)
{
	cv::Mat depth = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
	if (depth.empty()) throw std::runtime_error("Cannot read depth image: " + path.string());
	if (depth.channels() != 1) throw std::runtime_error("Depth image must be single-channel: " + path.string());
	if (depth.type() == CV_16UC1) return depth;

	cv::Mat converted;
	if (depth.type() == CV_32FC1)
	{
		depth.convertTo(converted, CV_16UC1);
		return converted;
	}
	if (depth.type() == CV_8UC1)
	{
		depth.convertTo(converted, CV_16UC1, 256.0);
		return converted;
	}

	throw std::runtime_error("Unsupported depth image type in: " + path.string());
}

cv::Mat load_color(const fs::path& path, int width, int height)
{
	cv::Mat color = cv::imread(path.string(), cv::IMREAD_COLOR);
	if (color.empty()) throw std::runtime_error("Cannot read color image: " + path.string());
	if (color.cols != width || color.rows != height)
	{
		cv::resize(color, color, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
	}
	return color;
}

std::vector<cv::Mat> load_depth_frame(const std::vector<fs::path>& files, int frame, int cameras)
{
	std::vector<cv::Mat> frameDepths;
	frameDepths.reserve(cameras);
	const int base = frame * cameras;
	for (int camera = 0; camera < cameras; ++camera)
	{
		frameDepths.push_back(load_depth(files[base + camera]));
	}
	return frameDepths;
}

std::vector<cv::Mat> load_color_frame(const std::vector<fs::path>& files, int frame, int cameras, int width, int height)
{
	std::vector<cv::Mat> frameColors;
	frameColors.reserve(cameras);
	const int base = frame * cameras;
	for (int camera = 0; camera < cameras; ++camera)
	{
		frameColors.push_back(load_color(files[base + camera], width, height));
	}
	return frameColors;
}

std::vector<double> parse_double_list(const std::string& text)
{
	std::vector<double> values;
	std::string current;
	for (char c : text)
	{
		if (c == ',' || c == ';')
		{
			if (!current.empty()) values.push_back(std::stod(current));
			current.clear();
		}
		else
		{
			current.push_back(c);
		}
	}
	if (!current.empty()) values.push_back(std::stod(current));
	return values;
}

void print_usage()
{
	std::cout
		<< "Usage:\n"
		<< "  DeformableFusionDemo --depth-dir ../Dataset/upperbody/data --output ../Dataset/outputs [options]\n\n"
		<< "Options:\n"
		<< "  --input DIR              Folder containing depth/rgb files; depth files are filtered by --depth-token.\n"
		<< "  --depth-dir DIR          Folder containing depth images. Overrides --input for depth.\n"
		<< "  --rgb-dir DIR            Optional folder containing RGB images. Overrides --input for RGB.\n"
		<< "  --depth-token TEXT       Filename substring for depth files when using --input.\n"
		<< "  --rgb-token TEXT         Filename substring for RGB files when using --input. Default: rgb.\n"
		<< "  --camera-count N         Number of depth images per frame. Default: 1.\n"
		<< "  --config FILE            Optional DeformableFusion cfg file.\n"
		<< "  --calib FILE             Optional calibCameras3DTM.json file.\n"
		<< "  --gpu N                  CUDA device id. Default: 0.\n"
		<< "  --fx V --fy V --cx V --cy V  Intrinsics fallback when --calib is absent.\n"
		<< "  --preview                Show a live OpenCV point-cloud preview from F2F outputs.\n"
		<< "  --preview-every N        Preview every N frames. Default: 1.\n"
		<< "  --preview-max-edge PX    Max rasterized preview triangle edge in pixels. Default: 80.\n"
		<< "  --viewer3d               Show an interactive live OpenGL 3D mesh viewer.\n"
		<< "  --dump-surface-every N   Export F2F surface every N frames. Default: disabled.\n"
		<< "  --dump-surface-format F  Export format: ply or obj. Default: ply.\n"
		<< "  --bbox xmin,xmax,ymin,ymax,zmin,zmax  Volume bounds in cm.\n";
}

std::string require_value(int argc, char** argv, int& i)
{
	if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + argv[i]);
	return argv[++i];
}

Options parse_options(int argc, char** argv)
{
	Options opt;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help" || arg == "-h")
		{
			print_usage();
			std::exit(0);
		}
		else if (arg == "--config") opt.configPath = require_value(argc, argv, i);
		else if (arg == "--input") opt.inputDir = require_value(argc, argv, i);
		else if (arg == "--depth-dir") opt.depthDir = require_value(argc, argv, i);
		else if (arg == "--rgb-dir") opt.rgbDir = require_value(argc, argv, i);
		else if (arg == "--output") opt.outputDir = require_value(argc, argv, i);
		else if (arg == "--depth-token") opt.depthToken = require_value(argc, argv, i);
		else if (arg == "--rgb-token") opt.rgbToken = require_value(argc, argv, i);
		else if (arg == "--calib") opt.calibrationPath = require_value(argc, argv, i);
		else if (arg == "--camera-count") opt.cameraCount = std::stoi(require_value(argc, argv, i));
		else if (arg == "--gpu") opt.gpu = std::stoi(require_value(argc, argv, i));
		else if (arg == "--fx") opt.fx = std::stod(require_value(argc, argv, i));
		else if (arg == "--fy") opt.fy = std::stod(require_value(argc, argv, i));
		else if (arg == "--cx") opt.cx = std::stod(require_value(argc, argv, i));
		else if (arg == "--cy") opt.cy = std::stod(require_value(argc, argv, i));
		else if (arg == "--preview") opt.preview = true;
		else if (arg == "--preview-every") opt.previewEvery = std::max(1, std::stoi(require_value(argc, argv, i)));
		else if (arg == "--preview-max-edge") opt.previewMaxEdge = std::max(1.0f, static_cast<float>(std::stod(require_value(argc, argv, i))));
		else if (arg == "--viewer3d") opt.viewer3d = true;
		else if (arg == "--dump-surface-every") opt.dumpSurfaceEvery = std::max(0, std::stoi(require_value(argc, argv, i)));
		else if (arg == "--dump-surface-format") opt.dumpSurfaceFormat = lower_string(require_value(argc, argv, i));
		else if (arg == "--bbox")
		{
			const auto values = parse_double_list(require_value(argc, argv, i));
			if (values.size() != 6) throw std::runtime_error("--bbox expects 6 comma-separated values");
			opt.bbox = BoundingBox3D(values[0], values[1], values[2], values[3], values[4], values[5]);
		}
		else
		{
			throw std::runtime_error("Unknown option: " + arg);
		}
	}

	if (opt.depthDir.empty()) opt.depthDir = opt.inputDir;
	if (opt.rgbDir.empty()) opt.rgbDir = opt.inputDir;
	if (opt.depthDir.empty()) throw std::runtime_error("Pass --depth-dir or --input");
	if (opt.cameraCount < 1) throw std::runtime_error("--camera-count must be >= 1");
	if (opt.depthToken.empty() && !opt.inputDir.empty() && opt.depthDir == opt.inputDir) opt.depthToken = "depth";
	if (opt.rgbToken.empty()) opt.rgbToken = "rgb";
	return opt;
}

std::vector<std::unique_ptr<GCameraView>> create_cameras(const Options& opt, int width, int height)
{
	std::vector<std::unique_ptr<GCameraView>> cameras;
	cameras.reserve(opt.cameraCount);
	for (int camera = 0; camera < opt.cameraCount; ++camera)
	{
		auto view = std::make_unique<GCameraView>(height, width);
		if (!opt.calibrationPath.empty())
		{
			if (!view->LoadCalibrationFrom3DTMFormat(opt.calibrationPath.c_str(), camera, "depth", -1.0))
			{
				throw std::runtime_error("Cannot load depth calibration camera " + std::to_string(camera));
			}
		}
		else
		{
			const double cx = opt.cx >= 0.0 ? opt.cx : width * 0.5;
			const double cy = opt.cy >= 0.0 ? opt.cy : height * 0.5;
			const double k[9] = { opt.fx, 0.0, cx, 0.0, opt.fy, cy, 0.0, 0.0, 1.0 };
			const double r[9] = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
			const double t[3] = { 0.0, 0.0, 0.0 };
			view->SetCalibrationMatrix(k, r, t, nullptr);
		}
		cameras.push_back(std::move(view));
	}
	return cameras;
}

std::vector<GCameraView*> raw_camera_ptrs(std::vector<std::unique_ptr<GCameraView>>& cameras)
{
	std::vector<GCameraView*> raw;
	raw.reserve(cameras.size());
	for (auto& camera : cameras) raw.push_back(camera.get());
	return raw;
}

template <typename FusionT>
void apply_common_config(FusionT& fusion, CConfig* conf, const char* prefix)
{
	if (!conf) return;
	conf->SetErrorIfNameNotFound(false);
	fusion.align_mu = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_align_mu").c_str(), fusion.align_mu);
	fusion.ed_nodes_res = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_ed_nodes_res").c_str(), fusion.ed_nodes_res);
	fusion.vxl_res = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_vxl_res").c_str(), fusion.vxl_res);
	fusion.ed_nodes_res_low = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_ed_nodes_res_low").c_str(), fusion.ed_nodes_res_low);
	fusion.vxl_res_low = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_vxl_res_low").c_str(), fusion.vxl_res_low);
	fusion.fusion_mu = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_fusion_mu").c_str(), fusion.fusion_mu);
	fusion.iso_surface_level = conf->GetValueWithDefault<float>("Fusion", (std::string(prefix) + "_iso_surface_level").c_str(), fusion.iso_surface_level);
	fusion.bBilateralFilter = conf->GetValueWithDefault<bool>("Fusion", (std::string(prefix) + "_bBilateralFilter").c_str(), fusion.bBilateralFilter);
	fusion.bUseDepthTopBitAsSeg = conf->GetValueWithDefault<bool>("Fusion", (std::string(prefix) + "_bUseDepthTopBitAsSeg").c_str(), fusion.bUseDepthTopBitAsSeg);
}

std::vector<cv::Mat> feed_optional_color(
	Fusion4DKeyFrames& fusion4d,
	CMotionFieldEstimationF2F& f2f,
	const std::vector<fs::path>& colorFiles,
	int frame,
	int cameras,
	int width,
	int height)
{
	if (colorFiles.empty()) return {};
	auto colorImages = load_color_frame(colorFiles, frame, cameras, width, height);
	fusion4d.vol_fusion.feed_color_textures(colorImages);
	f2f.vol_fusion.feed_color_textures(colorImages);
	return colorImages;
}
}

int main(int argc, char** argv)
{
	try
	{
		const Options opt = parse_options(argc, argv);
		DEPTH_CAMERAS_NUM = opt.cameraCount;

		if (!opt.configPath.empty())
		{
			if (!FusionConfig::initialize_config(opt.configPath))
			{
				return 2;
			}
			auto conf = FusionConfig::current();
			if (conf)
			{
				DEPTH_CAMERAS_NUM = conf->GetValueWithDefault<int>("DepthGeneration", "DepthCameraCount", DEPTH_CAMERAS_NUM);
				FrameCountKeyVolume = conf->GetValueWithDefault<int>("Fusion", "FrameCountKeyVolume", FrameCountKeyVolume);
			}
		}

		if (DEPTH_CAMERAS_NUM != opt.cameraCount)
		{
			throw std::runtime_error("Config DepthCameraCount differs from --camera-count");
		}

		cudaError_t cudaStatus = cudaSetDevice(opt.gpu);
		if (cudaStatus != cudaSuccess)
		{
			throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cudaStatus));
		}

		fs::create_directories(opt.outputDir);
		const auto depthFiles = list_images(opt.depthDir, opt.depthToken);
		const auto colorFiles = opt.rgbDir.empty() ? std::vector<fs::path>() : list_images(opt.rgbDir, opt.rgbToken);
		if (depthFiles.empty()) throw std::runtime_error("No depth images found");
		if (depthFiles.size() % opt.cameraCount != 0)
		{
			throw std::runtime_error("Depth image count is not divisible by --camera-count");
		}

		const int frameCount = static_cast<int>(depthFiles.size() / opt.cameraCount);
		const bool hasColor = colorFiles.size() >= depthFiles.size();
		if (!colorFiles.empty() && !hasColor)
		{
			std::cerr << "RGB files ignored: expected at least " << depthFiles.size() << ", found " << colorFiles.size() << "\n";
		}

		auto firstDepth = load_depth_frame(depthFiles, 0, opt.cameraCount);
		const int depthWidth = firstDepth.front().cols;
		const int depthHeight = firstDepth.front().rows;
		for (const auto& depth : firstDepth)
		{
			if (depth.cols != depthWidth || depth.rows != depthHeight)
			{
				throw std::runtime_error("All depth images in a frame must have the same size");
			}
		}

		auto cameras = create_cameras(opt, depthWidth, depthHeight);
		auto cameraPtrs = raw_camera_ptrs(cameras);

		Fusion4DKeyFrames fusion4d(nullptr, opt.cameraCount, depthWidth, depthHeight, FrameCountKeyVolume);
		CMotionFieldEstimationF2F f2f(nullptr, opt.cameraCount, depthWidth, depthHeight, FrameCountKeyVolume);
		fusion4d.setup_cameras(cameraPtrs);
		f2f.setup_cameras(cameraPtrs);

		auto conf = FusionConfig::current();
		apply_common_config(fusion4d, conf, "fusion4d");
		apply_common_config(f2f, conf, "f2f_motion");
		if (opt.viewer3d) initialize_live_3d_viewer(argc, argv);

		const fs::path fusion4dOutputDir = opt.outputDir / "fusion4d";
		const fs::path f2fOutputDir = opt.outputDir / "f2f";
		const fs::path surfaceDumpDir = opt.outputDir / "surface_dumps";
		fs::create_directories(fusion4dOutputDir);
		fs::create_directories(f2fOutputDir);
		const std::string fusion4dOut = fusion4dOutputDir.string();
		const std::string f2fOut = f2fOutputDir.string();
		const char* fusion4dOutDir = fusion4dOut.empty() ? nullptr : fusion4dOut.c_str();
		const char* f2fOutDir = f2fOut.empty() ? nullptr : f2fOut.c_str();

		std::cout << "Processing " << frameCount << " frame(s), " << opt.cameraCount
			<< " camera(s), " << depthWidth << "x" << depthHeight << ", GPU " << opt.gpu << "\n";
		std::cout << "F2F outputs: " << f2fOutputDir << "\n";
		std::cout << "Fusion4D outputs: " << fusion4dOutputDir << "\n";

		std::vector<cv::Mat> colorImages;
		if (hasColor) colorImages = feed_optional_color(fusion4d, f2f, colorFiles, 0, opt.cameraCount, depthWidth, depthHeight);
		fusion4d.set_up_1st_frame(firstDepth, opt.bbox, fusion4dOutDir, 0);
		f2f.set_up_1st_frame(firstDepth, opt.bbox, f2fOutDir, 0);
		{
			const cv::Mat* previewColor = colorImages.empty() ? nullptr : &colorImages.front();
			dump_surface_if_requested(f2fOutputDir / "accu_surface_0000.bin", surfaceDumpDir, 0, previewColor, opt, depthWidth, depthHeight);
			update_live_3d_viewer(f2fOutputDir / "accu_surface_0000.bin", previewColor, opt, depthWidth, depthHeight, 0);
		}
		if (opt.preview)
		{
			const cv::Mat* previewColor = colorImages.empty() ? nullptr : &colorImages.front();
			if (!show_live_preview(f2fOutputDir / "accu_surface_0000.bin", previewColor, opt, depthWidth, depthHeight)) return 0;
		}

		for (int frame = 1; frame < frameCount; ++frame)
		{
			auto depths = load_depth_frame(depthFiles, frame, opt.cameraCount);
			if (hasColor) colorImages = feed_optional_color(fusion4d, f2f, colorFiles, frame, opt.cameraCount, depthWidth, depthHeight);

			std::cout << "Frame " << frame << "/" << (frameCount - 1) << "\n";
			f2f.add_a_frame(depths, f2fOutDir, frame, false);
			EDNodesParasGPU edInit = f2f.ed_nodes_for_init();
			char surfaceFilename[64];
			std::snprintf(surfaceFilename, sizeof(surfaceFilename), "accu_surface_%04d.bin", frame);
			const cv::Mat* previewColor = colorImages.empty() ? nullptr : &colorImages.front();
			dump_surface_if_requested(f2fOutputDir / surfaceFilename, surfaceDumpDir, frame, previewColor, opt, depthWidth, depthHeight);
			update_live_3d_viewer(f2fOutputDir / surfaceFilename, previewColor, opt, depthWidth, depthHeight, frame);
			update_live_3d_graph(edInit, opt);
			if (opt.preview && frame % opt.previewEvery == 0)
			{
				if (!show_live_preview(f2fOutputDir / surfaceFilename, previewColor, opt, depthWidth, depthHeight)) return 0;
			}
			fusion4d.add_a_frame(depths, &edInit, fusion4dOutDir, frame, false);
			cudaDeviceSynchronize();
		}

		std::cout << "Done. Debug/surface outputs written to " << opt.outputDir << "\n";
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "DeformableFusionDemo error: " << ex.what() << "\n\n";
		print_usage();
		return 1;
	}
}
