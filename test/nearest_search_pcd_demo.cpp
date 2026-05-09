// nearest_search_pcd_demo.cpp - PCD nearest-neighbor demo for likd-tree

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

#include "likd_tree.hpp"

using PointType = pcl::PointXYZ;
using Clock = std::chrono::high_resolution_clock;

namespace {

void printUsage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " <map.pcd> <qx> <qy> <qz> [--no-vis]\n\n"
            << "Example:\n"
            << "  " << program << " map.pcd 10.0 -5.0 2.0\n";
}

bool parseFloat(const char* text, float& value) {
  try {
    size_t parsed = 0;
    value = std::stof(text, &parsed);
    return parsed == std::string(text).size();
  } catch (const std::exception&) {
    return false;
  }
}

bool isFinite(const PointType& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

float sqrDist(const PointType& a, const PointType& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

const PointType* bruteForceNearest(const PointVector<PointType>& points,
                                   const PointType& query,
                                   float& best_dist2) {
  const PointType* best_point = nullptr;
  best_dist2 = std::numeric_limits<float>::infinity();

  for (const auto& point : points) {
    const float dist2 = sqrDist(point, query);
    if (dist2 < best_dist2) {
      best_dist2 = dist2;
      best_point = &point;
    }
  }

  return best_point;
}

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void visualizeNearest(const pcl::PointCloud<PointType>::ConstPtr& cloud,
                      const PointType& query,
                      const PointType& nearest) {
  pcl::visualization::PCLVisualizer viewer(
      "likd-tree Nearest Neighbor Search");
  viewer.setBackgroundColor(0.03, 0.03, 0.03);
  viewer.addCoordinateSystem(1.0);

  pcl::visualization::PointCloudColorHandlerCustom<PointType> cloud_color(
      cloud, 170, 170, 170);
  viewer.addPointCloud<PointType>(cloud, cloud_color, "cloud");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

  pcl::PointCloud<PointType>::Ptr query_cloud(new pcl::PointCloud<PointType>);
  query_cloud->push_back(query);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> query_color(
      query_cloud, 0, 255, 0);
  viewer.addPointCloud<PointType>(query_cloud, query_color, "query");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "query");

  pcl::PointCloud<PointType>::Ptr nearest_cloud(new pcl::PointCloud<PointType>);
  nearest_cloud->push_back(nearest);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> nearest_color(
      nearest_cloud, 255, 0, 0);
  viewer.addPointCloud<PointType>(nearest_cloud, nearest_color, "nearest");
  viewer.setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 12, "nearest");

  viewer.addLine<PointType>(query, nearest, 255.0, 255.0, 0.0,
                            "query_to_nearest");
  viewer.setShapeRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "query_to_nearest");
  viewer.resetCamera();

  while (!viewer.wasStopped()) {
    viewer.spinOnce(16);
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    printUsage(argv[0]);
    return 1;
  }

  bool visualize = true;
  if (argc == 6) {
    if (std::string(argv[5]) != "--no-vis") {
      std::cerr << "Error: unknown option: " << argv[5] << '\n';
      printUsage(argv[0]);
      return 1;
    }
    visualize = false;
  }

  PointType query;
  if (!parseFloat(argv[2], query.x) || !parseFloat(argv[3], query.y) ||
      !parseFloat(argv[4], query.z)) {
    std::cerr << "Error: qx, qy and qz must be valid floating-point values.\n";
    printUsage(argv[0]);
    return 1;
  }
  if (!isFinite(query)) {
    std::cerr << "Error: query point coordinates must be finite values.\n";
    return 1;
  }

  pcl::PointCloud<PointType>::Ptr input_cloud(new pcl::PointCloud<PointType>);
  if (pcl::io::loadPCDFile<PointType>(argv[1], *input_cloud) < 0) {
    std::cerr << "Error: failed to read PCD file: " << argv[1] << '\n';
    return 1;
  }

  PointVector<PointType> points;
  points.reserve(input_cloud->size());

  pcl::PointCloud<PointType>::Ptr valid_cloud(new pcl::PointCloud<PointType>);
  valid_cloud->reserve(input_cloud->size());

  for (const auto& point : input_cloud->points) {
    if (!isFinite(point)) {
      continue;
    }
    points.push_back(point);
    valid_cloud->push_back(point);
  }

  if (points.empty()) {
    std::cerr << "Error: PCD file does not contain any finite XYZ points.\n";
    return 1;
  }

  KDTree<PointType> tree;

  const auto build_start = Clock::now();
  tree.build(points);
  const auto build_end = Clock::now();

  const auto query_start = Clock::now();
  const auto [nearest_ptr, distance] = tree.nearestNeighbors(query);
  const auto query_end = Clock::now();

  if (!nearest_ptr) {
    std::cerr << "Error: nearest-neighbor search returned no result.\n";
    return 1;
  }

  float brute_dist2 = 0.0f;
  const PointType* brute_nearest =
      bruteForceNearest(points, query, brute_dist2);
  const float brute_distance = std::sqrt(brute_dist2);
  const float tolerance =
      std::max(1e-5f, 1e-5f * std::max(distance, brute_distance));
  const bool match =
      brute_nearest && std::fabs(distance - brute_distance) <= tolerance;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Loaded points: " << input_cloud->size() << '\n';
  std::cout << "Finite points used: " << points.size() << '\n';
  std::cout << "Build time: " << elapsedMs(build_start, build_end) << " ms\n";
  std::cout << "Query time: " << elapsedMs(query_start, query_end) << " ms\n";
  std::cout << "Query: (" << query.x << ", " << query.y << ", " << query.z
            << ")\n";
  std::cout << "Nearest: (" << nearest_ptr->x << ", " << nearest_ptr->y
            << ", " << nearest_ptr->z << ")\n";
  std::cout << "Distance: " << distance << '\n';
  std::cout << "Brute-force distance: " << brute_distance << '\n';
  std::cout << "Brute-force check: " << (match ? "MATCH" : "MISMATCH")
            << '\n';

  if (visualize) {
    visualizeNearest(valid_cloud, query, *nearest_ptr);
  }
  return match ? 0 : 2;
}
