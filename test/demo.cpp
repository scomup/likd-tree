// demo.cpp - Simple demo showing how to use likd-tree
#include <pcl/point_types.h>

#include <iostream>
#include <random>
#include <vector>

#include "../src/likd_tree.hpp"

using PointType = pcl::PointXYZ;

void bruteForceNN(const PointVector<PointType>& pts,
                  const PointVector<PointType>& query,
                  std::vector<const PointType*>& best_pt,
                  std::vector<float>& best_dist2) {
  best_dist2.resize(query.size(), std::numeric_limits<float>::max());
  best_pt.resize(query.size(), nullptr);
  for (size_t i = 0; i < query.size(); ++i) {
    for (const auto& p : pts) {
      float dx = p.x - query[i].x;
      float dy = p.y - query[i].y;
      float dz = p.z - query[i].z;
      float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_dist2[i]) {
        best_dist2[i] = d2;
        best_pt[i] = &p;
      }
    }
  }
}

int main() {
  std::cout << "=== likd-tree Demo ===" << std::endl;
  std::cout << "A Lightweight Incremental KD-Tree implementation\n" << std::endl;

  // Step 1: Create a likd-tree instance
  KDTree<PointType> tree;
  
  // Step 2: Generate some random 3D points
  std::cout << "1. Generating 100,000 random points..." << std::endl;
  PointVector<PointType> points;
  points.reserve(100000);
  
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  
  for (int i = 0; i < 100000; ++i) {
    PointType pt;
    pt.x = dist(rng);
    pt.y = dist(rng);
    pt.z = dist(rng);
    points.push_back(pt);
  }
  
  // Step 3: Build the tree with initial points
  std::cout << "2. Building tree with all points..." << std::endl;
  tree.build(points);
  std::cout << "   Tree size: " << tree.size() << " points\n" << std::endl;
  
  // Step 4: Add more points incrementally
  std::cout << "3. Adding 10,000 more points incrementally..." << std::endl;
  PointVector<PointType> new_points;
  for (int i = 0; i < 10000; ++i) {
    PointType pt;
    pt.x = dist(rng);
    pt.y = dist(rng);
    pt.z = dist(rng);
    new_points.push_back(pt);
  }
  tree.addPoints(new_points);
  std::cout << "   Tree size: " << tree.size() << " points\n" << std::endl;
  
  // Step 5: Perform batch nearest neighbor queries
  std::cout << "4. Performing batch queries (5 query points)..." << std::endl;
  PointVector<PointType> queries;
  for (int i = 0; i < 5; ++i) {
    PointType q;
    q.x = dist(rng);
    q.y = dist(rng);
    q.z = dist(rng);
    queries.push_back(q);
  }
  
  PointVector<PointType> res;
  std::vector<float> dists;
  tree.nearestNeighbors(queries, res, dists);
  std::vector<const PointType*> bf_res;
  std::vector<float> bf_dists;
  bruteForceNN(points, queries, bf_res, bf_dists); // Just to show brute-force usage
  
  std::cout << "\n   Query results:" << std::endl;
  for (size_t i = 0; i < queries.size(); ++i) {
    // theck the results same as brute-force
    bool match = (bf_res[i] != nullptr &&
                  res[i].x == bf_res[i]->x &&
                  res[i].y == bf_res[i]->y &&
                  res[i].z == bf_res[i]->z);
    printf(" res[%zu]: (%.3f, %.3f, %.3f), dist2=%.3f %s\n",
           i, res[i].x, res[i].y,  res[i].z, dists[i],
           match ? "[MATCH]" : "[MISMATCH]");
  }
  
  
  std::cout << "\n=== Demo completed! ===" << std::endl;
  std::cout << "\nKey features demonstrated:" << std::endl;
  std::cout << "  ✓ Batch building with build()" << std::endl;
  std::cout << "  ✓ Incremental insertion with addPoints()" << std::endl;
  std::cout << "  ✓ Batch queries with nearestNeighbors()" << std::endl;
  std::cout << "  ✓ Thread-safe concurrent queries" << std::endl;
  std::cout << "  ✓ Automatic background rebalancing" << std::endl;
  
  return 0;
}
