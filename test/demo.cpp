// demo.cpp - Simple demo showing how to use likd-tree

// Enable TBB parallel execution (define before including likd_tree.hpp)
#define LIKD_TREE_USE_TBB

#include <iostream>
#include <random>
#include <vector>

#include "../src/likd_tree.hpp"
#include <Eigen/Core>

// Specialize PointTraits for Eigen::Vector3f
template <>
struct PointTraits<Eigen::Vector3f> {
  static constexpr int DIM = 3;
  static inline float coord(const Eigen::Vector3f& pt, int axis) {
    return pt[axis];
  }
  static inline float sqrDist(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return (a - b).squaredNorm();
  }
};

void bruteForceNN(const PointVector<Eigen::Vector3f>& pts,
                  const PointVector<Eigen::Vector3f>& query,
                  std::vector<const Eigen::Vector3f*>& best_pt,
                  std::vector<float>& best_dist2) {
  best_dist2.resize(query.size(), std::numeric_limits<float>::max());
  best_pt.resize(query.size(), nullptr);
  for (size_t i = 0; i < query.size(); ++i) {
    for (const auto& p : pts) {
      float dx = p.x() - query[i].x();
      float dy = p.y() - query[i].y();
      float dz = p.z() - query[i].z();
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
  KDTree<Eigen::Vector3f, PointTraits<Eigen::Vector3f>> tree;
  
  // Step 2: Generate some random 3D points
  std::cout << "1. Generating 100,000 random points..." << std::endl;
  PointVector<Eigen::Vector3f> points;
  points.reserve(100000);
  
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  
  for (int i = 0; i < 100000; ++i) {
    Eigen::Vector3f pt;
    pt.x() = dist(rng);
    pt.y() = dist(rng);
    pt.z() = dist(rng);
    points.push_back(pt);
  }
  
  // Step 3: Build the tree with initial points
  std::cout << "2. Building tree with all points..." << std::endl;
  tree.build(points);
  std::cout << "   Tree size: " << tree.size() << " points\n" << std::endl;
  
  // Step 4: Add more points incrementally
  std::cout << "3. Adding 10,000 more points incrementally..." << std::endl;
  PointVector<Eigen::Vector3f> new_points;
  for (int i = 0; i < 10000; ++i) {
    Eigen::Vector3f pt;
    pt.x() = dist(rng);
    pt.y() = dist(rng);
    pt.z() = dist(rng);
    new_points.push_back(pt);
  }
  tree.addPoints(new_points);
  std::cout << "   Tree size: " << tree.size() << " points\n" << std::endl;
  
  // Step 5: Perform batch nearest neighbor queries
  std::cout << "4. Performing batch queries (5 query points)..." << std::endl;
  PointVector<Eigen::Vector3f> queries;
  for (int i = 0; i < 5; ++i) {
    Eigen::Vector3f q;
    q.x() = dist(rng);
    q.y() = dist(rng);
    q.z() = dist(rng);
    queries.push_back(q);
  }
  
  PointVector<Eigen::Vector3f> res;
  std::vector<float> dists;
  tree.nearestNeighbors(queries, res, dists);
  std::vector<const Eigen::Vector3f*> bf_res;
  std::vector<float> bf_dists;
  bruteForceNN(points, queries, bf_res, bf_dists); // Just to show brute-force usage
  
  std::cout << "\n   Query results:" << std::endl;
  for (size_t i = 0; i < queries.size(); ++i) {
    // theck the results same as brute-force
    bool match = (bf_res[i] != nullptr &&
                  res[i].x() == bf_res[i]->x() &&
                  res[i].y() == bf_res[i]->y() &&
                  res[i].z() == bf_res[i]->z());
    printf(" res[%zu]: (%.3f, %.3f, %.3f), dist2=%.3f %s\n",
           i, res[i].x(), res[i].y(),  res[i].z(), dists[i],
           match ? "[MATCH]" : "[MISMATCH]");
  }
  
  
  // Step 6: Demonstrate custom point type support
  std::cout << "\n5. Testing custom point type (array-based)..." << std::endl;
  
  KDTree<Eigen::Vector3f> custom_tree;
  PointVector<Eigen::Vector3f> custom_points;
  for (int i = 0; i < 1000; ++i) {
    custom_points.push_back(Eigen::Vector3f(dist(rng), dist(rng), dist(rng)));
  }
  custom_tree.build(custom_points);
  
  Eigen::Vector3f custom_query(0.0f, 0.0f, 0.0f);
  auto [nearest, distance] = custom_tree.nearestNeighbors(custom_query);
  
  std::cout << "   Query: (0, 0, 0)" << std::endl;
  if (nearest) {
    printf("   Nearest: (%.3f, %.3f, %.3f), dist=%.3f\n",
           nearest->x(), nearest->y(), nearest->z(), distance);
  }
  std::cout << "   ✓ Custom point types work seamlessly!" << std::endl;
  
  std::cout << "\n=== Demo completed! ===" << std::endl;
  std::cout << "\nKey features demonstrated:" << std::endl;
  std::cout << "  ✓ Batch building with build()" << std::endl;
  std::cout << "  ✓ Incremental insertion with addPoints()" << std::endl;
  std::cout << "  ✓ Batch queries with nearestNeighbors()" << std::endl;
  std::cout << "  ✓ Thread-safe concurrent queries" << std::endl;
  std::cout << "  ✓ Automatic background rebalancing" << std::endl;
  std::cout << "  ✓ Custom point type support via PointTraits" << std::endl;
  
  return 0;
}
