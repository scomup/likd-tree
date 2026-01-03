// test_kdtree.cpp
// likd-tree vs ikd-tree benchmark

// Enable TBB parallel execution (define before including likd_tree.hpp)
#define LIKD_TREE_USE_TBB

#include <pcl/point_types.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../src/likd_tree.hpp"
#include "ikd_Tree.h"


using PointType = pcl::PointXYZ;

void bruteForceNN(const PointVector<PointType>& pts,
                  const PointType& query,
                  const PointType*& best_pt,
                  float& best_dist2) {
  best_dist2 = std::numeric_limits<float>::max();
  best_pt = nullptr;
  for (const auto& p : pts) {
    float dx = p.x - query.x;
    float dy = p.y - query.y;
    float dz = p.z - query.z;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < best_dist2) {
      best_dist2 = d2;
      best_pt = &p;
    }
  }
}

void queryMyKDtree(const KDTree<PointType>& tree,
                   const PointVector<PointType>& pts,
                   int num_points
) {

  const int Q = num_points;
  // Only query the first Q points, not all points!
  PointVector<PointType> queries(pts.begin(), pts.begin() + Q);
  PointVector<PointType> nn_pts;
  std::vector<float> nn_dists;
  tree.nearestNeighbors(queries, nn_pts, nn_dists);

}

void queryIKD(KD_TREE<PointType>* ikd,
                     const PointVector<PointType>& pts,
                     int num_points) {
  const int Q = num_points;
  for (int i = 0; i < Q; ++i) {
    PointType q = pts[i];
    PointVector<PointType> ikd_nn_pts;
    std::vector<float> ikd_nn_dists;
    ikd->Nearest_Search(q, 1, ikd_nn_pts, ikd_nn_dists);
  }
}

int main() {
  // Generate 1 million random points
  const size_t total_points = 100000;
  const int batch_size = 1000;
  
  std::cout << "Generating " << total_points << " random points..." << std::endl;
  
  PointVector<PointType> pts;
  pts.reserve(total_points);
  
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  
  for (size_t i = 0; i < total_points; ++i) {
    PointType pt;
    pt.x = dist(rng);
    pt.y = dist(rng);
    pt.z = dist(rng);
    pts.push_back(pt);
  }
  
  using clock = std::chrono::high_resolution_clock;
  
  // ============================================================
  // Part 1: Batch Build Test (Build all points at once)
  // ============================================================
  std::cout << "\n=== Part 1: Batch Build Test ===" << std::endl;
  std::cout << "Building trees with all " << total_points << " points at once..." << std::endl;
  
  KDTree<PointType> tree_batch;
  KD_TREE<PointType>* ikd_batch = new KD_TREE<PointType>();
  
  auto t0 = clock::now();
  tree_batch.build(pts);
  auto t1 = clock::now();
  
  ikd_batch->Build(pts);
  auto t2 = clock::now();
  
  double likd_build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ikd_build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  
  printf("Batch build time: likd-tree %.2f ms, ikd-tree %.2f ms\n", likd_build_ms, ikd_build_ms);
  
  // Query test after batch build
  queryMyKDtree(tree_batch, pts, 1000);
  auto t3 = clock::now();
  
  queryIKD(ikd_batch, pts, 1000);
  auto t4 = clock::now();
  
  double likd_query_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  double ikd_query_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
  
  printf("Query time (1000 queries): likd-tree %.2f ms, ikd-tree %.2f ms\n", likd_query_ms, ikd_query_ms);
  
  delete ikd_batch;
  
  // ============================================================
  // Part 2: Incremental Insertion Test
  // ============================================================
  std::cout << "\n=== Part 2: Incremental Insertion Test ===" << std::endl;
  std::cout << "Inserting points in batches of " << batch_size << "..." << std::endl;
  
  KDTree<PointType> tree_incr;
  KD_TREE<PointType>* ikd_incr = new KD_TREE<PointType>();
  
  double likd_incr_total_insert = 0.0;
  double ikd_incr_total_insert = 0.0;
  double likd_incr_total_query = 0.0;
  double ikd_incr_total_query = 0.0;
  int num_iterations = 0;
  
  for (size_t start = 0; start < pts.size(); start += batch_size) {
    size_t end = std::min(start + batch_size, pts.size());
    PointVector<PointType> batch(pts.begin() + start, pts.begin() + end);
    
    auto t0 = clock::now();
    if (start == 0) {
      tree_incr.build(batch);
    } else {
      tree_incr.addPoints(batch);
    }
    auto t1 = clock::now();

    if (start == 0) {
      ikd_incr->Build(batch);
    } else {
      ikd_incr->Add_Points(batch, false);
    }
    auto t2 = clock::now();

    queryMyKDtree(tree_incr, pts, 1000);
    auto t3 = clock::now();
    
    queryIKD(ikd_incr, pts, 1000);
    auto t4 = clock::now();

    double mkd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ikd_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double mkd_query_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double ikd_query_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    likd_incr_total_insert += mkd_ms;
    ikd_incr_total_insert += ikd_ms;
    likd_incr_total_query += mkd_query_ms;
    ikd_incr_total_query += ikd_query_ms;
    num_iterations++;
    
    printf("Inserted: %zu / %zu, insert: likd %.2f ms, ikd %.2f ms, query: likd %.2f ms, ikd %.2f ms\n", 
           end, pts.size(), mkd_ms, ikd_ms, mkd_query_ms, ikd_query_ms);
  }
  
  delete ikd_incr;
  
  // ============================================================
  // Final Comparison Report
  // ============================================================
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "                    FINAL COMPARISON REPORT" << std::endl;
  std::cout << std::string(70, '=') << std::endl;
  std::cout << "Total points: " << total_points << std::endl;
  std::cout << std::string(70, '-') << std::endl;
  
  std::cout << "\n[Part 1] Batch Build Performance:" << std::endl;
  printf("  Build Time:        likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_build_ms, ikd_build_ms, ikd_build_ms / likd_build_ms);
  printf("  Query Time (1000): likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_query_ms, ikd_query_ms, ikd_query_ms / likd_query_ms);
  
  std::cout << "\n[Part 2] Incremental Insertion Performance:" << std::endl;
  printf("  Total Insert Time: likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_incr_total_insert, ikd_incr_total_insert, ikd_incr_total_insert / likd_incr_total_insert);
  printf("  Avg Insert/Batch:  likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_incr_total_insert / num_iterations, ikd_incr_total_insert / num_iterations,
         (ikd_incr_total_insert / num_iterations) / (likd_incr_total_insert / num_iterations));
  printf("  Total Query Time:  likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_incr_total_query, ikd_incr_total_query, ikd_incr_total_query / likd_incr_total_query);
  printf("  Avg Query/Batch:   likd-tree %8.2f ms  |  ikd-tree %8.2f ms  |  Ratio: %.2fx\n", 
         likd_incr_total_query / num_iterations, ikd_incr_total_query / num_iterations,
         (ikd_incr_total_query / num_iterations) / (likd_incr_total_query / num_iterations));
  
  std::cout << "\n[Summary] Overall Winner:" << std::endl;
  double likd_total = likd_build_ms + likd_query_ms + likd_incr_total_insert + likd_incr_total_query;
  double ikd_total = ikd_build_ms + ikd_query_ms + ikd_incr_total_insert + ikd_incr_total_query;
  printf("  Total Time:        likd-tree %8.2f ms  |  ikd-tree %8.2f ms\n", likd_total, ikd_total);
  if (likd_total < ikd_total) {
    printf("  🏆 likd-tree is %.2fx FASTER overall!\n", ikd_total / likd_total);
  } else {
    printf("  🏆 ikd-tree is %.2fx faster overall.\n", likd_total / ikd_total);
  }
  
  std::cout << std::string(70, '=') << std::endl;
  return 0;
}
