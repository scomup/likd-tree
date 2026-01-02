/*
Copyright 2026 Liu Yang
Distributed under MIT license. See LICENSE for more information.
*/



#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

constexpr int KDTREE_DIM = 3;
constexpr double KDTREE_ALPHA = 0.75;
constexpr int INSERTION_BATCH_SIZE = 100;
constexpr int MIN_SUB_NUM = 8;

template <typename PointType>
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

template <typename PointType>
class KDTree {
 public:

  struct AABB {
    std::array<float, KDTREE_DIM> min, max;
    AABB();
    void expand(const PointType& pt);
    void expand(const AABB& box);
    float sqrDist(const PointType& pt) const;
  };

  struct Node {
    Node* left = nullptr;
    Node* right = nullptr;
    Node* parent = nullptr;
    AABB aabb;
    PointType point;
    int axis;
    int subtree_size = 1;
    bool is_left_child = false;  // true if this is parent's left child
    bool need_rebuild = false;
    Node(const PointType& pt, int ax);
    ~Node();
  };

  KDTree();
  ~KDTree();

  void build(const PointVector<PointType>& pts);
  void addPoints(const PointVector<PointType>& pts);
  std::pair<const PointType*, float> nearestNeighbors(
      const PointType& query) const;
  // Batch query API - queries multiple points at once
  void nearestNeighbors(const PointVector<PointType>& queries,
                            PointVector<PointType>& results,
                            std::vector<float>& distances) const;
  int size() const;

 private:
  Node* root_ = nullptr;
  mutable std::shared_mutex tree_mutex_;  // Protects tree structure
  std::atomic<bool> rebuilding_{false};

  // Pending points buffer during rebuild
  PointVector<PointType> pending_points_;
  std::mutex pending_mutex_;

  Node* insertInternal(Node* node, const PointType& pt, int depth,
                       std::optional<Node**> renode);
  void update(Node* node);
  bool needRebuild(Node* node) const;
  void collect(Node* node, PointVector<PointType>& pts);
  Node* buildRecursive(PointVector<PointType>& pts, size_t l, size_t r,
                       int axis);
  void nearestNeighborInternal(Node* node, const PointType& query,
                               const PointType*& best_pt,
                               float& best_dist2) const;
  static inline float coord(const PointType& pt, int axis) {
    return axis == 0 ? pt.x : (axis == 1 ? pt.y : pt.z);
  }
  static inline float sqrDist(const PointType& a, const PointType& b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
  }
  bool checkAncestorNeedsRebuild(Node* node) const;

  // Background rebuild
  void backgroundRebuild(std::vector<Node*> nodes_to_rebuild);
  void insertPendingPoints();
};
