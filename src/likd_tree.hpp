/*
Copyright 2026 Liu Yang
Distributed under MIT license. See LICENSE for more information.
*/

// likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion
// with automatic background rebalancing. Header-only C++17 library.

#pragma once

#include <Eigen/Core>
#include <atomic>
#include <execution>
#include <limits>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

constexpr int KDTREE_DIM = 3;
constexpr double KDTREE_ALPHA = 0.75;
constexpr int INSERTION_BATCH_SIZE = 100;
constexpr int MIN_SUB_NUM = 8;
#ifdef LIKD_TREE_USE_TBB
#define TREE_PAR std::execution::par
#else
#define TREE_PAR std::execution::seq
#endif

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
  void nearestNeighbors(const PointVector<PointType>& queries,
                        PointVector<PointType>& results,
                        std::vector<float>& distances) const;
  int size() const;
  void waitForRebuild() const;  // Wait for any pending rebuild to complete

 private:
   static inline float coord(const PointType& pt, int axis) {
    return axis == 0 ? pt.x : (axis == 1 ? pt.y : pt.z);
  }
  static inline float sqrDist(const PointType& a, const PointType& b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
  }
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
  bool checkAncestorNeedsRebuild(Node* node) const;
  void backgroundRebuild(std::vector<Node*> nodes_to_rebuild);
  void insertPendingPoints();

  Node* root_ = nullptr;
  mutable std::shared_mutex tree_mutex_;
  std::atomic<bool> rebuilding_{false};
  PointVector<PointType> pending_points_;
  std::mutex pending_mutex_;

};

// AABB: axis-aligned bounding boxes
template <typename PointType>
KDTree<PointType>::AABB::AABB() {
  for (int i = 0; i < KDTREE_DIM; ++i) {
    min[i] = std::numeric_limits<float>::max();
    max[i] = std::numeric_limits<float>::lowest();
  }
}

template <typename PointType>
void KDTree<PointType>::AABB::expand(const PointType& pt) {
  min[0] = std::min(min[0], pt.x);
  min[1] = std::min(min[1], pt.y);
  min[2] = std::min(min[2], pt.z);
  max[0] = std::max(max[0], pt.x);
  max[1] = std::max(max[1], pt.y);
  max[2] = std::max(max[2], pt.z);
}

template <typename PointType>
void KDTree<PointType>::AABB::expand(const AABB& box) {
  for (int i = 0; i < KDTREE_DIM; ++i) {
    min[i] = std::min(min[i], box.min[i]);
    max[i] = std::max(max[i], box.max[i]);
  }
}

template <typename PointType>
float KDTree<PointType>::AABB::sqrDist(const PointType& pt) const {
  float d2 = 0;
  for (int i = 0; i < KDTREE_DIM; ++i) {
    float v = coord(pt, i);
    if (v < min[i])
      d2 += (min[i] - v) * (min[i] - v);
    else if (v > max[i])
      d2 += (v - max[i]) * (v - max[i]);
  }
  return d2;
}

template <typename PointType>
KDTree<PointType>::Node::Node(const PointType& pt, int ax)
    : point(pt), axis(ax) {
  aabb.expand(pt);
}

template <typename PointType>
KDTree<PointType>::Node::~Node() {
  delete left;
  delete right;
}

template <typename PointType>
KDTree<PointType>::KDTree() : root_(nullptr) {}

template <typename PointType>
KDTree<PointType>::~KDTree() {
  while (rebuilding_.load())
    std::this_thread::yield();
  std::unique_lock<std::shared_mutex> lock(tree_mutex_);
  delete root_;
}

template <typename PointType>
void KDTree<PointType>::build(const PointVector<PointType>& pts) {
  delete root_;
  if (pts.empty()) {
    root_ = nullptr;
    return;
  }
  PointVector<PointType> tmp = pts;
  root_ = buildRecursive(tmp, 0, tmp.size(), 0);
}

template <typename PointType>
void KDTree<PointType>::addPoints(const PointVector<PointType>& pts) {
  // If rebuilding, add points to pending buffer and return
  if (rebuilding_.load()) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_points_.insert(pending_points_.end(), pts.begin(), pts.end());
    return;
  }

  std::vector<Node*> nodes_to_rebuild;

  // Insertion + filtering phase - protected by exclusive lock
  {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);
    std::vector<Node*> rebuild_candidates;
    for (const auto& p : pts) {
      Node* renode = nullptr;
      root_ = insertInternal(root_, p, 0, &renode);
      if (renode) {
        rebuild_candidates.push_back(renode);
      }
    }

    // Filter: remove nodes whose ancestors also need rebuild
    for (Node* candidate : rebuild_candidates) {
      if (!checkAncestorNeedsRebuild(candidate)) {
        nodes_to_rebuild.push_back(candidate);
      }
    }
  }

  // Launch background rebuild if needed and not already rebuilding
  if (!nodes_to_rebuild.empty() && !rebuilding_.exchange(true)) {
    std::thread rebuild_thread(&KDTree::backgroundRebuild, this,
                               std::move(nodes_to_rebuild));
    rebuild_thread.detach();
  }
}

template <typename PointType>
void KDTree<PointType>::nearestNeighbors(const PointVector<PointType>& queries,
                                         PointVector<PointType>& results,
                                         std::vector<float>& distances) const {
  std::shared_lock<std::shared_mutex> lock(tree_mutex_);

  results.resize(queries.size());
  distances.assign(queries.size(), INFINITY);

  // If tree is empty, all queries return INFINITY distance
  if (root_ == nullptr) {
    return;
  }

  std::vector<size_t> indices(queries.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    const PointType* best_pt = nullptr;
    float best_dist2 = INFINITY;
    nearestNeighborInternal(root_, queries[i], best_pt, best_dist2);
    distances[i] = std::sqrt(best_dist2);
    results[i] = *best_pt;
  });
}

template <typename PointType>
int KDTree<PointType>::size() const {
  return root_ ? root_->subtree_size : 0;
}

template <typename PointType>
void KDTree<PointType>::waitForRebuild() const {
  while (rebuilding_.load()) {
    std::this_thread::yield();
  }
}

template <typename PointType>
typename KDTree<PointType>::Node* KDTree<PointType>::insertInternal(
    Node* node, const PointType& pt, int depth, std::optional<Node**> renode) {
  if (!node)
    return new Node(pt, depth % KDTREE_DIM);
  int ax = node->axis;
  float v = coord(pt, ax);
  float nv = coord(node->point, ax);
  if (v < nv) {
    node->left = insertInternal(node->left, pt, depth + 1, renode);
    node->left->parent = node;
    node->left->is_left_child = true;
  } else {
    node->right = insertInternal(node->right, pt, depth + 1, renode);
    node->right->parent = node;
    node->right->is_left_child = false;
  }
  update(node);
  // Only set need_rebuild to true, never clear it
  // Only delete a node marked by need_rebuild in rebuilding thread
  if (renode && !node->need_rebuild && needRebuild(node)) {
    node->need_rebuild = true;
    **renode = node;
  }
  return node;
}

template <typename PointType>
void KDTree<PointType>::update(Node* node) {
  node->subtree_size = 1;
  node->aabb = AABB();
  node->aabb.expand(node->point);
  if (node->left) {
    node->subtree_size += node->left->subtree_size;
    node->aabb.expand(node->left->aabb);
  }
  if (node->right) {
    node->subtree_size += node->right->subtree_size;
    node->aabb.expand(node->right->aabb);
  }
}

template <typename PointType>
bool KDTree<PointType>::needRebuild(Node* node) const {
  int lsz = node->left ? node->left->subtree_size : 0;
  int rsz = node->right ? node->right->subtree_size : 0;
  int maxsz = std::max(lsz, rsz);
  return maxsz > KDTREE_ALPHA * node->subtree_size &&
         node->subtree_size >= MIN_SUB_NUM;
}

template <typename PointType>
void KDTree<PointType>::collect(Node* node, PointVector<PointType>& pts) {
  if (!node)
    return;
  pts.push_back(node->point);
  collect(node->left, pts);
  collect(node->right, pts);
}

template <typename PointType>
typename KDTree<PointType>::Node* KDTree<PointType>::buildRecursive(
    PointVector<PointType>& pts, size_t l, size_t r, int axis) {
  if (l >= r)
    return nullptr;
  size_t m = l + (r - l) / 2;
  std::nth_element(pts.begin() + l, pts.begin() + m, pts.begin() + r,
                   [&](const PointType& a, const PointType& b) {
                     return coord(a, axis) < coord(b, axis);
                   });
  Node* node = new Node(pts[m], axis);
  node->left = buildRecursive(pts, l, m, (axis + 1) % KDTREE_DIM);
  if (node->left) {
    node->left->parent = node;
    node->left->is_left_child = true;
  }
  node->right = buildRecursive(pts, m + 1, r, (axis + 1) % KDTREE_DIM);
  if (node->right) {
    node->right->parent = node;
    node->right->is_left_child = false;
  }
  update(node);
  return node;
}

template <typename PointType>
void KDTree<PointType>::nearestNeighborInternal(Node* node,
                                                const PointType& query,
                                                const PointType*& best_pt,
                                                float& best_dist2) const {
  if (!node)
    return;
  float d2 = sqrDist(node->point, query);
  if (d2 < best_dist2) {
    best_dist2 = d2;
    best_pt = &node->point;
  }
  int ax = node->axis;
  float qv = coord(query, ax);
  float nv = coord(node->point, ax);
  Node* near = qv < nv ? node->left : node->right;
  Node* far = qv < nv ? node->right : node->left;
  if (near)
    nearestNeighborInternal(near, query, best_pt, best_dist2);
  if (far && far->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(far, query, best_pt, best_dist2);
}

template <typename PointType>
bool KDTree<PointType>::checkAncestorNeedsRebuild(Node* node) const {
  Node* ancestor = node->parent;
  bool needs_rebuild = false;
  while (ancestor) {
    if (ancestor->need_rebuild) {
      needs_rebuild = true;
      break;
    }
    ancestor = ancestor->parent;
  }
  return needs_rebuild;
}

// Background rebuild runs in separate thread
template <typename PointType>
void KDTree<PointType>::backgroundRebuild(std::vector<Node*> nodes_to_rebuild) {
  // Pre-allocate new_nodes for thread-safe parallel access
  std::vector<Node*> new_nodes(nodes_to_rebuild.size());

  // Create index vector for parallel iteration
  std::vector<size_t> indices(nodes_to_rebuild.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    PointVector<PointType> pts;
    collect(nodes_to_rebuild[i], pts);

    new_nodes[i] =
        buildRecursive(pts, 0, pts.size(), nodes_to_rebuild[i]->axis);

    new_nodes[i]->parent = nodes_to_rebuild[i]->parent;
    new_nodes[i]->is_left_child = nodes_to_rebuild[i]->is_left_child;
  });

  // Critical section - swap pointers (brief exclusive lock)
  {
    std::unique_lock<std::shared_mutex> lock(tree_mutex_);

    for (size_t i = 0; i < nodes_to_rebuild.size(); ++i) {
      Node* new_node = new_nodes[i];

      // Update parent's pointer to new subtree
      if (new_node->parent) {
        if (new_node->is_left_child) {
          new_node->parent->left = new_node;
        } else {
          new_node->parent->right = new_node;
        }
      } else {
        // This was the root
        root_ = new_node;
      }
      // Delete old subtree
      delete nodes_to_rebuild[i];
    }
  }
  // try to insert pending points accumulated during rebuild
  // the work will not block qeueries for too long.
  insertPendingPoints();
  // Lock released
  rebuilding_ = false;
}

// Insert pending points that accumulated during rebuild
template <typename PointType>
void KDTree<PointType>::insertPendingPoints() {
  PointVector<PointType> points_to_insert;

  // Move pending points to local buffer
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (pending_points_.empty()) {
      return;
    }
    points_to_insert = std::move(pending_points_);
    pending_points_.clear();
  }

  // Insert in small batches to allow queries to interleave
  for (size_t i = 0; i < points_to_insert.size(); i += INSERTION_BATCH_SIZE) {
    {
      std::unique_lock<std::shared_mutex> lock(tree_mutex_);

      size_t end = std::min(i + INSERTION_BATCH_SIZE, points_to_insert.size());
      for (size_t j = i; j < end; ++j) {
        root_ = insertInternal(root_, points_to_insert[j], 0, std::nullopt);
      }
    }
    std::this_thread::yield();
  }
}
