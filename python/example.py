"""
likd-tree Python bindings - Comparison with SciPy cKDTree
"""

import sys
import os
import time

# Add build directory to path to find likd_tree module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import likd_tree


# ============================================================================
# Basic Usage Example
# ============================================================================
print("=" * 70)
print("likd-tree Python Bindings - Basic Usage")
print("=" * 70)

# Create random points
np.random.seed(42)
n_points = 100
points = np.random.randn(n_points, 3).astype(np.float32)

# Create tree and build
tree = likd_tree.KDTree()
print(f"\nInitial tree size: {tree.size()}")

tree.build(points)
print(f"Tree size after build: {tree.size()}")

# Query: returns (distances, indices)
queries = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5]], dtype=np.float32)
distances, indices = tree.nearest_neighbors(queries)

print("\nNearest neighbor queries (likd-tree):")
for i in range(len(queries)):
    print(f"  Query {i}: {queries[i]} -> Index: {indices[i]}, Distance: {distances[i]:.4f}")

# Add more points
new_points = np.random.randn(50, 3).astype(np.float32)
tree.add_points(new_points)
print(f"\nTree size after adding points: {tree.size()}")

# Query after adding
d, idx = tree.nearest_neighbors(np.array([[2, 2, 2]], dtype=np.float32))
print(f"Query [2, 2, 2] -> Index: {idx[0]}, Distance: {d[0]:.4f}")

