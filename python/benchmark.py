#!/usr/bin/env python3
"""
likd-tree Performance Benchmark (Python)
Comparison with SciPy cKDTree
"""

import sys
import os
import time
import numpy as np

from likd_tree import KDTree as LIKDTree
from scipy.spatial import cKDTree
from scipy.spatial import KDTree


def benchmark_batch_build():
    """Test batch tree building performance"""    
    n_points = 100000
    np.random.seed(42)
    points = np.random.randn(n_points, 3).astype(np.float32)
    
    t0 = time.time()
    likd_tree = LIKDTree()
    likd_tree.build(points)
    likd_build_time = (time.time() - t0) * 1000
    
    t0 = time.time()
    ckdtree = cKDTree(points)
    ckdtree_build_time = (time.time() - t0) * 1000

    t0 = time.time()
    kdtree = KDTree(points)
    kdtree_build_time = (time.time() - t0) * 1000
    
    # Batch query test
    n_queries = 10000
    queries = np.random.randn(n_queries, 3).astype(np.float32)
        
    t0 = time.time()
    d_likd, idx_likd = likd_tree.nearest_neighbors(queries)
    likd_query_time = (time.time() - t0) * 1000
    
    t0 = time.time()
    d_ckdtree, idx_ckdtree = ckdtree.query(queries)
    ckdtree_query_time = (time.time() - t0) * 1000

    t0 = time.time()
    d_kdtree, idx_kdtree = kdtree.query(queries)
    kdtree_query_time = (time.time() - t0) * 1000

    # Verify correctness
    if not np.allclose(d_likd, d_kdtree) or not np.all(idx_likd == idx_kdtree):
        print("❌ Mismatch found between likd-tree and cKDTree results!")
    else:
        print("✅ likd-tree and cKDTree results match.")
    
    # Results
    print(f"\n{'-' * 70}")
    print(f"{'Metric':<20} {'likd-tree':<20} {'cKDTree':<20} {'KDTree':<20}")
    print(f"{'-' * 70}")
    print(f"{'Build Time (ms)':<20} {likd_build_time:<20.2f}", end='')
    print(f"{ckdtree_build_time:<20.2f}{kdtree_build_time:<20.2f}")
    
    print(f"{'Query Time (ms)':<20} {likd_query_time:<20.2f}", end='')
    print(f"{ckdtree_query_time:<20.2f}{kdtree_query_time:<20.2f}")
    

def benchmark_incremental_build():
    n_points = 11
    np.random.seed(42)
    points = np.random.randn(n_points, 3).astype(np.float32)

    # Batch query test
    n_queries = n_points
    queries = np.random.randn(n_queries, 3).astype(np.float32)


    likd_tree = LIKDTree()
    likd_tree.build(points)
    batch = 2
    for i in range(0, n_points, batch):
        if i == 0:
            likd_tree.build(points[i:i+batch])
        else:
            likd_tree.add_points(points[i:i+batch])

    ckdtree = cKDTree(points)
    d_ckdtree, idx_ckdtree = ckdtree.query(queries)
    d_likd, idx_likd = likd_tree.nearest_neighbors(queries)

    # Verify correctness
    if not np.allclose(d_likd, d_ckdtree) or not np.all(idx_likd == idx_ckdtree):
        print("❌ Mismatch found between likd-tree and cKDTree results in incremental build!")
    else:
        print("✅ likd-tree and cKDTree results match in incremental build.")




def main():
    # benchmark_batch_build()
    benchmark_incremental_build()

if __name__ == "__main__":
    main()
