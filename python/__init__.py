"""
likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion

A Python binding for the C++17 likd-tree library, providing efficient
dynamic point insertion with automatic background rebalancing.

Basic usage:

    import numpy as np
    import likd_tree
    
    # Create and build tree
    points = np.random.randn(1000, 3).astype(np.float32)
    tree = likd_tree.KDTree()
    tree.build(points)
    
    # Query
    queries = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    distances, indices = tree.nearest_neighbors(queries)
    
    # Add more points incrementally
    new_points = np.random.randn(100, 3).astype(np.float32)
    tree.add_points(new_points)
    
    print(f"Tree size: {tree.size()}")

For more information, see: https://github.com/scomup/likd-tree
"""

__version__ = '1.0.0'
__author__ = 'Liu Yang'
__license__ = 'MIT'

try:
    from likd_tree import KDTree
    __all__ = ['KDTree']
except ImportError:
    __all__ = []
