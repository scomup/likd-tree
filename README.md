# likd-tree

**A Lightweight Incremental KD-Tree for Robotic Applications**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![PyPI version](https://img.shields.io/pypi/v/likd-tree.svg)](https://pypi.org/project/likd-tree/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`likd-tree` is a lightweight incremental KD-tree designed for dynamic point insertion with automatic rebuilding.

## C++ Version
Inspired by [ikd-tree](https://github.com/hku-mars/ikd-Tree), `likd-tree` is completely reimplemented using modern C++17 and features a more intelligent and principled rebuild strategy, which significantly improves efficiency while keeping the structure lightweight and easy to maintain.

## Python Version

**The python version likd-tree is also available now!🎉** 

To the best of our knowledge, this is the **first Python KD-tree library that supports incremental insertion with automatic rebuilding**. Install it via PyPI:
```bash
pip install likd-tree
```
For details see [Python Usage](#python-usage)

## 🚀 Key Features

- **🔄 Incremental**: Dynamic point insertion with automatic background rebalancing
- **🪶 Lightweight**: Header-only library (~450 lines of clean C++17) - no build required
- **⚡ Fast**: 2.44x faster incremental insertion than ikd-tree
- **🧠 Intelligent**: Smarter rebuild strategy with delayed and batched rebuilding of multiple non-overlapping unbalanced subtrees *(paper-worthy?)* 

## 📊 Performance Comparison

Benchmark on 100K random 3D points (Intel CPU, -O3 optimization, with TBB parallel execution):

### Batch Build Performance
| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Build Time | 23.70 ms | 36.70 ms | **1.55x** |
| Query Time (1000 queries) | 1.23 ms | 1.30 ms | **1.06x** |

### Incremental Insertion Performance
100K points inserted in batches of 1000:

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Total Insert Time | 84.04 ms | 151.82 ms | **1.81x**⭐ |
| Total Query Time | 17.37 ms | 71.38 ms | **4.11x**⭐  |


### Reproduce these results:
```bash
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark
```

## 🎯 Quick Start

### C++ Header-Only Usage

Simply include `likd_tree.hpp` in your project - no build or installation needed!

```cpp
#include <pcl/point_types.h>
// Define LIKD_TREE_USE_TBB BEFORE including the header to enable TBB parallel acceleration
#define LIKD_TREE_USE_TBB
#include "likd_tree.hpp"

using PointType = pcl::PointXYZ;

// Create tree
KDTree<PointType> tree;

// Build with initial points
PointVector<PointType> points = {...};
tree.build(points);

// Add more points incrementally
PointVector<PointType> new_points = {...};
tree.addPoints(new_points);

// Batch nearest neighbor queries
PointVector<PointType> queries = {...};
PointVector<PointType> results;
std::vector<float> distances;
tree.nearestNeighbors(queries, results, distances);
```

**To enable TBB parallel acceleration:**
- Add `#define LIKD_TREE_USE_TBB` before including `likd_tree.hpp`
- Link against TBB library in your build system (CMakeLists.txt or build script)

**To disable TBB (sequential execution):**
- Simply don't define `LIKD_TREE_USE_TBB`, or comment it out


### Python Usage

Install via PyPI:
```bash
pip install likd_tree
```

Usage:

```python
import numpy as np
from likd_tree import KDTree

# Create and build tree
points = np.random.rand(10000, 3).astype(np.float32)
tree = KDTree()
tree.build(points)

# Query nearest neighbors
queries = np.random.rand(100, 3).astype(np.float32)
distances, indices = tree.nearest_neighbors(queries)

# Add more points incrementally
new_points = np.random.rand(1000, 3).astype(np.float32)
tree.add_points(new_points)

print(f"Tree size: {tree.size()}")
```

## 🛠️ Demo & Benchmark

### Run Demo

```bash
git clone https://github.com/scomup/likd-tree.git
cd likd-tree
cmake -B build
cmake --build build
./build/demo
```

### Run Benchmark (Compare with ikd-tree)

```bash
git clone https://github.com/scomup/likd-tree.git
cd likd-tree
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark
```

### Rebuild visualization demo

![img](imgs/rebuild.gif)

**Note:** Benchmarks and demos require CMake to compile, but the library itself is pure header-only and needs no build step for integration into your project.

## 📋 TODO

**Planned Features:**
- [ ] Node deletion support (Node deletion not supported now)
- [ ] k-nearest neighbors (k-NN) query
- [ ] box/radius queries

> **Note:** If you require these features immediately, consider using [ikd-tree](https://github.com/hku-mars/ikd-Tree) instead.
