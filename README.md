# likd-tree

**A Lightweight Incremental KD-Tree for Robotic Applications**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`likd-tree` is a lightweight incremental KD-tree designed for dynamic point insertion without requiring full tree reconstruction.

Inspired by [ikd-tree](https://github.com/hku-mars/ikd-Tree), `likd-tree` is completely reimplemented using modern C++17 and features a more intelligent and principled rebuild strategy, which significantly improves efficiency while keeping the structure lightweight and easy to maintain.

## 🚀 Key Features

- **🔄 Incremental**: Dynamic point insertion with automatic background rebalancing
- **🪶 Lightweight**: Header-only library (~450 lines of clean C++17) - no build required
- **⚡ Fast**: 2.44x faster incremental insertion than ikd-tree

## 📊 Performance Comparison

Benchmark on 100,000 random 3D points (Intel CPU, -O3 optimization):

### Test1: Batch Build Performance
| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Build Time | 24.28 ms | 37.92 ms | **1.56x** |
| Query Time (1000 queries) | 0.97 ms | 1.02 ms | **1.05x** |

### Test2: Incremental Insertion Performance
(100K points inserted in batches of 1000)

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Total Insert Time | 56.82 ms | 138.45 ms | **2.44x** |
| Total Query Time | 39.22 ms | 62.81 ms | **1.60x** |

### Reproduce these results:
```bash
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark
```

## 🎯 Quick Start

### Header-Only Usage

Simply include `likd_tree.hpp` in your project - no build or installation needed!

```cpp
#include "likd_tree.hpp"
#include <pcl/point_types.h>

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

**Note:** Benchmarks and demos require CMake to compile, but the library itself is pure header-only and needs no build step for integration into your project.

## 📋 TODO

**Planned Features:**
- [ ] Node deletion support (Node deletion not supported now)
- [ ] k-nearest neighbors (k-NN) query
- [ ] box/radius queries

> **Note:** If you require these features immediately, consider using [ikd-tree](https://github.com/hku-mars/ikd-Tree) instead.
