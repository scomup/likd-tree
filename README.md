# likd-tree

**A Lightweight Incremental KD-Tree for Robotic Applications**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`likd-tree` is a lightweight incremental KD-tree designed for dynamic point insertion without requiring full tree reconstruction.

Inspired by [ikd-tree](https://github.com/hku-mars/ikd-Tree), `likd-tree` is completely reimplemented using modern C++17 and features a more intelligent and principled rebuild strategy, which significantly improves efficiency while keeping the structure lightweight and easy to maintain.

## 🚀 Key Features

- **🔄 Incremental**: Dynamic point insertion with automatic background rebalancing
- **🪶 Lightweight**: ~375 lines of clean, modern C++17 code (vs ikd-tree's ~1700 lines)
- **⚡ Fast**: 1.7x faster overall than ikd-tree on typical workloads

## 📊 Performance Comparison

Benchmark on 100,000 random 3D points (Intel CPU, -O3 optimization):

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| **Batch Build** | 25.51 ms | 37.90 ms | **1.49x** |
| **Batch Query (1000)** | 0.98 ms | 0.98 ms | 1.00x |
| **Incremental Insert** | 66.00 ms | 127.18 ms | **1.93x** |
| **Concurrent Query** | 42.50 ms | 64.15 ms | **1.51x** |


### To reproduce these results:
```bash
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark
```

## 🎯 Quick Start

### Basic Usage

```cpp
#include "likd_tree.h"
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
tree.nearestNeighbor(queries, results, distances);
```

## 🛠️ Build Instructions

### Prerequisites

- CMake >= 3.10
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- PCL (Point Cloud Library) - optional but recommended

### Build Demo

```bash
git clone https://github.com/liu-yangs/kdtree.git
cd kdtree
cmake -B build
cmake --build build
./build/demo
```

### Build Benchmark (Compare with ikd-tree)

```bash
git clone https://github.com/liu-yangs/kdtree.git
cd kdtree
git submodule update --init  # Download ikd-tree
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark
```

## 📋 TODO

**Planned Features:**
- [ ] Node deletion support (Node deletion not supported now)
- [ ] k-nearest neighbors (k-NN) query
- [ ] box/radius queries

> **Note:** If you need point deletion or downsampling, consider using [ikd-tree](https://github.com/hku-mars/ikd-Tree) instead.

**Made with ❤️ for the robotics community**
