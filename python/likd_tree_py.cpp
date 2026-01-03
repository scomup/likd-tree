/*
Copyright 2026 Liu Yang
Distributed under MIT license. See LICENSE for more information.
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <cmath>
#include "../src/likd_tree.hpp"

namespace py = pybind11;

// Simple Point struct compatible with likd-tree
struct Point {
    float x, y, z;
    int idx;  // Point index for identification
    Point() : x(0), y(0), z(0), idx(-1) {}
    Point(float x_, float y_, float z_) : x(x_), y(y_), z(z_), idx(-1) {}
};

using PointType = Point;

PYBIND11_MODULE(likd_tree, m) {
    m.doc() = "likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion";

    // Bind KDTree class
    py::class_<KDTree<PointType>>(m, "KDTree")
        .def(py::init<>())
        
        .def("build", 
             [](KDTree<PointType>& tree, py::object points_obj) {
                 auto points_array = py::array_t<float>::ensure(points_obj);
                 auto buf = points_array.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     throw std::runtime_error("Input array must be Nx3");
                 }
                 
                 float* ptr = static_cast<float*>(buf.ptr);
                 PointVector<PointType> points;
                 points.reserve(buf.shape[0]);
                 
                 for (size_t i = 0; i < buf.shape[0]; ++i) {
                     PointType p(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
                     p.idx = i;
                     points.push_back(p);
                 }
                 tree.build(points);
             },
             "points")
        
        .def("add_points", 
             [](KDTree<PointType>& tree, py::object points_obj) {
                 auto points_array = py::array_t<float>::ensure(points_obj);
                 auto buf = points_array.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     throw std::runtime_error("Input array must be Nx3");
                 }
                 
                 float* ptr = static_cast<float*>(buf.ptr);
                 PointVector<PointType> points;
                 points.reserve(buf.shape[0]);
                 
                 int base_idx = tree.size();
                 for (size_t i = 0; i < buf.shape[0]; ++i) {
                     PointType p(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
                     p.idx = base_idx + i;
                     points.push_back(p);
                 }
                 tree.addPoints(points);
             },
             "points")
        
        .def("nearest_neighbors",
             [](const KDTree<PointType>& tree, py::object queries_obj) {
                 auto queries_array = py::array_t<float>::ensure(queries_obj);
                 auto buf = queries_array.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     throw std::runtime_error("Input array must be Nx3");
                 }
                 
                 float* ptr = static_cast<float*>(buf.ptr);
                 PointVector<PointType> queries;
                 queries.reserve(buf.shape[0]);
                 
                 for (size_t i = 0; i < buf.shape[0]; ++i) {
                     queries.push_back(PointType(
                         ptr[i * 3],
                         ptr[i * 3 + 1],
                         ptr[i * 3 + 2]
                     ));
                 }
                 
                 PointVector<PointType> results;
                 std::vector<float> distances;
                 tree.nearestNeighbors(queries, results, distances);
                 
                 auto dist_array = py::array_t<float>(distances.size());
                 auto idx_array = py::array_t<int>(results.size());
                 
                 auto dist_buf = dist_array.request();
                 auto idx_buf = idx_array.request();
                 float* dist_ptr = static_cast<float*>(dist_buf.ptr);
                 int* idx_ptr = static_cast<int*>(idx_buf.ptr);
                 
                 for (size_t i = 0; i < results.size(); ++i) {
                     dist_ptr[i] = distances[i];
                     idx_ptr[i] = results[i].idx;
                 }
                 return py::make_tuple(dist_array, idx_array);
             },
             "x")
        
        .def("size", &KDTree<PointType>::size);
}


