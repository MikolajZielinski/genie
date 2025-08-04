#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>  // <-- Add this for torch::Tensor
#include "Header.cuh"

namespace py = pybind11;

// Example wrapper around CUDA_KNN_Init
bool py_cuda_knn_init(float chi_square_squared_radius, S_CUDA_KNN &knn) {
    return CUDA_KNN_Init(chi_square_squared_radius, &knn);
}

// Forward declare the C function
extern "C" bool CUDA_KNN_Fit(float4 *means, int number_of_means, S_CUDA_KNN* knn);

// Wrapper for CUDA_KNN_Fit that accepts a torch::Tensor
bool py_cuda_knn_fit(torch::Tensor means, int number_of_means, S_CUDA_KNN &knn) {
    // Check tensor is on CUDA and contiguous
    if (!means.is_cuda()) throw std::runtime_error("means tensor must be on CUDA");
    if (!means.is_contiguous()) throw std::runtime_error("means tensor must be contiguous");
    if (means.size(1) != 4) throw std::runtime_error("means tensor must have shape (N, 4)");

    // Get pointer to data as float4*
    float4* means_ptr = reinterpret_cast<float4*>(means.data_ptr<float>());
    return CUDA_KNN_Fit(means_ptr, number_of_means, &knn);
}

// Pybind11 wrapper for CUDA_KNN_KNeighbors
bool py_cuda_knn_kneighbors(
    torch::Tensor queried_points,  // (N, 4) float32, CUDA
    int K,
    torch::Tensor distances,       // (N, K) float32, CUDA
    torch::Tensor indices,         // (N, K) int32, CUDA
    S_CUDA_KNN &knn
) {
    // Checks
    if (!queried_points.is_cuda() || !queried_points.is_contiguous() || queried_points.scalar_type() != torch::kFloat32)
        throw std::runtime_error("queried_points must be contiguous float32 CUDA tensor");
    if (queried_points.size(1) != 4)
        throw std::runtime_error("queried_points must have shape (N, 4)");
    if (!distances.is_cuda() || !distances.is_contiguous() || distances.scalar_type() != torch::kFloat32)
        throw std::runtime_error("distances must be contiguous float32 CUDA tensor");
    if (!indices.is_cuda() || !indices.is_contiguous() || indices.scalar_type() != torch::kInt32)
        throw std::runtime_error("indices must be contiguous int32 CUDA tensor");
    if (distances.sizes() != indices.sizes())
        throw std::runtime_error("distances and indices must have the same shape");
    if (distances.size(0) != K || distances.size(1) != queried_points.size(0))
        throw std::runtime_error("distances/indices must have shape (K, N)");

    int N = queried_points.size(0);

    float4* queried_points_ptr = reinterpret_cast<float4*>(queried_points.data_ptr<float>());
    float* distances_ptr = distances.data_ptr<float>();
    int* indices_ptr = indices.data_ptr<int>();

    return CUDA_KNN_KNeighbors(
        queried_points_ptr,
        N,
        K,
        distances_ptr,
        indices_ptr,
        &knn
    );
}

bool py_cuda_knn_destroy(S_CUDA_KNN &cknn) {
    // Assuming you have a destroy function for S_CUDA_KNN
    return CUDA_KNN_Destroy(&cknn);
}

PYBIND11_MODULE(optix_knn, m) {
    py::class_<S_CUDA_KNN>(m, "S_CUDA_KNN")
        .def(py::init<>())
        // Add member bindings if needed
        ;

    m.def("CUDA_KNN_Init", &py_cuda_knn_init, py::arg("chi_square_squared_radius"), py::arg("knn"));
    m.def("CUDA_KNN_Fit", &py_cuda_knn_fit, py::arg("means"), py::arg("number_of_means"), py::arg("knn"));
    m.def("CUDA_KNN_KNeighbors", &py_cuda_knn_kneighbors,
        py::arg("queried_points"),
        py::arg("K"),
        py::arg("distances"),
        py::arg("indices"),
        py::arg("knn")
    );
    m.def("CUDA_KNN_Destroy", &py_cuda_knn_destroy, py::arg("knn"));
}