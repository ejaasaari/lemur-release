#include "maxsim.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_maxsim, m) {
  m.doc() = "Lemur maxsim extension";

  py::class_<MaxSim>(m, "MaxSim")
      .def(py::init([](py::array_t<float> train,
                       py::array_t<int32_t> train_counts) {
             py::buffer_info train_buf = train.request();
             py::buffer_info counts_buf = train_counts.request();

             if (train_buf.ndim != 2) {
               throw std::runtime_error("train must be a 2-dimensional array "
                                        "(total_vectors, vec_dim)");
             }
             if (counts_buf.ndim != 1) {
               throw std::runtime_error(
                   "train_counts must be a 1-dimensional array");
             }

             const float *train_ptr = static_cast<float *>(train_buf.ptr);
             const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
             int num_train_points = counts_buf.shape[0];
             int vec_dim = train_buf.shape[1];

             return new MaxSim(train_ptr, counts_ptr, vec_dim,
                                num_train_points);
           }),
           py::arg("train"), py::arg("train_counts"))

      .def(
          "rerank_subset",
          [](MaxSim &self, py::array_t<float> queries,
             py::array_t<int32_t> query_counts, int k,
             py::array_t<int> indices, int num_threads) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info counts_buf = query_counts.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 2 && queries_buf.ndim != 1) {
              throw std::runtime_error(
                  "queries must be a 2-dimensional array "
                  "(total_query_rows, vec_dim) or a 1-dimensional flattened array");
            }
            if (counts_buf.ndim != 1) {
              throw std::runtime_error(
                  "query_counts must be a 1-dimensional array");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool queries_contiguous = false;
            if (queries_buf.ndim == 2) {
              queries_contiguous =
                  queries_buf.strides[1] == sizeof(float) &&
                  queries_buf.strides[0] ==
                      queries_buf.shape[1] * sizeof(float);
            } else {
              queries_contiguous = queries_buf.strides[0] == sizeof(float);
            }
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int32_t *counts_ptr = static_cast<int32_t *>(counts_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = counts_buf.shape[0];
            int num_indices = indices_buf.shape[1];

            auto results = self.batch_query_subset(
                queries_ptr, num_queries, counts_ptr, k, indices_ptr,
                num_indices, num_threads);

            // Convert to 2D numpy array
            py::array_t<int> result_array({num_queries, k});
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              int actual_k = static_cast<int>(results[i].size());
              for (int j = 0; j < actual_k; j++) {
                result_ptr[i * k + j] = results[i][j];
              }
              for (int j = actual_k; j < k; j++) {
                result_ptr[i * k + j] = -1;
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("query_counts"), py::arg("k"),
          py::arg("indices"), py::arg("num_threads") = -1)

      .def(
          "rerank_subset_fixed",
          [](MaxSim &self, py::array_t<float> queries, int k,
             py::array_t<int> indices, int num_threads) {
            py::buffer_info queries_buf = queries.request();
            py::buffer_info indices_buf = indices.request();

            if (queries_buf.ndim != 3) {
              throw std::runtime_error(
                  "queries must be a 3-dimensional array (num_queries, "
                  "query_vec_count, vec_dim)");
            }
            if (indices_buf.ndim != 2) {
              throw std::runtime_error("indices must be a 2-dimensional array "
                                       "(num_queries, num_indices)");
            }

            bool queries_contiguous =
                queries_buf.strides[2] == sizeof(float) &&
                queries_buf.strides[1] == queries_buf.shape[2] * sizeof(float) &&
                queries_buf.strides[0] == queries_buf.shape[1] * queries_buf.shape[2] * sizeof(float);
            if (!queries_contiguous) {
              throw std::runtime_error(
                  "queries array must be C-contiguous (use np.ascontiguousarray)");
            }

            bool indices_contiguous =
                indices_buf.strides[1] == sizeof(int) &&
                indices_buf.strides[0] == indices_buf.shape[1] * sizeof(int);
            if (!indices_contiguous) {
              throw std::runtime_error(
                  "indices array must be C-contiguous (use np.ascontiguousarray)");
            }

            const float *queries_ptr = static_cast<float *>(queries_buf.ptr);
            const int *indices_ptr = static_cast<int *>(indices_buf.ptr);
            int num_queries = queries_buf.shape[0];
            int query_vec_count = queries_buf.shape[1];
            int num_indices = indices_buf.shape[1];

            auto results = self.batch_query_subset_fixed(
                queries_ptr, num_queries, query_vec_count, k, indices_ptr,
                num_indices, num_threads);

            // Convert to 2D numpy array
            py::array_t<int> result_array({num_queries, k});
            auto result_buf = result_array.request();
            int *result_ptr = static_cast<int *>(result_buf.ptr);

            for (int i = 0; i < num_queries; i++) {
              int actual_k = static_cast<int>(results[i].size());
              for (int j = 0; j < actual_k; j++) {
                result_ptr[i * k + j] = results[i][j];
              }
              for (int j = actual_k; j < k; j++) {
                result_ptr[i * k + j] = -1;
              }
            }

            return result_array;
          },
          py::arg("queries"), py::arg("k"), py::arg("indices"),
          py::arg("num_threads") = -1);
}
