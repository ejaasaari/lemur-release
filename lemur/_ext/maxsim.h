#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <immintrin.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

class MaxSim {
private:
  const float *train_;
  const int32_t *train_counts_;
  int vec_dim_;
  int num_train_points_;

  std::vector<int32_t> train_offsets_;
  std::vector<const float *> train_ptrs_;

public:
  static inline float dot_product(const float *x1, const float *x2,
                                  size_t length) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= length; i += 16) {
      __m512 v1 = _mm512_loadu_ps(x1 + i);
      __m512 v2 = _mm512_loadu_ps(x2 + i);
      sum = _mm512_fmadd_ps(v1, v2, sum);
    }
    if (i < length) {
      const uint32_t rem = static_cast<uint32_t>(length - i);
      const __mmask16 m = static_cast<__mmask16>((1u << rem) - 1u);
      __m512 v1 = _mm512_maskz_loadu_ps(m, x1 + i);
      __m512 v2 = _mm512_maskz_loadu_ps(m, x2 + i);
      sum = _mm512_fmadd_ps(v1, v2, sum);
    }

    auto sumh = _mm256_add_ps(_mm512_castps512_ps256(sum),
                              _mm512_extractf32x8_ps(sum, 1));
    auto sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh),
                            _mm256_extractf128_ps(sumh, 1));
    auto tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
    auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
  }

private:
  static inline float hsum512_ps(__m512 v) {
    __m256 sumh =
        _mm256_add_ps(_mm512_castps512_ps256(v), _mm512_extractf32x8_ps(v, 1));
    __m128 sumhh = _mm_add_ps(_mm256_castps256_ps128(sumh),
                              _mm256_extractf128_ps(sumh, 1));
    __m128 tmp1 = _mm_add_ps(sumhh, _mm_movehl_ps(sumhh, sumhh));
    __m128 tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
    return _mm_cvtss_f32(tmp2);
  }

  template <int D>
  static inline void transpose_query_32(const float *__restrict query,
                                        float *__restrict qT) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 64);
#endif
    for (int d = 0; d < D; ++d) {
      const float *__restrict col = query + d;
      float *__restrict out = qT + d * 32;
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
      for (int q = 0; q < 32; ++q) {
        out[q] = col[(int64_t)q * D];
      }
    }
  }

  template <int D>
  static inline float maxsim_Dx32_qT_2x(const float *__restrict qT,
                                        const float *__restrict train_block,
                                        int count) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    __m512 best0 = _mm512_set1_ps(neg_inf);
    __m512 best1 = _mm512_set1_ps(neg_inf);

#if defined(__GNUC__) || defined(__clang__)
    __builtin_assume_aligned(qT, 64);
#endif

    int tv = 0;
    for (; tv + 1 < count; tv += 2) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      const float *__restrict x1 = x0 + D;

      __m512 acc00 = _mm512_setzero_ps();
      __m512 acc01 = _mm512_setzero_ps();
      __m512 acc10 = _mm512_setzero_ps();
      __m512 acc11 = _mm512_setzero_ps();

      for (int d = 0; d < D; d += 4) {
        __m512 q00 = _mm512_load_ps(qT + (d + 0) * 32);
        __m512 q01 = _mm512_load_ps(qT + (d + 0) * 32 + 16);
        __m512 bx00 = _mm512_set1_ps(x0[d + 0]);
        __m512 bx10 = _mm512_set1_ps(x1[d + 0]);
        acc00 = _mm512_fmadd_ps(q00, bx00, acc00);
        acc01 = _mm512_fmadd_ps(q01, bx00, acc01);
        acc10 = _mm512_fmadd_ps(q00, bx10, acc10);
        acc11 = _mm512_fmadd_ps(q01, bx10, acc11);

        __m512 q10 = _mm512_load_ps(qT + (d + 1) * 32);
        __m512 q11 = _mm512_load_ps(qT + (d + 1) * 32 + 16);
        __m512 bx01 = _mm512_set1_ps(x0[d + 1]);
        __m512 bx11 = _mm512_set1_ps(x1[d + 1]);
        acc00 = _mm512_fmadd_ps(q10, bx01, acc00);
        acc01 = _mm512_fmadd_ps(q11, bx01, acc01);
        acc10 = _mm512_fmadd_ps(q10, bx11, acc10);
        acc11 = _mm512_fmadd_ps(q11, bx11, acc11);

        __m512 q20 = _mm512_load_ps(qT + (d + 2) * 32);
        __m512 q21 = _mm512_load_ps(qT + (d + 2) * 32 + 16);
        __m512 bx02 = _mm512_set1_ps(x0[d + 2]);
        __m512 bx12 = _mm512_set1_ps(x1[d + 2]);
        acc00 = _mm512_fmadd_ps(q20, bx02, acc00);
        acc01 = _mm512_fmadd_ps(q21, bx02, acc01);
        acc10 = _mm512_fmadd_ps(q20, bx12, acc10);
        acc11 = _mm512_fmadd_ps(q21, bx12, acc11);

        __m512 q30 = _mm512_load_ps(qT + (d + 3) * 32);
        __m512 q31 = _mm512_load_ps(qT + (d + 3) * 32 + 16);
        __m512 bx03 = _mm512_set1_ps(x0[d + 3]);
        __m512 bx13 = _mm512_set1_ps(x1[d + 3]);
        acc00 = _mm512_fmadd_ps(q30, bx03, acc00);
        acc01 = _mm512_fmadd_ps(q31, bx03, acc01);
        acc10 = _mm512_fmadd_ps(q30, bx13, acc10);
        acc11 = _mm512_fmadd_ps(q31, bx13, acc11);
      }

      best0 = _mm512_max_ps(best0, acc00);
      best1 = _mm512_max_ps(best1, acc01);
      best0 = _mm512_max_ps(best0, acc10);
      best1 = _mm512_max_ps(best1, acc11);
    }

    if (tv < count) {
      const float *__restrict x0 = train_block + (int64_t)tv * D;
      __m512 acc00 = _mm512_setzero_ps();
      __m512 acc01 = _mm512_setzero_ps();

      for (int d = 0; d < D; d += 4) {
        __m512 q00 = _mm512_load_ps(qT + (d + 0) * 32);
        __m512 q01 = _mm512_load_ps(qT + (d + 0) * 32 + 16);
        __m512 bx00 = _mm512_set1_ps(x0[d + 0]);
        acc00 = _mm512_fmadd_ps(q00, bx00, acc00);
        acc01 = _mm512_fmadd_ps(q01, bx00, acc01);

        __m512 q10 = _mm512_load_ps(qT + (d + 1) * 32);
        __m512 q11 = _mm512_load_ps(qT + (d + 1) * 32 + 16);
        __m512 bx01 = _mm512_set1_ps(x0[d + 1]);
        acc00 = _mm512_fmadd_ps(q10, bx01, acc00);
        acc01 = _mm512_fmadd_ps(q11, bx01, acc01);

        __m512 q20 = _mm512_load_ps(qT + (d + 2) * 32);
        __m512 q21 = _mm512_load_ps(qT + (d + 2) * 32 + 16);
        __m512 bx02 = _mm512_set1_ps(x0[d + 2]);
        acc00 = _mm512_fmadd_ps(q20, bx02, acc00);
        acc01 = _mm512_fmadd_ps(q21, bx02, acc01);

        __m512 q30 = _mm512_load_ps(qT + (d + 3) * 32);
        __m512 q31 = _mm512_load_ps(qT + (d + 3) * 32 + 16);
        __m512 bx03 = _mm512_set1_ps(x0[d + 3]);
        acc00 = _mm512_fmadd_ps(q30, bx03, acc00);
        acc01 = _mm512_fmadd_ps(q31, bx03, acc01);
      }

      best0 = _mm512_max_ps(best0, acc00);
      best1 = _mm512_max_ps(best1, acc01);
    }

    return hsum512_ps(best0) + hsum512_ps(best1);
  }

  inline float maxsim_generic(const float *__restrict query,
                              int32_t query_vec_count, int train_index) const {
    const int32_t count = train_counts_[train_index];
    const float *__restrict train_block = train_ptrs_[train_index];

    float total = 0.0f;
    for (int32_t qv = 0; qv < query_vec_count; ++qv) {
      const float *__restrict q = query + (int64_t)qv * vec_dim_;
      float best = -std::numeric_limits<float>::infinity();

      const float *__restrict x = train_block;
      for (int32_t tv = 0; tv < count; ++tv, x += vec_dim_) {
        const float s = dot_product(q, x, (size_t)vec_dim_);
        best = (s > best) ? s : best;
      }
      total += best;
    }
    return total;
  }

  inline float maxsim_one_query_one_train(const float *__restrict query,
                                          int32_t query_vec_count,
                                          float *__restrict qT_scratch,
                                          int train_index) const {
    const int32_t count = train_counts_[train_index];
    const float *__restrict train_block = train_ptrs_[train_index];

    if (query_vec_count == 32 && (vec_dim_ == 48 || vec_dim_ == 64 ||
                                  vec_dim_ == 96 || vec_dim_ == 128)) {
      if (vec_dim_ == 48)
        transpose_query_32<48>(query, qT_scratch);
      else if (vec_dim_ == 64)
        transpose_query_32<64>(query, qT_scratch);
      else if (vec_dim_ == 96)
        transpose_query_32<96>(query, qT_scratch);
      else
        transpose_query_32<128>(query, qT_scratch);

      if (vec_dim_ == 48)
        return maxsim_Dx32_qT_2x<48>(qT_scratch, train_block, count);
      else if (vec_dim_ == 64)
        return maxsim_Dx32_qT_2x<64>(qT_scratch, train_block, count);
      else if (vec_dim_ == 96)
        return maxsim_Dx32_qT_2x<96>(qT_scratch, train_block, count);
      else
        return maxsim_Dx32_qT_2x<128>(qT_scratch, train_block, count);
    }

    return maxsim_generic(query, query_vec_count, train_index);
  }

public:
  MaxSim(const float *train, const int32_t *train_counts, int vec_dim,
          int num_train_points)
      : train_(train), train_counts_(train_counts), vec_dim_(vec_dim),
        num_train_points_(num_train_points),
        train_offsets_((size_t)num_train_points + 1, 0),
        train_ptrs_((size_t)num_train_points, nullptr) {
    int32_t off = 0;
    train_offsets_[0] = 0;
    for (int i = 0; i < num_train_points_; ++i) {
      train_ptrs_[i] = train_ + (int64_t)off * vec_dim_;
      off += train_counts_[i];
      train_offsets_[i + 1] = off;
    }
  }

  std::vector<std::vector<int>>
  batch_query_subset_fixed(const float *__restrict queries, int num_queries,
                           int32_t query_vec_count, int k,
                           const int *__restrict indices_matrix,
                           int num_indices, int num_threads) const {
    std::vector<std::vector<int>> results((size_t)num_queries);
    if (num_queries <= 0 || query_vec_count <= 0 || vec_dim_ <= 0 ||
        num_indices <= 0 || k <= 0) {
      return results;
    }

    const int kk = std::min(k, num_indices);
    for (int i = 0; i < num_queries; ++i)
      results[i].resize((size_t)kk);

    const int threads =
        (num_threads == -1) ? omp_get_max_threads() : std::max(1, num_threads);

    const bool fast_dim =
        (vec_dim_ == 48 || vec_dim_ == 64 || vec_dim_ == 96 || vec_dim_ == 128);
    const bool fast_path = (query_vec_count == 32) && fast_dim;

#pragma omp parallel num_threads(threads)
    {
      std::vector<std::pair<float, int>> cand((size_t)num_indices);
      alignas(64) float qT[128 * 32];

      auto cmp = [](const std::pair<float, int> &a,
                    const std::pair<float, int> &b) {
        if (a.first != b.first)
          return a.first > b.first;
        return a.second < b.second;
      };

#pragma omp for schedule(static)
      for (int qi = 0; qi < num_queries; ++qi) {
        const float *__restrict query = queries + (int64_t)qi *
                                                      (int64_t)query_vec_count *
                                                      (int64_t)vec_dim_;
        const int *__restrict row =
            indices_matrix + (int64_t)qi * (int64_t)num_indices;

        if (fast_path) {
          if (vec_dim_ == 48)
            transpose_query_32<48>(query, qT);
          else if (vec_dim_ == 64)
            transpose_query_32<64>(query, qT);
          else if (vec_dim_ == 96)
            transpose_query_32<96>(query, qT);
          else
            transpose_query_32<128>(query, qT);

          if (vec_dim_ == 48) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<48>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else if (vec_dim_ == 64) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<64>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else if (vec_dim_ == 96) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<96>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else { // 128
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<128>(qT, train_ptrs_[idx],
                                                        train_counts_[idx]),
                                 idx};
            }
          }
        } else {
          for (int j = 0; j < num_indices; ++j) {
            const int idx = row[j];
            cand[(size_t)j] = {maxsim_generic(query, query_vec_count, idx),
                               idx};
          }
        }

        if (kk < num_indices) {
          std::nth_element(cand.begin(), cand.begin() + kk, cand.end(), cmp);
          std::sort(cand.begin(), cand.begin() + kk, cmp);
        } else {
          std::sort(cand.begin(), cand.end(), cmp);
        }

        auto &out = results[qi];
        for (int t = 0; t < kk; ++t)
          out[(size_t)t] = cand[(size_t)t].second;
      }
    }

    return results;
  }

  std::vector<std::vector<int>>
  batch_query_subset(const float *__restrict queries, int num_queries,
                     const int32_t *__restrict query_vec_counts, int k,
                     const int *__restrict indices_matrix, int num_indices,
                     int num_threads) const {
    std::vector<std::vector<int>> results((size_t)num_queries);
    if (num_queries <= 0 || vec_dim_ <= 0 || num_indices <= 0 || k <= 0) {
      return results;
    }

    const int kk = std::min(k, num_indices);
    for (int i = 0; i < num_queries; ++i)
      results[i].resize((size_t)kk);

    int requested =
        (num_threads == -1) ? omp_get_max_threads() : std::max(1, num_threads);
    int T = std::min(requested, num_queries);
    if (T <= 0)
      T = 1;

    std::vector<int> q_begin((size_t)T + 1);
    std::vector<int> q_end((size_t)T);
    {
      int base = num_queries / T;
      int rem = num_queries % T;
      int s = 0;
      for (int t = 0; t < T; ++t) {
        int take = base + (t < rem ? 1 : 0);
        q_begin[t] = s;
        q_end[t] = s + take;
        s += take;
      }
      q_begin[T] = num_queries;
    }

    std::vector<int64_t> vec_off((size_t)T + 1, 0);
    {
      int boundary_t = 1;
      int next_boundary = (T >= 1) ? q_begin[1] : num_queries;
      int64_t cum = 0;
      vec_off[0] = 0;

      for (int qi = 0; qi < num_queries; ++qi) {
        if (boundary_t <= T && qi == next_boundary) {
          vec_off[boundary_t] = cum;
          ++boundary_t;
          next_boundary = (boundary_t <= T) ? q_begin[boundary_t] : num_queries;
        }
        cum += (int64_t)query_vec_counts[qi];
      }
      vec_off[T] = cum;
    }

    const bool fast_dim =
        (vec_dim_ == 48 || vec_dim_ == 64 || vec_dim_ == 96 || vec_dim_ == 128);

#pragma omp parallel num_threads(T)
    {
      const int tid = omp_get_thread_num();
      const int qb = q_begin[tid];
      const int qe = q_end[tid];

      const float *__restrict qptr = queries + vec_off[tid] * (int64_t)vec_dim_;

      std::vector<std::pair<float, int>> cand((size_t)num_indices);
      alignas(64) float qT[128 * 32];

      auto cmp = [](const std::pair<float, int> &a,
                    const std::pair<float, int> &b) {
        if (a.first != b.first)
          return a.first > b.first;
        return a.second < b.second;
      };

      for (int qi = qb; qi < qe; ++qi) {
        const int32_t qcount = query_vec_counts[qi];
        const float *__restrict query = qptr;
        qptr += (int64_t)qcount * (int64_t)vec_dim_;

        const int *__restrict row =
            indices_matrix + (int64_t)qi * (int64_t)num_indices;

        if (qcount <= 0) {
          auto &out = results[qi];
          for (int t = 0; t < kk; ++t)
            out[(size_t)t] = row[t];
          continue;
        }

        const bool fast_path = (qcount == 32) && fast_dim;

        if (fast_path) {
          if (vec_dim_ == 48)
            transpose_query_32<48>(query, qT);
          else if (vec_dim_ == 64)
            transpose_query_32<64>(query, qT);
          else if (vec_dim_ == 96)
            transpose_query_32<96>(query, qT);
          else
            transpose_query_32<128>(query, qT);

          if (vec_dim_ == 48) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<48>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else if (vec_dim_ == 64) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<64>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else if (vec_dim_ == 96) {
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<96>(qT, train_ptrs_[idx],
                                                       train_counts_[idx]),
                                 idx};
            }
          } else { // 128
            for (int j = 0; j < num_indices; ++j) {
              const int idx = row[j];
              if (j + 1 < num_indices)
                _mm_prefetch((const char *)train_ptrs_[row[j + 1]],
                             _MM_HINT_T0);
              cand[(size_t)j] = {maxsim_Dx32_qT_2x<128>(qT, train_ptrs_[idx],
                                                        train_counts_[idx]),
                                 idx};
            }
          }
        } else {
          for (int j = 0; j < num_indices; ++j) {
            const int idx = row[j];
            cand[(size_t)j] = {maxsim_generic(query, qcount, idx), idx};
          }
        }

        if (kk < num_indices) {
          std::nth_element(cand.begin(), cand.begin() + kk, cand.end(), cmp);
          std::sort(cand.begin(), cand.begin() + kk, cmp);
        } else {
          std::sort(cand.begin(), cand.end(), cmp);
        }

        auto &out = results[qi];
        for (int t = 0; t < kk; ++t)
          out[(size_t)t] = cand[(size_t)t].second;
      }
    }

    return results;
  }
};
