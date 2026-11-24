/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>
#include <numeric>
#include "velox/common/base/ArmSveNeonGather.h"

#if XSIMD_WITH_NEON
namespace xsimd::types {
XSIMD_DECLARE_SIMD_REGISTER(
    bool,
    neon,
    detail::neon_vector_type<unsigned char>);
} // namespace xsimd::types
#endif

#if XSIMD_WITH_SVE
#include <arm_sve.h>
namespace xsimd::types {
XSIMD_DECLARE_SIMD_REGISTER(bool, sve, detail::sve_vector_type<unsigned char>);
}
#endif

DECLARE_bool(velox_arm_sve_gather_enabled);

namespace facebook::velox::simd {

namespace detail {

template <typename T, typename A>
int genericToBitMask(xsimd::batch_bool<T, A> mask) {
  static_assert(mask.size <= 32);
  alignas(A::alignment()) bool tmp[mask.size];
  mask.store_aligned(tmp);
  int ans = 0;
  for (int i = 0; i < mask.size; ++i) {
    ans |= tmp[i] << i;
  }
  return ans;
}

#if XSIMD_WITH_SVE
// xsimd uses fixed-size SVE batches. Use the batch's lane count throughout,
// including when constructing predicates, rather than assuming a 128-bit VL.
template <typename T, typename A>
int sveToBitMask(xsimd::batch_bool<T, A> mask) {
  constexpr auto N = xsimd::batch_bool<T, A>::size;
  static_assert(N <= 32);
  if constexpr (sizeof(T) == 1) {
    const auto lanes = svindex_u8(0, 1);
    const auto weights = svlsl_u8_x(
        svptrue_b8(), svdup_u8(1), svand_n_u8_x(svptrue_b8(), lanes, 7));
    uint32_t result = 0;
    for (unsigned begin = 0; begin < N; begin += 8) {
      const auto group = svand_b_z(
          mask.data,
          svcmpge_n_u8(svptrue_b8(), lanes, begin),
          svcmplt_n_u8(svptrue_b8(), lanes, begin + 8));
      result |= static_cast<uint32_t>(svaddv_u8(group, weights)) << begin;
    }
    return static_cast<int>(result);
  } else if constexpr (sizeof(T) == 2) {
    const auto weights =
        svlsl_u16_x(svptrue_b16(), svdup_u16(1), svindex_u16(0, 1));
    return svaddv_u16(mask.data, weights);
  } else if constexpr (sizeof(T) == 4) {
    const auto weights =
        svlsl_u32_x(svptrue_b32(), svdup_u32(1), svindex_u32(0, 1));
    return svaddv_u32(mask.data, weights);
  } else {
    const auto weights =
        svlsl_u64_x(svptrue_b64(), svdup_u64(1), svindex_u64(0, 1));
    return svaddv_u64(mask.data, weights);
  }
}
#endif

template <typename T, typename A>
struct FromBitMask {
  FromBitMask() {
    static_assert(N <= 8);
    for (int i = 0; i < (1 << N); ++i) {
      bool tmp[N];
      for (int bit = 0; bit < N; ++bit) {
        tmp[bit] = (i & (1 << bit)) ? true : false;
      }
      memo_[i] = xsimd::batch_bool<T, A>::load_unaligned(tmp);
    }
  }

  xsimd::batch_bool<T, A> operator[](size_t i) const {
    return memo_[i];
  }

 private:
  static constexpr int N = xsimd::batch_bool<T, A>::size;
  xsimd::batch_bool<T, A> memo_[1 << N];
};

extern const FromBitMask<int32_t, xsimd::default_arch> fromBitMask32;
extern const FromBitMask<int64_t, xsimd::default_arch> fromBitMask64;

template <typename T, typename A>
struct BitMask<T, A, 1> {
  static constexpr int kAllSet = bits::lowMask(xsimd::batch_bool<T, A>::size);

#if XSIMD_WITH_SVE
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    return sveToBitMask(mask);
  }
#endif

#if XSIMD_WITH_AVX2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::avx2&) {
    return _mm256_movemask_epi8(mask);
  }
#endif

#if XSIMD_WITH_SSE2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sse2&) {
    return _mm_movemask_epi8(mask);
  }
#endif

#if XSIMD_WITH_NEON
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::neon&) {
    static constexpr uint8x16_t vmask_const = {
        0x01,
        0x02,
        0x04,
        0x08,
        0x10,
        0x20,
        0x40,
        0x80, // lower half
        0x01,
        0x02,
        0x04,
        0x08,
        0x10,
        0x20,
        0x40,
        0x80}; // upper half
    uint8x16_t vmask = vandq_u8(mask.data, vmask_const);
    return (vaddv_u8(vget_high_u8(vmask)) << 8) | vaddv_u8(vget_low_u8(vmask));
  }
#endif

  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericToBitMask(mask);
  }
};

template <typename T, typename A>
struct BitMask<T, A, 2> {
  static constexpr int kAllSet = bits::lowMask(xsimd::batch_bool<T, A>::size);

#if XSIMD_WITH_SVE
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    return sveToBitMask(mask);
  }
#endif

#if XSIMD_WITH_AVX2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::avx2&) {
    // There is no intrinsic for extracting high bits of a 16x16
    // vector.  Hence take every second bit of the high bits of a 32x1
    // vector.
    //
    // NOTE: TVL might have a more efficient implementation for this.
    return bits::extractBits<uint32_t>(_mm256_movemask_epi8(mask), 0xAAAAAAAA);
  }
#endif

#if XSIMD_WITH_SSE2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sse2&) {
    return bits::extractBits<uint32_t>(_mm_movemask_epi8(mask), 0xAAAA);
  }
#endif

#if XSIMD_WITH_NEON
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::neon&) {
    static constexpr uint16x8_t vmask_const = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    uint16x8_t vmask = vandq_u16(mask.data, vmask_const);
    return vaddvq_u16(vmask);
  }
#endif

  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericToBitMask(mask);
  }
};

template <typename T, typename A>
struct BitMask<T, A, 4> {
  static constexpr int kAllSet = bits::lowMask(xsimd::batch_bool<T, A>::size);

#if XSIMD_WITH_SVE
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    return sveToBitMask(mask);
  }
#endif

#if XSIMD_WITH_AVX
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::avx&) {
    return _mm256_movemask_ps(reinterpret_cast<__m256>(mask.data));
  }
#endif

#if XSIMD_WITH_SSE2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sse2&) {
    return _mm_movemask_ps(reinterpret_cast<__m128>(mask.data));
  }
#endif

#if XSIMD_WITH_NEON
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::neon&) {
    static constexpr uint32x4_t vmask_const = {0x1, 0x2, 0x4, 0x8};
    uint32x4_t vmask = vandq_u32(mask.data, vmask_const);
    return vaddvq_u32(vmask);
  }
#endif

  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericToBitMask(mask);
  }

  static xsimd::batch_bool<T, A> fromBitMask(
      int mask,
      const xsimd::default_arch&) {
    return fromBitMask32[mask];
  }
};

template <typename T, typename A>
struct BitMask<T, A, 8> {
  static constexpr int kAllSet = bits::lowMask(xsimd::batch_bool<T, A>::size);

#if XSIMD_WITH_SVE
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    return sveToBitMask(mask);
  }
#endif

#if XSIMD_WITH_AVX
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::avx&) {
    return _mm256_movemask_pd(reinterpret_cast<__m256d>(mask.data));
  }
#endif

#if XSIMD_WITH_SSE2
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::sse2&) {
    return _mm_movemask_pd(reinterpret_cast<__m128d>(mask.data));
  }
#endif

#if XSIMD_WITH_NEON
  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::neon&) {
    static constexpr uint64x2_t vmask_const = {0x1, 0x2};
    const uint64x2_t vmask = vandq_u64(mask.data, vmask_const);
    return vaddvq_u64(vmask);
  }
#endif

  static int toBitMask(xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericToBitMask(mask);
  }

  static xsimd::batch_bool<T, A> fromBitMask(
      int mask,
      const xsimd::default_arch&) {
    return fromBitMask64[mask];
  }
};

} // namespace detail

template <typename A>
int32_t indicesOfSetBits(
    const uint64_t* bits,
    int32_t begin,
    int32_t end,
    int32_t* result,
    const A&) {
  if (end <= begin) {
    return 0;
  }
  int32_t row = begin & ~63;
  auto originalResult = result;
  int32_t endWord = bits::roundUp(end, 64) / 64;
  auto firstWord = begin / 64;
  for (auto wordIndex = firstWord; wordIndex < endWord; ++wordIndex) {
    uint64_t word = bits[wordIndex];
    if (!word) {
      row += 64;
      continue;
    }
    if (wordIndex == firstWord && begin != firstWord * 64) {
      word &= bits::highMask(64 - (begin - firstWord * 64));
      if (!word) {
        row += 64;
        continue;
      }
    }
    if (wordIndex == endWord - 1) {
      int32_t lastBits = end - (endWord - 1) * 64;
      if (lastBits < 64) {
        word &= bits::lowMask(lastBits);
        if (!word) {
          break;
        }
      }
    }
    if (result - originalResult < (row >> 2)) {
      do {
        *result++ = __builtin_ctzll(word) + row;
        word = word & (word - 1);
      } while (word);
      row += 64;
    } else {
      for (auto byteCnt = 0; byteCnt < 8; ++byteCnt) {
        uint8_t byte = word;
        word = word >> 8;
        if (byte) {
          using Batch = xsimd::batch<int32_t, A>;
          auto indices = byteSetBits(byte);
          if constexpr (Batch::size == 8) {
            (Batch::load_aligned(indices) + row).store_unaligned(result);
            result += __builtin_popcount(byte);
          } else {
            static_assert(Batch::size == 4);
            auto lo = byte & ((1 << 4) - 1);
            auto hi = byte >> 4;
            int pop = 0;
            if (lo) {
              (Batch::load_aligned(indices) + row).store_unaligned(result);
              pop = __builtin_popcount(lo);
              result += pop;
            }
            if (hi) {
              (Batch::load_unaligned(indices + pop) + row)
                  .store_unaligned(result);
              result += __builtin_popcount(hi);
            }
          }
        }
        row += 8;
      }
    }
  }
  return result - originalResult;
}

namespace detail {

template <typename T, typename A>
struct LeadingMask {
  LeadingMask() {
    bool tmp[N]{};
    for (int i = 0; i < N; ++i) {
      memo_[i] = xsimd::batch_bool<T, A>::load_unaligned(tmp);
      tmp[i] = true;
    }
    memo_[N] = xsimd::batch_bool<T, A>::load_unaligned(tmp);
  }

  xsimd::batch_bool<T, A> operator[](size_t i) const {
    return memo_[i];
  }

 private:
  static constexpr int N = xsimd::batch_bool<T, A>::size;
  xsimd::batch_bool<T, A> memo_[N + 1];
};

extern const LeadingMask<int32_t, xsimd::default_arch> leadingMask32;
extern const LeadingMask<int64_t, xsimd::default_arch> leadingMask64;

template <typename T, typename A>
xsimd::batch_bool<T, xsimd::default_arch> leadingMask(int i, const A&);

template <>
inline xsimd::batch_bool<int32_t, xsimd::default_arch> leadingMask(
    int i,
    const xsimd::default_arch&) {
  return leadingMask32[i];
}

template <>
inline xsimd::batch_bool<float, xsimd::default_arch> leadingMask(
    int i,
    const xsimd::default_arch&) {
  /*
  With GCC builds, compiler throws an error "invalid cast" on reintepreting to
  the same data type, in SVE 256's case, svbool_t
  __attribute__((arm_sve_vector_bits(256))).
  So this is a workaround for now. Can be updated once the bug in GCC is
  resolved in future GCC versions.
  */

#if XSIMD_WITH_SVE && defined(__GNUC__) && !defined(__clang__)
  return xsimd::batch_bool<float, xsimd::default_arch>(leadingMask32[i].data);
#else
  return reinterpret_cast<
      xsimd::batch_bool<float, xsimd::default_arch>::register_type>(
      leadingMask32[i].data);
#endif
}

template <>
inline xsimd::batch_bool<int64_t, xsimd::default_arch> leadingMask(
    int i,
    const xsimd::default_arch&) {
  return leadingMask64[i];
}

template <>
inline xsimd::batch_bool<double, xsimd::default_arch> leadingMask(
    int i,
    const xsimd::default_arch&) {
  /*
  With GCC builds, compiler throws an error "invalid cast" on reintepreting to
  the same data type, in SVE 256's case, svbool_t
  __attribute__((arm_sve_vector_bits(256))).
  So this is a workaround for now. Can be updated once the bug in GCC is
  resolved in future GCC versions.
  */

#if XSIMD_WITH_SVE && defined(__GNUC__) && !defined(__clang__)
  return xsimd::batch_bool<double, xsimd::default_arch>(leadingMask64[i].data);
#else
  return reinterpret_cast<
      xsimd::batch_bool<double, xsimd::default_arch>::register_type>(
      leadingMask64[i].data);
#endif
}

} // namespace detail

template <typename T, typename A>
xsimd::batch_bool<T, A> leadingMask(int n, const A& arch) {
  constexpr int N = xsimd::batch_bool<T, A>::size;
  return detail::leadingMask<T, A>(std::min(n, N), arch);
}

namespace detail {

template <typename T, typename A>
struct CopyWord {
  static void apply(void* to, const void* from) {
    *reinterpret_cast<T*>(to) = *reinterpret_cast<const T*>(from);
  }
};

template <typename A>
struct CopyWord<xsimd::batch<int8_t, A>, A> {
  static void apply(void* to, const void* from) {
    xsimd::batch<int8_t, A>::load_unaligned(
        reinterpret_cast<const int8_t*>(from))
        .store_unaligned(reinterpret_cast<int8_t*>(to));
  }
};

// Copies one element of T and advances 'to', 'from', and 'bytes' by
// sizeof(T). Returns false if 'bytes' went to 0.
template <typename T, typename A>
inline bool copyNextWord(void*& to, const void*& from, int64_t& bytes) {
  if (bytes >= static_cast<int64_t>(sizeof(T))) {
    CopyWord<T, A>::apply(to, from);
    bytes -= sizeof(T);
    if (!bytes) {
      return false;
    }
    from = addBytes(from, sizeof(T));
    to = addBytes(to, sizeof(T));
    return true;
  }
  return true;
}
} // namespace detail

template <typename A>
inline void memcpy(void* to, const void* from, int64_t bytes, const A& arch) {
  while (bytes >= batchByteSize(arch)) {
    if (!detail::copyNextWord<xsimd::batch<int8_t, A>, A>(to, from, bytes)) {
      return;
    }
  }
  while (bytes >= static_cast<int64_t>(sizeof(int64_t))) {
    if (!detail::copyNextWord<int64_t, A>(to, from, bytes)) {
      return;
    }
  }
  if (!detail::copyNextWord<int32_t, A>(to, from, bytes)) {
    return;
  }
  if (!detail::copyNextWord<int16_t, A>(to, from, bytes)) {
    return;
  }
  detail::copyNextWord<int8_t, A>(to, from, bytes);
}

namespace detail {

template <typename T, typename A>
struct SetWord {
  static void apply(void* to, T data) {
    *reinterpret_cast<T*>(to) = data;
  }
};

template <typename A>
struct SetWord<xsimd::batch<int8_t, A>, A> {
  static void apply(void* to, xsimd::batch<int8_t, A> data) {
    data.store_unaligned(reinterpret_cast<int8_t*>(to));
  }
};

template <typename T, typename A>
inline bool setNextWord(void*& to, T data, int32_t& bytes, const A&) {
  if (bytes >= sizeof(T)) {
    SetWord<T, A>::apply(to, data);
    bytes -= sizeof(T);
    if (!bytes) {
      return false;
    }
    to = addBytes(to, sizeof(T));
    return true;
  }
  return true;
}

} // namespace detail

template <typename A>
void memset(void* to, char data, int32_t bytes, const A& arch) {
  auto v = xsimd::batch<int8_t, A>::broadcast(data);
  while (bytes >= batchByteSize(arch)) {
    if (!detail::setNextWord(to, v, bytes, arch)) {
      return;
    }
  }
  int64_t data64 = *reinterpret_cast<int64_t*>(&v);
  while (bytes >= static_cast<int64_t>(sizeof(int64_t))) {
    if (!detail::setNextWord<int64_t>(to, data64, bytes, arch)) {
      return;
    }
  }
  if (!detail::setNextWord<int32_t>(to, data64, bytes, arch)) {
    return;
  }
  if (!detail::setNextWord<int16_t>(to, data64, bytes, arch)) {
    return;
  }
  detail::setNextWord<int8_t>(to, data64, bytes, arch);
}

namespace detail {

template <typename T, typename A, int kScale, typename IndexType>
xsimd::batch<T, A> genericGather(const T* base, const IndexType* indices) {
  constexpr int N = xsimd::batch<T, A>::size;
  alignas(A::alignment()) T dst[N];
  auto bytes = reinterpret_cast<const char*>(base);
  for (int i = 0; i < N; ++i) {
    dst[i] = *reinterpret_cast<const T*>(bytes + indices[i] * kScale);
  }
  return xsimd::load_aligned(dst);
}

template <typename T, typename A, int kScale, typename IndexType>
xsimd::batch<T, A> genericMaskGather(
    xsimd::batch<T, A> src,
    xsimd::batch_bool<T, A> mask,
    const T* base,
    const IndexType* indices) {
  constexpr int N = xsimd::batch<T, A>::size;
  alignas(A::alignment()) T dst[N];
  alignas(A::alignment()) T sr[N];
  alignas(A::alignment()) bool ma[N];
  src.store_aligned(sr);
  mask.store_aligned(ma);
  auto bytes = reinterpret_cast<const char*>(base);
  for (int i = 0; i < N; ++i) {
    if (ma[i]) {
      dst[i] = *reinterpret_cast<const T*>(bytes + indices[i] * kScale);
    } else {
      dst[i] = sr[i];
    }
  }
  return xsimd::load_aligned(dst);
}

// GCC 12 and earlier have incomplete support for these SVE gather forms.
// Allow builds and regression tests to explicitly select the fallback.
#ifndef VELOX_SVE_GATHER_INTRINSICS_SAFE
#if XSIMD_WITH_SVE && \
    (defined(__clang__) || (defined(__GNUC__) && __GNUC__ > 12))
#define VELOX_SVE_GATHER_INTRINSICS_SAFE 1
#else
#define VELOX_SVE_GATHER_INTRINSICS_SAFE 0
#endif
#endif

#if XSIMD_WITH_SVE && VELOX_SVE_GATHER_INTRINSICS_SAFE
template <typename T, typename A, int kScale, typename IndexVector>
xsimd::batch<T, A> sveMaskGather(
    xsimd::batch<T, A> src,
    xsimd::batch_bool<T, A> mask,
    const T* base,
    IndexVector indices) {
  if constexpr (sizeof(T) == 4) {
    const auto source = xsimd::bitwise_cast<xsimd::batch<uint32_t, A>>(src);
    svuint32_t loaded;
    if constexpr (kScale == 4) {
      loaded = svld1_gather_s32index_u32(
          mask.data, reinterpret_cast<const uint32_t*>(base), indices);
    } else {
      loaded = svld1_gather_s32offset_u32(
          mask.data,
          reinterpret_cast<const uint32_t*>(base),
          svmul_n_s32_x(mask.data, indices, kScale));
    }
    const xsimd::batch<uint32_t, A> result(
        svsel_u32(mask.data, loaded, source.data));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(result);
  } else {
    const auto source = xsimd::bitwise_cast<xsimd::batch<uint64_t, A>>(src);
    const auto loaded = svld1_gather_s64offset_u64(
        mask.data,
        reinterpret_cast<const uint64_t*>(base),
        svmul_n_s64_x(mask.data, indices, kScale));
    const xsimd::batch<uint64_t, A> result(
        svsel_u64(mask.data, loaded, source.data));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(result);
  }
}

template <typename T, typename IndexType, typename A, int kScale>
xsimd::batch<T, A> sveMaskGatherFromMemory(
    xsimd::batch<T, A> src,
    xsimd::batch_bool<T, A> mask,
    const T* base,
    const IndexType* indices) {
  if constexpr (sizeof(T) == 4) {
    return sveMaskGather<T, A, kScale>(
        src, mask, base, svld1_s32(mask.data, indices));
  } else if constexpr (sizeof(IndexType) == 4) {
    // Sign-extend one 32-bit index per 64-bit result lane. This also avoids
    // overreading the Batch64/Batch128 index buffers.
    return sveMaskGather<T, A, kScale>(
        src, mask, base, svld1sw_s64(mask.data, indices));
  } else {
    return sveMaskGather<T, A, kScale>(
        src, mask, base, svld1_s64(mask.data, indices));
  }
}
#endif

template <typename T, typename A>
struct Gather<T, int32_t, A, 2> {
  using VIndexType = xsimd::batch<int32_t, A>;

  // Load 8 indices only.
  static VIndexType loadIndices(const int32_t* indices, const A& arch) {
    return Gather<int32_t, int32_t, A>::loadIndices(indices, arch);
  }
};

template <typename T, typename A>
struct Gather<T, int32_t, A, 4> {
  using VIndexType = xsimd::batch<int32_t, A>;

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::sve& arch) {
    if constexpr (sizeof(T) == 8 && xsimd::batch<T, A>::size == 2) {
      // Two independent scalar loads are faster than an unmasked SVE gather
      // on 128-bit implementations. Keep predicated gathers vectorized.
      return genericGather<T, A, kScale>(base, indices);
    } else {
      return maskApply<kScale>(
          xsimd::batch<T, A>(T{}),
          xsimd::batch_bool<T, A>(true),
          base,
          indices,
          arch);
    }
  }
#endif

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType indices, const xsimd::sve& arch) {
    return maskApply<kScale>(
        xsimd::batch<T, A>(T{}),
        xsimd::batch_bool<T, A>(true),
        base,
        indices,
        arch);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType indices,
      const xsimd::sve&) {
#if VELOX_SVE_GATHER_INTRINSICS_SAFE
    return sveMaskGather<T, A, kScale>(src, mask, base, indices.data);
#else
    alignas(A::alignment()) int32_t indexArray[VIndexType::size];
    indices.store_aligned(indexArray);
    return genericMaskGather<T, A, kScale>(src, mask, base, indexArray);
#endif
  }
#endif

  static VIndexType loadIndices(const int32_t* indices, const xsimd::generic&) {
    return xsimd::load_unaligned<A>(indices);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::avx2& arch) {
    return apply<kScale>(base, loadIndices(indices, arch), arch);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, indices);
  }

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType vindex, const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_i32gather_epi32(
            reinterpret_cast<const int32_t*>(base), vindex, kScale));
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType vindex, const xsimd::generic&) {
    alignas(A::alignment()) int32_t indices[vindex.size];
    vindex.store_aligned(indices);
    return genericGather<T, A, kScale>(base, indices);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::avx2& arch) {
    return maskApply<kScale>(src, mask, base, loadIndices(indices, arch), arch);
  }

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::sve& arch) {
#if VELOX_SVE_GATHER_INTRINSICS_SAFE
    return sveMaskGatherFromMemory<T, int32_t, A, kScale>(
        src, mask, base, indices);
#else
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
#endif
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
  }

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_mask_i32gather_epi32(
            reinterpret_cast<__m256i>(src.data),
            reinterpret_cast<const int32_t*>(base),
            vindex,
            reinterpret_cast<__m256i>(mask.data),
            kScale));
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::generic&) {
    alignas(A::alignment()) int32_t indices[vindex.size];
    vindex.store_aligned(indices);
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
  }

#if VELOX_SVE_NEON_GATHER_AVAILABLE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::neon& arch) {
    return apply<kScale>(base, VIndexType::load_unaligned(indices), arch);
  }

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType indices, const xsimd::neon&) {
    if constexpr (kScale == 1 || kScale == 4) {
      if (FLAGS_velox_arm_sve_gather_enabled) {
        return arm_detail::gather32<T, kScale>(base, indices.data);
      }
    }
    alignas(A::alignment()) int32_t scalar[VIndexType::size];
    indices.store_aligned(scalar);
    return genericGather<T, A, kScale>(base, scalar);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> source,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::neon& arch) {
    return maskApply<kScale>(
        source, mask, base, VIndexType::load_unaligned(indices), arch);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> source,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType indices,
      const xsimd::neon&) {
    if constexpr (kScale == 1 || kScale == 4) {
      if (FLAGS_velox_arm_sve_gather_enabled) {
        return arm_detail::maskGather32<T, kScale>(
            source.data, mask.data, base, indices.data);
      }
    }
    alignas(A::alignment()) int32_t scalar[VIndexType::size];
    indices.store_aligned(scalar);
    return genericMaskGather<T, A, kScale>(source, mask, base, scalar);
  }
#endif
};

template <typename T, typename A>
struct Gather<T, int32_t, A, 8> {
#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::sve& arch) {
    if constexpr (sizeof(T) == 8 && xsimd::batch<T, A>::size == 2) {
      // Two independent scalar loads are faster than an unmasked SVE gather
      // on 128-bit implementations. Keep predicated gathers vectorized.
      return genericGather<T, A, kScale>(base, indices);
    } else {
      return maskApply<kScale>(
          xsimd::batch<T, A>(T{}),
          xsimd::batch_bool<T, A>(true),
          base,
          indices,
          arch);
    }
  }
#endif
#if XSIMD_WITH_AVX
  static xsimd::batch<int32_t, xsimd::sse2> loadIndices(
      const int32_t* indices,
      const xsimd::avx&) {
    return _mm_lddqu_si128(reinterpret_cast<const __m128i*>(indices));
  }
#endif

  static Batch64<int32_t> loadIndices(
      const int32_t* indices,
      const xsimd::sse2&) {
    return Batch64<int32_t>::load_unaligned(indices);
  }

#if (XSIMD_WITH_SVE && SVE_BITS == 128)

  static Batch64<int32_t> loadIndices(
      const int32_t* indices,
      const xsimd::sve&) {
    return Batch64<int32_t>::load_unaligned(indices);
  }
#endif
#if (XSIMD_WITH_SVE && SVE_BITS == 256)
  static Batch128<int32_t> loadIndices(
      const int32_t* indices,
      const xsimd::sve&) {
    return Batch128<int32_t>::load_unaligned(indices);
  }
#endif

  static Batch64<int32_t> loadIndices(
      const int32_t* indices,
      const xsimd::neon&) {
    return Batch64<int32_t>::load_unaligned(indices);
  }

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::avx2& arch) {
    return apply<kScale>(base, loadIndices(indices, arch), arch);
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, indices);
  }

#if (XSIMD_WITH_SVE && SVE_BITS == 256)
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, Batch128<int32_t> vindex, const xsimd::sve& arch) {
    return apply<kScale>(base, vindex.data, arch);
  }
#endif

#if (XSIMD_WITH_SVE && SVE_BITS == 128)
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, Batch64<int32_t> vindex, const xsimd::sve& arch) {
    return apply<kScale>(base, vindex.data, arch);
  }
#endif

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A> apply(
      const T* base,
      xsimd::batch<int32_t, xsimd::sse2> vindex,
      const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_i32gather_epi64(
            reinterpret_cast<const long long*>(base), vindex, kScale));
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
  }

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::sve& arch) {
#if VELOX_SVE_GATHER_INTRINSICS_SAFE
    return sveMaskGatherFromMemory<T, int32_t, A, kScale>(
        src, mask, base, indices);
#else
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
#endif
  }
#endif

#if (XSIMD_WITH_SVE && SVE_BITS == 128)
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      Batch64<int32_t> vindex,
      const xsimd::sve& arch) {
    constexpr int N = Batch64<int32_t>::size;
    alignas(A::alignment()) int32_t indices[N];
    vindex.store_unaligned(indices);
    return maskApply<kScale>(src, mask, base, indices, arch);
  }
#endif

#if (XSIMD_WITH_SVE && SVE_BITS == 256)
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      Batch128<int32_t> vindex,
      const xsimd::sve& arch) {
    constexpr int N = Batch128<int32_t>::size;
    alignas(A::alignment()) int32_t indices[N];
    vindex.store_unaligned(indices);
    return maskApply<kScale>(src, mask, base, indices, arch);
  }
#endif

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::avx2& arch) {
    return maskApply<kScale>(src, mask, base, loadIndices(indices, arch), arch);
  }
#endif

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      xsimd::batch<int32_t, xsimd::sse2> vindex,
      const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_mask_i32gather_epi64(
            reinterpret_cast<__m256i>(src.data),
            reinterpret_cast<const long long*>(base),
            vindex,
            reinterpret_cast<__m256i>(mask.data),
            kScale));
  }
#endif

#if VELOX_SVE_NEON_GATHER_AVAILABLE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int32_t* indices, const xsimd::neon&) {
    if (FLAGS_velox_arm_sve_gather_enabled) {
      return arm_detail::gather64<T, kScale>(base, vld1_s32(indices));
    }
    return genericGather<T, A, kScale>(base, indices);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> source,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int32_t* indices,
      const xsimd::neon&) {
    if (FLAGS_velox_arm_sve_gather_enabled) {
      return arm_detail::maskGather64<T, kScale>(
          source.data, mask.data, base, vld1_s32(indices));
    }
    return genericMaskGather<T, A, kScale>(source, mask, base, indices);
  }
#endif
};

template <typename T, typename A>
struct Gather<T, int64_t, A, 8> {
  using VIndexType = xsimd::batch<int64_t, A>;

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType indices, const xsimd::generic&) {
    alignas(A::alignment()) int64_t indexArray[VIndexType::size];
    indices.store_aligned(indexArray);
    return genericGather<T, A, kScale>(base, indexArray);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int64_t* indices,
      const xsimd::generic&) {
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
  }

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int64_t* indices, const xsimd::sve& arch) {
    if constexpr (sizeof(T) == 8 && xsimd::batch<T, A>::size == 2) {
      // Two independent scalar loads are faster than an unmasked SVE gather
      // on 128-bit implementations. Keep predicated gathers vectorized.
      return genericGather<T, A, kScale>(base, indices);
    } else {
      return maskApply<kScale>(
          xsimd::batch<T, A>(T{}),
          xsimd::batch_bool<T, A>(true),
          base,
          indices,
          arch);
    }
  }
#endif

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType indices, const xsimd::sve& arch) {
    return maskApply<kScale>(
        xsimd::batch<T, A>(T{}),
        xsimd::batch_bool<T, A>(true),
        base,
        indices,
        arch);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType indices,
      const xsimd::sve&) {
#if VELOX_SVE_GATHER_INTRINSICS_SAFE
    return sveMaskGather<T, A, kScale>(src, mask, base, indices.data);
#else
    alignas(A::alignment()) int64_t indexArray[VIndexType::size];
    indices.store_aligned(indexArray);
    return genericMaskGather<T, A, kScale>(src, mask, base, indexArray);
#endif
  }
#endif

  static VIndexType loadIndices(const int64_t* indices, const xsimd::generic&) {
    return xsimd::load_unaligned<A>(indices);
  }

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int64_t* indices, const xsimd::avx2& arch) {
    return apply<kScale>(base, loadIndices(indices, arch), arch);
  }
#endif

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, VIndexType vindex, const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_i64gather_epi64(
            reinterpret_cast<const long long*>(base), vindex, kScale));
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A>
  apply(const T* base, const int64_t* indices, const xsimd::generic&) {
    return genericGather<T, A, kScale>(base, indices);
  }

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int64_t* indices,
      const xsimd::avx2& arch) {
    return maskApply<kScale>(src, mask, base, loadIndices(indices, arch), arch);
  }

#if XSIMD_WITH_SVE
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      const int64_t* indices,
      const xsimd::sve& arch) {
#if VELOX_SVE_GATHER_INTRINSICS_SAFE
    return sveMaskGatherFromMemory<T, int64_t, A, kScale>(
        src, mask, base, indices);
#else
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
#endif
  }
#endif

#if XSIMD_WITH_AVX2
  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_mask_i64gather_epi64(
            src,
            reinterpret_cast<const long long*>(base),
            vindex,
            mask,
            kScale));
  }
#endif

  template <int kScale>
  static xsimd::batch<T, A> maskApply(
      xsimd::batch<T, A> src,
      xsimd::batch_bool<T, A> mask,
      const T* base,
      VIndexType vindex,
      const xsimd::generic&) {
    alignas(A::alignment()) int64_t indices[vindex.size];
    vindex.store_aligned(indices);
    return genericMaskGather<T, A, kScale>(src, mask, base, indices);
  }
};

// Concatenates the low 16 bits of each lane in 'x' and 'y' and
// returns the result as 16x16 bits.
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::generic&);

#if XSIMD_WITH_SSE4_1
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::sse4_1&) {
  return _mm_packus_epi32(x & 0xFFFF, y & 0xFFFF);
}
#endif

#if XSIMD_WITH_NEON
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::neon&) {
  return vcombine_s16(vmovn_s32(x), vmovn_s32(y));
}
#endif

#if XSIMD_WITH_SVE
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::sve&) {
  return svuzp1_s16(
      svreinterpret_s16_s32(x.data), svreinterpret_s16_s32(y.data));
}
#endif

template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::generic&) {
  constexpr std::size_t size = xsimd::batch<int32_t, A>::size;
  alignas(A) int32_t xArr[size];
  alignas(A) int32_t yArr[size];
  alignas(A) int16_t resultArr[2 * size];

  x.store_unaligned(xArr);
  y.store_unaligned(yArr);

  for (std::size_t i = 0; i < size; ++i) {
    resultArr[i] = static_cast<int16_t>(xArr[i]);
    resultArr[i + size] = static_cast<int16_t>(yArr[i]);
  }
  return xsimd::batch<int16_t, A>::load_unaligned(resultArr);
}

#if XSIMD_WITH_AVX2
template <typename A>
xsimd::batch<int16_t, A> pack32(
    xsimd::batch<int32_t, A> x,
    xsimd::batch<int32_t, A> y,
    const xsimd::avx2&) {
  constexpr int64_t k64Low16 = 0x0000ffff0000ffff;
  auto lows = _mm256_inserti128_si256(x, _mm256_extracti128_si256(y, 0), 1);
  auto highs = _mm256_inserti128_si256(y, _mm256_extracti128_si256(x, 1), 0);
  return _mm256_packus_epi32(lows & k64Low16, highs & k64Low16);
}
#endif

template <typename T, typename A>
xsimd::batch<T, A> genericPermute(xsimd::batch<T, A> data, const int32_t* idx) {
  constexpr int N = xsimd::batch<T, A>::size;
  alignas(A::alignment()) T src[N];
  alignas(A::alignment()) T dst[N];
  data.store_aligned(src);
  for (int i = 0; i < N; ++i) {
    dst[i] = src[idx[i]];
  }
  return xsimd::load_aligned<A>(dst);
}

template <typename T, typename A>
xsimd::batch<T, A> genericPermute(
    xsimd::batch<T, A> data,
    xsimd::batch<int32_t, A> idx) {
  static_assert(data.size >= idx.size);
  alignas(A::alignment()) int32_t pos[idx.size];
  idx.store_aligned(pos);
  return genericPermute(data, pos);
}

template <typename T>
Batch64<T> genericPermute(Batch64<T> data, Batch64<int32_t> idx) {
  static_assert(data.size >= idx.size);
  Batch64<T> ans;
  for (size_t i = 0; i < idx.size; ++i) {
    ans.data[i] = data.data[idx.data[i]];
  }
  return ans;
}

template <typename T>
Batch128<T> genericPermute(Batch128<T> data, Batch128<int32_t> idx) {
  static_assert(data.size >= idx.size);
  Batch128<T> ans;
  for (int i = 0; i < idx.size; ++i) {
    ans.data[i] = data.data[idx.data[i]];
  }
  return ans;
}

template <typename T, typename A, size_t kSizeT = sizeof(T)>
struct Permute;

template <typename T, typename A>
struct Permute<T, A, 4> {
  static xsimd::batch<T, A> apply(
      xsimd::batch<T, A> data,
      xsimd::batch<int32_t, A> idx,
      const xsimd::generic&) {
    return genericPermute(data, idx);
  }

  static HalfBatch<T, A> apply(
      HalfBatch<T, A> data,
      HalfBatch<int32_t, A> idx,
      const xsimd::generic&) {
    return genericPermute(data, idx);
  }

#if XSIMD_WITH_AVX2
  static xsimd::batch<T, A> apply(
      xsimd::batch<T, A> data,
      xsimd::batch<int32_t, A> idx,
      const xsimd::avx2&) {
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_permutevar8x32_epi32(reinterpret_cast<__m256i>(data.data), idx));
  }
#endif

#if XSIMD_WITH_AVX
  static HalfBatch<T, A>
  apply(HalfBatch<T, A> data, HalfBatch<int32_t, A> idx, const xsimd::avx&) {
    return reinterpret_cast<typename HalfBatch<T, A>::register_type>(
        _mm_permutevar_ps(reinterpret_cast<__m128>(data.data), idx));
  }
#endif

#if XSIMD_WITH_NEON64
  // TBL counterpart of _mm256_permutevar8x32_epi32: lane i takes data[idx[i]].
  static xsimd::batch<T, A> apply(
      xsimd::batch<T, A> data,
      xsimd::batch<int32_t, A> idx,
      const xsimd::neon&) {
    const uint8x16_t bytes = vreinterpretq_u8_u32(vmlaq_n_u32(
        vdupq_n_u32(0x03020100),
        vreinterpretq_u32_s32(idx.data),
        0x04040404));
    using U = xsimd::batch<uint8_t, A>;
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(
        U(vqtbl1q_u8(xsimd::bitwise_cast<U>(data).data, bytes)));
  }
#endif
};

} // namespace detail

template <int kScale, typename A>
xsimd::batch<int16_t, A> gather(
    const int16_t* base,
    const int32_t* indices,
    int numIndices,
    const A& arch) {
  auto first = maskGather<int32_t, int32_t, kScale>(
      xsimd::batch<int32_t, A>::broadcast(0),
      leadingMask<int32_t>(numIndices, arch),
      reinterpret_cast<const int32_t*>(base),
      indices,
      arch);
  xsimd::batch<int32_t, A> second;
  constexpr int kIndicesBatchSize = xsimd::batch<int32_t, A>::size;
  if (numIndices > kIndicesBatchSize) {
    second = maskGather<int32_t, int32_t, kScale>(
        xsimd::batch<int32_t, A>::broadcast(0),
        leadingMask<int32_t>(numIndices - kIndicesBatchSize, arch),
        reinterpret_cast<const int32_t*>(base),
        indices + kIndicesBatchSize,
        arch);
  } else {
    second = xsimd::batch<int32_t, A>::broadcast(0);
  }
  auto packed = detail::pack32(first, second, arch);
  return packed;
}

namespace detail {

template <typename A>
uint8_t gather8BitsImpl(
    const void* bits,
    xsimd::batch<int32_t, A> vindex,
    int32_t numIndices,
    const xsimd::generic&) {
  alignas(A::alignment()) int32_t indices[vindex.size];
  vindex.store_aligned(indices);
  auto base = reinterpret_cast<const char*>(bits);
  uint8_t ans = 0;
  for (int i = 0, n = std::min<int>(vindex.size, numIndices); i < n; ++i) {
    bits::setBit(&ans, i, bits::isBitSet(base, indices[i]));
  }
  return ans;
}

#if XSIMD_WITH_AVX2
template <typename A>
uint8_t gather8BitsImpl(
    const void* bits,
    xsimd::batch<int32_t, A> vindex,
    int32_t numIndices,
    const xsimd::avx2&) {
  // Gather aligned 32-bit words, then select the requested bit in each word.
  // A byte-addressed 32-bit gather can read past the bitmap's last byte.
  // Word-aligned addresses stay within its complete uint64_t backing words.
  // Mask inactive lanes before gathering so their indices need not point
  // to readable memory.
  //
  // A plain local, not a function-local static: a static's initialization
  // guard is checked on every call, and it keeps this function from inlining
  // into gatherBits(), which calls it once per 8 bits.
  const auto wordBits = xsimd::batch<int32_t, A>::broadcast(1);
  auto maskV = wordBits << (vindex & 31);
  auto zero = xsimd::batch<int32_t, A>::broadcast(0);
  auto data = detail::Gather<int32_t, int32_t, A>::template maskApply<4>(
      zero,
      leadingMask<int32_t>(numIndices, A{}),
      reinterpret_cast<const int32_t*>(bits),
      vindex >> 5,
      A{});
  return allSetBitMask<int32_t>(A{}) ^ toBitMask((data & maskV) == zero, A{});
}
#endif

} // namespace detail

template <typename A>
uint8_t gather8Bits(
    const void* bits,
    xsimd::batch<int32_t, A> vindex,
    int32_t numIndices,
    const A& arch) {
  return detail::gather8BitsImpl(bits, vindex, numIndices, arch);
}

namespace detail {

template <typename T, typename A>
xsimd::batch<T, A> genericMaskLoad(
    const T* addr,
    xsimd::batch_bool<T, A> mask) {
  return xsimd::select<T, A>(
      mask, xsimd::load_unaligned<A, T>(addr), xsimd::broadcast<T, A>(0));
}

template <typename T, typename A>
struct MaskLoad<T, A, 4> {
#if XSIMD_WITH_SVE
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    const xsimd::batch<uint32_t, A> values(
        svld1_u32(mask.data, reinterpret_cast<const uint32_t*>(addr)));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(values);
  }
#endif
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericMaskLoad(addr, mask);
  }

#if XSIMD_WITH_AVX2
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::avx2&) {
    return _mm256_maskload_epi32(addr, mask);
  }
#endif
};

template <typename T, typename A>
struct MaskLoad<T, A, 8> {
#if XSIMD_WITH_SVE
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::sve&) {
    const xsimd::batch<uint64_t, A> values(
        svld1_u64(mask.data, reinterpret_cast<const uint64_t*>(addr)));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(values);
  }
#endif
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::generic&) {
    return genericMaskLoad(addr, mask);
  }

#if XSIMD_WITH_AVX2
  static xsimd::batch<T, A>
  apply(const T* addr, xsimd::batch_bool<T, A> mask, const xsimd::avx2&) {
    return _mm256_maskload_epi64(addr, mask);
  }
#endif
};

} // namespace detail

namespace detail {

template <typename A>
struct GetHalf<int64_t, int32_t, A> {
#if XSIMD_WITH_SVE
  template <bool kSecond>
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::sve&) {
    if constexpr (kSecond) {
      return svunpkhi_s64(data.data);
    } else {
      return svunpklo_s64(data.data);
    }
  }
#endif
#if XSIMD_WITH_AVX2
  template <bool kSecond>
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::avx2&) {
    return _mm256_cvtepi32_epi64(_mm256_extracti128_si256(data, kSecond));
  }
#endif

#if XSIMD_WITH_SSE4_1
  template <bool kSecond>
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::sse4_1&) {
    return _mm_cvtepi32_epi64(
        _mm_set_epi64x(0, _mm_extract_epi64(data, kSecond)));
  }
#endif

  template <bool kSecond>
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::generic&) {
    constexpr std::size_t input_size = xsimd::batch<int32_t, A>::size;
    constexpr std::size_t half_size = input_size / 2;

    std::array<int32_t, input_size> input_buffer;
    data.store_aligned(input_buffer.data());

    std::array<int64_t, half_size> output_buffer;
    for (std::size_t i = 0; i < half_size; ++i) {
      output_buffer[i] = static_cast<int64_t>(
          kSecond ? input_buffer[i + half_size] : input_buffer[i]);
    }

    return xsimd::load_aligned(output_buffer.data());
  }

#if XSIMD_WITH_NEON
  template <bool kSecond>
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::neon&) {
    int32x2_t half;
    if constexpr (!kSecond) {
      half = vget_low_s32(data);
    } else {
      half = vget_high_s32(data);
    }
    return vmovl_s32(half);
  }
#endif
};

template <typename A>
struct GetHalf<uint64_t, int32_t, A> {
#if XSIMD_WITH_SVE
  template <bool kSecond>
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::sve&) {
    const auto values = svreinterpret_u32_s32(data.data);
    if constexpr (kSecond) {
      return svunpkhi_u64(values);
    } else {
      return svunpklo_u64(values);
    }
  }
#endif
#if XSIMD_WITH_AVX2
  template <bool kSecond>
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::avx2&) {
    return _mm256_cvtepu32_epi64(_mm256_extracti128_si256(data, kSecond));
  }
#endif

#if XSIMD_WITH_SSE4_1
  template <bool kSecond>
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::sse4_1&) {
    return _mm_cvtepu32_epi64(
        _mm_set_epi64x(0, _mm_extract_epi64(data, kSecond)));
  }
#endif

#if XSIMD_WITH_NEON
  template <bool kSecond>
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::neon&) {
    int32x2_t half;
    if constexpr (!kSecond) {
      half = vget_low_s32(data);
    } else {
      half = vget_high_s32(data);
    }
    return vmovl_u32(vreinterpret_u32_s32(half));
  }
#endif

  template <bool kSecond>
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::generic&) {
    constexpr std::size_t input_size = xsimd::batch<int32_t, A>::size;
    constexpr std::size_t half_size = input_size / 2;
    std::array<int32_t, input_size> input_buffer;
    data.store_aligned(input_buffer.data());
    std::array<uint64_t, half_size> output_buffer;
    for (std::size_t i = 0; i < half_size; ++i) {
      output_buffer[i] = static_cast<uint64_t>(
          kSecond ? static_cast<uint32_t>(input_buffer[i + half_size])
                  : static_cast<uint32_t>(input_buffer[i]));
    }
    return xsimd::load_aligned(output_buffer.data());
  }
};

} // namespace detail

namespace detail {

// Indices to use in 8x32 bit permute for extracting words from 4x64
// bits.  The entry at 5 (bits 0 and 2 set) is {0, 1, 4, 5, 4, 5, 6,
// 7}, meaning 64 bit words at 0 and 2 are moved in front (to 0, 1).
extern int32_t permute4x64Indices[16][8];

#if XSIMD_WITH_AVX2
template <typename A, int kLane>
__m128i
filterHalf(xsimd::batch<int16_t, A> data, int mask, const xsimd::avx2&) {
  xsimd::batch<int32_t, A> data32 =
      _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data, kLane));
  auto out32 = filter(data32, mask, A{});
  return _mm_packs_epi32(
      _mm256_extractf128_si256(out32, 0), _mm256_extractf128_si256(out32, 1));
}
#endif

template <typename T, typename A>
struct Filter<T, A, 2> {
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::generic&) {
    return genericPermute(data, byteSetBits[mask]);
  }

#if XSIMD_WITH_AVX2
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::avx2& arch) {
    xsimd::batch<T, A> ans;
    auto mask1 = mask & 0xFF;
    *reinterpret_cast<__m128i_u*>(&ans) =
        detail::filterHalf<A, 0>(data, mask1, arch);
    *reinterpret_cast<__m128i_u*>(
        reinterpret_cast<int16_t*>(&ans) + __builtin_popcount(mask1)) =
        detail::filterHalf<A, 1>(data, mask >> 8, arch);
    return ans;
  }
#endif

#if XSIMD_WITH_SVE
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::sve& arch) {
    constexpr int lane_count = xsimd::batch<T, A>::size;
    if constexpr (lane_count == 8) {
      // COMPACT has only 32/64-bit forms. Narrow the existing permutation
      // table to 16-bit indices and use TBL for eight 16-bit lanes.
      const auto low = svld1_s32(svptrue_b32(), byteSetBits[mask]);
      const auto high = svld1_s32(svptrue_b32(), byteSetBits[mask] + 4);
      const auto indices =
          svuzp1_u16(svreinterpret_u16_s32(low), svreinterpret_u16_s32(high));
      using U = xsimd::batch<uint16_t, A>;
      const auto values = xsimd::bitwise_cast<U>(data);
      return xsimd::bitwise_cast<xsimd::batch<T, A>>(
          U(svtbl_u16(values.data, indices)));
    }
    T compressed[lane_count]{};
    int idx = 0;
    for (int i = 0; i < lane_count; i++) {
      if (mask & (1 << i)) {
        compressed[idx++] = data.get(i);
      }
    }
    return xsimd::load_unaligned(compressed);
  }
#endif

#if XSIMD_WITH_NEON64
  // Narrows the byteSetBits permutation to eight 16-bit lane indices and
  // compacts with one TBL instead of a store/scalar-gather/load round trip.
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::neon&) {
    const auto* lanes = byteSetBits[mask & 0xFF];
    const uint16x8_t idx = vuzp1q_u16(
        vreinterpretq_u16_s32(vld1q_s32(lanes)),
        vreinterpretq_u16_s32(vld1q_s32(lanes + 4)));
    const uint8x16_t bytes =
        vreinterpretq_u8_u16(vmlaq_n_u16(vdupq_n_u16(0x0100), idx, 0x0202));
    using U = xsimd::batch<uint8_t, A>;
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(
        U(vqtbl1q_u8(xsimd::bitwise_cast<U>(data).data, bytes)));
  }
#endif
};

template <typename T, typename A>
struct Filter<T, A, 4> {
#if XSIMD_WITH_SVE
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::sve&) {
    using U = uint32_t;
    const auto values = xsimd::bitwise_cast<xsimd::batch<U, A>>(data);
    const auto lanes = svindex_u32(0, 1);
    const auto selected = svcmpne_n_u32(
        svptrue_b32(),
        svand_n_u32_x(
            svptrue_b32(),
            svlsr_u32_x(svptrue_b32(), svdup_u32(mask), lanes),
            1),
        0);
    const xsimd::batch<U, A> packed(svcompact_u32(selected, values.data));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(packed);
  }
#endif

  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::generic&) {
    auto vindex = xsimd::batch<int32_t, A>::load_aligned(byteSetBits[mask]);
    return Permute<T, A>::apply(data, vindex, A{});
  }

  static HalfBatch<T, A> apply(HalfBatch<T, A> data, int mask, const A& arch) {
    auto vindex = HalfBatch<int32_t, A>::load_aligned(byteSetBits[mask]);
    return Permute<T, A>::apply(data, vindex, arch);
  }
};

template <typename T, typename A>
struct Filter<T, A, 8> {
#if XSIMD_WITH_SVE
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::sve&) {
    using U = uint64_t;
    const auto values = xsimd::bitwise_cast<xsimd::batch<U, A>>(data);
    const auto lanes = svindex_u64(0, 1);
    const auto selected = svcmpne_n_u64(
        svptrue_b64(),
        svand_n_u64_x(
            svptrue_b64(),
            svlsr_u64_x(svptrue_b64(), svdup_u64(mask), lanes),
            1),
        0);
    const xsimd::batch<U, A> packed(svcompact_u64(selected, values.data));
    return xsimd::bitwise_cast<xsimd::batch<T, A>>(packed);
  }
#endif

  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::generic&) {
    return genericPermute(data, byteSetBits[mask]);
  }

#if XSIMD_WITH_AVX2
  static xsimd::batch<T, A>
  apply(xsimd::batch<T, A> data, int mask, const xsimd::avx2&) {
    auto vindex =
        xsimd::batch<int32_t, A>::load_aligned(permute4x64Indices[mask]);
    return reinterpret_cast<typename xsimd::batch<T, A>::register_type>(
        _mm256_permutevar8x32_epi32(
            reinterpret_cast<__m256i>(data.data), vindex));
  }
#endif
};

template <typename A>
struct Crc32<uint64_t, A> {
  static uint32_t
  apply(uint32_t checksum, uint64_t value, const xsimd::generic&) {
    checksum ^= static_cast<uint32_t>(value);
    for (int i = 0; i < 32; ++i) {
      checksum = (checksum >> 1) ^ (0x82F63B78 & -(checksum & 1));
    }
    checksum ^= static_cast<uint32_t>(value >> 32);
    for (int i = 0; i < 32; ++i) {
      checksum = (checksum >> 1) ^ (0x82F63B78 & -(checksum & 1));
    }
    return checksum;
  }

#if XSIMD_WITH_SSE4_2
  static uint32_t
  apply(uint32_t checksum, uint64_t value, const xsimd::sse4_2&) {
    return _mm_crc32_u64(checksum, value);
  }
#endif

#if XSIMD_WITH_AVX
  static uint32_t apply(uint32_t checksum, uint64_t value, const xsimd::avx&) {
    return apply(checksum, value, xsimd::sse4_2{});
  }
#endif

#if XSIMD_WITH_SVE
  static uint32_t apply(uint32_t checksum, uint64_t value, const xsimd::sve&) {
    __asm__("crc32cx %w[c], %w[c], %x[v]"
            : [c] "+r"(checksum)
            : [v] "r"(value));
    return checksum;
  }
#endif

#if XSIMD_WITH_NEON
  static uint32_t apply(uint32_t checksum, uint64_t value, const xsimd::neon&) {
    __asm__("crc32cx %w[c], %w[c], %x[v]"
            : [c] "+r"(checksum)
            : [v] "r"(value));
    return checksum;
  }
#endif
};

} // namespace detail

template <typename T, typename A>
xsimd::batch<T, A> iota(const A&) {
  static const auto kMemo = ({
    constexpr int N = xsimd::batch<T, A>::size;
    T tmp[N];
    std::iota(tmp, tmp + N, 0);
    xsimd::load_unaligned(tmp);
  });
  return kMemo;
}

namespace detail {

#if (XSIMD_WITH_SVE && SVE_BITS == 256)
template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<std::is_base_of_v<xsimd::sve, A>>> {
  using Type = Batch128<T>;
};
#endif

#if (XSIMD_WITH_SVE && SVE_BITS == 128)
template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<std::is_base_of_v<xsimd::sve, A>>> {
  using Type = Batch64<T>;
};
#endif

template <typename T, typename A>
struct HalfBatchImpl<T, A, std::enable_if_t<std::is_base_of_v<xsimd::avx, A>>> {
  using Type = xsimd::batch<T, xsimd::sse2>;
};

template <typename T, typename A>
struct HalfBatchImpl<
    T,
    A,
    std::enable_if_t<
        std::is_base_of_v<xsimd::sse2, A> ||
        std::is_base_of_v<xsimd::neon, A>>> {
  using Type = Batch64<T>;
};

template <typename T, typename U, typename A>
struct ReinterpretBatch {
#if XSIMD_WITH_SSE2
  static xsimd::batch<T, A> apply(xsimd::batch<U, A> data, const xsimd::sse2&) {
    return xsimd::batch<T, A>(data);
  }
#endif

#if XSIMD_WITH_AVX
  static xsimd::batch<T, A> apply(xsimd::batch<U, A> data, const xsimd::avx&) {
    return xsimd::batch<T, A>(data);
  }
#endif
};

template <typename T, typename A>
struct ReinterpretBatch<T, T, A> {
  static xsimd::batch<T, A> apply(xsimd::batch<T, A> data, const A&) {
    return data;
  }
};

#if XSIMD_WITH_NEON || XSIMD_WITH_NEON64 || XSIMD_WITH_SVE

template <typename A>
struct ReinterpretBatch<uint8_t, int8_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint8_t, A> apply(
      xsimd::batch<int8_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u8_s8(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint8_t, A> apply(
      xsimd::batch<int8_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u8_s8(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint8_t, A> apply(
      xsimd::batch<int8_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u8_s8(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<int8_t, uint8_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<int8_t, A> apply(
      xsimd::batch<uint8_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_s8_u8(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<int8_t, A> apply(
      xsimd::batch<uint8_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_s8_u8(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<int8_t, A> apply(
      xsimd::batch<uint8_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_s8_u8(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint16_t, int16_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint16_t, A> apply(
      xsimd::batch<int16_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u16_s16(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint16_t, A> apply(
      xsimd::batch<int16_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u16_s16(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint16_t, A> apply(
      xsimd::batch<int16_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u16_s16(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<int16_t, uint16_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<int16_t, A> apply(
      xsimd::batch<uint16_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_s16_u16(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<int16_t, A> apply(
      xsimd::batch<uint16_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_s16_u16(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<int16_t, A> apply(
      xsimd::batch<uint16_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_s16_u16(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint32_t, int32_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u32_s32(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u32_s32(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int32_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u32_s32(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<int32_t, uint32_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<int32_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_s32_u32(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<int32_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_s32_u32(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<int32_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_s32_u32(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint64_t, uint32_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u64_u32(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u64_u32(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<uint32_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u64_u32(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint64_t, int64_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u64_s64(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u64_s64(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint64_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u64_s64(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint32_t, int64_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u32_s64(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u32_s64(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<int64_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u32_s64(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<int64_t, uint64_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_s64_u64(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_s64_u64(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<int64_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_s64_u64(data.data);
  }
#endif
};

template <typename A>
struct ReinterpretBatch<uint32_t, uint64_t, A> {
#if XSIMD_WITH_SVE
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::sve&) {
    return svreinterpret_u32_u64(data.data);
  }
#endif

#if XSIMD_WITH_NEON
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::neon&) {
    return vreinterpret_u32_u64(data.data);
  }
#endif

#if XSIMD_WITH_NEON64
  static xsimd::batch<uint32_t, A> apply(
      xsimd::batch<uint64_t, A> data,
      const xsimd::neon64&) {
    return vreinterpretq_u32_u64(data.data);
  }
#endif
};

#endif

} // namespace detail

template <typename T, typename U, typename A>
xsimd::batch<T, A> reinterpretBatch(xsimd::batch<U, A> data, const A& arch) {
  return detail::ReinterpretBatch<T, U, A>::apply(data, arch);
}

template <typename A>
inline bool memEqualUnsafe(const void* x, const void* y, int32_t size) {
  constexpr int32_t kBatch = xsimd::batch<uint8_t, A>::size;

  auto left = reinterpret_cast<const uint8_t*>(x);
  auto right = reinterpret_cast<const uint8_t*>(y);
  while (size > 0) {
    auto bits = toBitMask(
        xsimd::batch<uint8_t, A>::load_unaligned(left) ==
        xsimd::batch<uint8_t, A>::load_unaligned(right));
    if (bits == allSetBitMask<uint8_t, A>()) {
      left += kBatch;
      right += kBatch;
      size -= kBatch;
      continue;
    }
    auto leading = __builtin_ctz(~bits);
    return leading >= size;
  }
  return true;
}

namespace detail {

/// NOTE: SSE_4_2`s the performance of simdStrStr is a little slower than
/// std::find in first-char-unmatch(read only one char per match.) Use AVX2 the
/// performance will be better than std::find in that case.
#if XSIMD_WITH_AVX2
using CharVector = xsimd::batch<uint8_t, xsimd::avx2>;
#define VELOX_SIMD_STRSTR 1
#elif XSIMD_WITH_SVE
using CharVector = xsimd::batch<uint8_t, xsimd::sve>;
#define VELOX_SIMD_STRSTR 1
#elif XSIMD_WITH_NEON
using CharVector = xsimd::batch<uint8_t, xsimd::neon>;
#define VELOX_SIMD_STRSTR 1
#else
#define VELOX_SIMD_STRSTR 0
#endif

#if VELOX_SIMD_STRSTR

template <bool kCompiled, size_t kNeedleSize>
size_t FOLLY_ALWAYS_INLINE smidStrstrMemcmp(
    const char* s,
    int64_t n,
    const char* needle,
    int64_t needleSize) {
  static_assert(kNeedleSize >= 2);
  VELOX_DCHECK_GT(needleSize, 1);
  VELOX_DCHECK_GT(n, 0);
  auto first = CharVector::broadcast(needle[0]);
  auto last = CharVector::broadcast(needle[needleSize - 1]);
  int64_t i = 0;

  for (int64_t end = n - needleSize - CharVector::size; i <= end;
       i += CharVector::size) {
    auto blockFirst = CharVector::load_unaligned(s + i);
    const auto eqFirst = (first == blockFirst);
    /// std:find handle the fast-path for first-char-unmatch, so we also need
    /// to handle eqFirst.
    if (eqFirst.mask() == 0) {
      continue;
    }
    auto blockLast = CharVector::load_unaligned(s + i + needleSize - 1);
    const auto eqLast = (last == blockLast);
    auto mask = (eqFirst && eqLast).mask();
    while (mask != 0) {
      const auto bitpos = __builtin_ctz(mask);
      if constexpr (kCompiled) {
        if constexpr (kNeedleSize == 2) {
          return i + bitpos;
        }
        if (memcmp(s + i + bitpos + 1, needle + 1, kNeedleSize - 2) == 0) {
          return i + bitpos;
        }
      } else {
        if (memcmp(s + i + bitpos + 1, needle + 1, needleSize - 2) == 0) {
          return i + bitpos;
        }
      }
      mask = mask & (mask - 1);
    }
  }
  // Fallback path for generic path.
  if (i + needleSize <= n) {
    std::string_view sv(s, n);
    if constexpr (kCompiled) {
      return sv.find(needle, i, kNeedleSize);
    } else {
      return sv.find(needle, i, needleSize);
    }
  }

  return std::string::npos;
}

#endif

} // namespace detail

/// A faster implementation for std::find, about 2x faster than string_view`s
/// find() in almost cases, proved by StringSearchBenchmark.cpp. Use xsmid-batch
/// to compare first&&last char first, use fixed-memcmp to compare left chars.
/// Inline in header file will be 30% faster.
FOLLY_ALWAYS_INLINE size_t
simdStrstr(const char* s, size_t n, const char* needle, size_t k) {
#if VELOX_SIMD_STRSTR
  size_t result = std::string::npos;

  if (n < k) {
    return result;
  }

  switch (k) {
    case 0:
      return 0;

    case 1: {
      const char* res = strchr(s, needle[0]);

      return (res != nullptr) ? res - s : std::string::npos;
    }
#define VELOX_SIMD_STRSTR_CASE(size)                                   \
  case size:                                                           \
    result = detail::smidStrstrMemcmp<true, size>(s, n, needle, size); \
    break;
      VELOX_SIMD_STRSTR_CASE(2)
      VELOX_SIMD_STRSTR_CASE(3)
      VELOX_SIMD_STRSTR_CASE(4)
      VELOX_SIMD_STRSTR_CASE(5)
      VELOX_SIMD_STRSTR_CASE(6)
      VELOX_SIMD_STRSTR_CASE(7)
      VELOX_SIMD_STRSTR_CASE(8)
      VELOX_SIMD_STRSTR_CASE(9)
      VELOX_SIMD_STRSTR_CASE(10)
      VELOX_SIMD_STRSTR_CASE(11)
      VELOX_SIMD_STRSTR_CASE(12)
      VELOX_SIMD_STRSTR_CASE(13)
      VELOX_SIMD_STRSTR_CASE(14)
      VELOX_SIMD_STRSTR_CASE(15)
      VELOX_SIMD_STRSTR_CASE(16)
      VELOX_SIMD_STRSTR_CASE(17)
      VELOX_SIMD_STRSTR_CASE(18)
#if XSIMD_WITH_AVX2
      VELOX_SIMD_STRSTR_CASE(19)
      VELOX_SIMD_STRSTR_CASE(20)
      VELOX_SIMD_STRSTR_CASE(21)
      VELOX_SIMD_STRSTR_CASE(22)
      VELOX_SIMD_STRSTR_CASE(23)
      VELOX_SIMD_STRSTR_CASE(24)
      VELOX_SIMD_STRSTR_CASE(25)
      VELOX_SIMD_STRSTR_CASE(26)
      VELOX_SIMD_STRSTR_CASE(27)
      VELOX_SIMD_STRSTR_CASE(28)
      VELOX_SIMD_STRSTR_CASE(29)
      VELOX_SIMD_STRSTR_CASE(30)
      VELOX_SIMD_STRSTR_CASE(31)
      VELOX_SIMD_STRSTR_CASE(32)
      VELOX_SIMD_STRSTR_CASE(33)
      VELOX_SIMD_STRSTR_CASE(34)
#endif
    default:
      result = detail::smidStrstrMemcmp<false, 2>(s, n, needle, k);
      break;
  }
#undef VELOX_SIMD_STRSTR_CASE
  // load_unaligned is used for better performance, so result maybe bigger than
  // n-k.
  if (result <= n - k) {
    return result;
  } else {
    return std::string::npos;
  }
#endif
  return std::string_view(s, n).find(std::string_view(needle, k));
}

} // namespace facebook::velox::simd
