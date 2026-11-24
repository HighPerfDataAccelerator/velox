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

#include "velox/functions/sparksql/MightContainBatch.h"

#include <array>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/container/F14Map.h>
#include <gflags/gflags.h>

#include "velox/common/base/BloomFilter.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

DEFINE_bool(
    velox_might_contain_memo,
    false,
    "Remember recent results of a batched might_contain by input value, so "
    "that low-cardinality inputs skip the Bloom filter lookups");

DEFINE_bool(
    velox_might_contain_share_filters,
    false,
    "Let batched might_contain functions of all tasks in the process probe "
    "one shared copy of an identical serialized Bloom filter");

namespace facebook::velox::functions::sparksql {
namespace bloom_probe {

void probeGeneric(
    const uint64_t* bloom,
    uint32_t bloomSize,
    const int64_t* values,
    int32_t size,
    uint8_t* out) {
  // Hash a block and prefetch its words before testing, so that up to a
  // block of independent cache misses are in flight.
  constexpr int32_t kBlock = 32;
  std::array<uint64_t, kBlock> hashes;
  for (int32_t start = 0; start < size; start += kBlock) {
    const int32_t end = std::min(size, start + kBlock);
    for (int32_t i = start; i < end; ++i) {
      const uint64_t hash = folly::hasher<int64_t>()(values[i]);
      hashes[i - start] = hash;
      __builtin_prefetch(bloom + ((hash >> 24) & (bloomSize - 1)));
    }
    for (int32_t i = start; i < end; ++i) {
      const uint64_t hash = hashes[i - start];
      const uint64_t bits = mask(hash);
      out[i] = (bloom[(hash >> 24) & (bloomSize - 1)] & bits) == bits;
    }
  }
}

#if VELOX_ARM_SVE_BLOOM_PROBE_AVAILABLE
template <bool packed>
static void probeSveImpl(
    const uint64_t* bloom,
    uint32_t bloomSize,
    const int64_t* values,
    int32_t size,
    uint8_t* out) {
  // Software pipeline over blocks: vector-hash block k + 1 and prefetch its
  // words before gathering and testing block k, so that two blocks of
  // independent misses overlap. 128-bit vectors alone keep too few loads in
  // flight.
  constexpr int32_t kBlock = 32;
  const uint64_t sizeMask = bloomSize - 1;
  const int64_t lanes = svcntd();
  std::array<uint64_t, 2 * kBlock> hashes;
  const auto indices = [&](svbool_t active, svuint64_t key) {
    return svand_n_u64_x(active, svlsr_n_u64_x(active, key, 24), sizeMask);
  };
  const auto hashBlock = [&](int64_t start, uint64_t* target) {
    const int64_t end = std::min<int64_t>(size, start + kBlock);
    for (int64_t i = start; i < end; i += lanes) {
      const svbool_t active = svwhilelt_b64(i, end);
      svuint64_t key = svreinterpret_u64_s64(svld1_s64(active, values + i));
      // folly::hash::twang_mix64, lane by lane.
      key = svadd_u64_x(
          active, svnot_u64_x(active, key), svlsl_n_u64_x(active, key, 21));
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 24));
      key = svmul_n_u64_x(active, key, 265);
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 14));
      key = svmul_n_u64_x(active, key, 21);
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 28));
      key = svadd_u64_x(active, key, svlsl_n_u64_x(active, key, 31));
      svst1_u64(active, target + (i - start), key);
      svprfd_gather_u64index(active, bloom, indices(active, key), SV_PLDL1KEEP);
    }
  };
  const auto gather = [&](svbool_t active, svuint64_t key) {
    return svld1_gather_u64index_u64(active, bloom, indices(active, key));
  };
  const auto hitBits = [](svbool_t active, svuint64_t key, svuint64_t words) {
    // Align each selected bit with bit zero and test them together.
    const auto bit = [&](uint64_t shift) {
      const auto position =
          svand_n_u64_x(active, svlsr_n_u64_x(active, key, shift), 63);
      return svlsr_u64_x(active, words, position);
    };
    const auto hits = svand_u64_x(
        active,
        svand_u64_x(active, bit(0), bit(6)),
        svand_u64_x(active, bit(12), bit(18)));
    return svand_n_u64_x(active, hits, 1);
  };

  if (size > 0) {
    hashBlock(0, hashes.data());
  }
  for (int64_t start = 0, slot = 0; start < size; start += kBlock, slot ^= 1) {
    if (start + kBlock < size) {
      hashBlock(start + kBlock, hashes.data() + (slot ^ 1) * kBlock);
    }
    const uint64_t* current = hashes.data() + slot * kBlock;
    const int64_t length = std::min<int64_t>(size - start, kBlock);
    if constexpr (packed) {
      const auto two = [&](const uint64_t* h) {
        const auto active = svptrue_b64();
        const auto key = svld1_u64(active, h);
        return svcmpne_n_u64(
            active, hitBits(active, key, gather(active, key)), 0);
      };
      // Compact eight two-lane predicates into 16 consecutive result bits.
      const auto four = [&](const uint64_t* h) {
        return svuzp1_b32(two(h), two(h + 2));
      };
      const auto eight = [&](const uint64_t* h) {
        return svuzp1_b16(four(h), four(h + 4));
      };
      int32_t i = 0;
      for (; i + 16 <= length; i += 16) {
        const auto mask = svuzp1_b8(eight(current + i), eight(current + i + 8));
        auto* dst =
            reinterpret_cast<unsigned char (*)[2]>(out + (start + i) / 8);
        asm("str %1, %0" : "=Q"(*dst) : "Upa"(mask));
      }
      for (; i < length; ++i) {
        const auto key = current[i];
        const auto required = mask(key);
        uint64_t word;
        std::memcpy(
            &word,
            reinterpret_cast<const uint8_t*>(bloom) +
                8 * ((key >> 24) & sizeMask),
            8);
        bits::setBit(out, start + i, (word & required) == required);
      }
    } else {
      auto* output = out + start;
      const auto storeHits = [&](svbool_t active,
                                 svuint64_t key,
                                 svuint64_t words,
                                 uint8_t* output) {
        svst1b_u64(active, output, hitBits(active, key, words));
      };
      for (int64_t i = 0; i < length; i += lanes) {
        const auto active = svwhilelt_b64(i, length);
        const auto key = svld1_u64(active, current + i);
        storeHits(active, key, gather(active, key), output + i);
      }
    }
  }
}
#endif

void probeSve(
    const uint64_t* bloom,
    uint32_t bloomSize,
    const int64_t* values,
    int32_t size,
    uint8_t* out) {
#if VELOX_ARM_SVE_BLOOM_PROBE_AVAILABLE
  probeSveImpl<false>(bloom, bloomSize, values, size, out);
#else
  probeGeneric(bloom, bloomSize, values, size, out);
#endif
}

} // namespace bloom_probe

namespace {

void writeDenseHits(
    uint64_t* result,
    int32_t start,
    const uint8_t* hits,
    int32_t size,
    [[maybe_unused]] bool useSve) {
  int32_t i = 0;
#if VELOX_ARM_SVE_BLOOM_PROBE_AVAILABLE
  if (useSve && size >= 16 && svcntb() == 16) {
    for (; i < size && ((start + i) & 7) != 0; ++i) {
      bits::setBit(result, start + i, hits[i] != 0);
    }
    auto* bytes = reinterpret_cast<uint8_t*>(result);
    for (; size - i >= 16; i += 16) {
      const auto lanes = svptrue_b8();
      const auto mask = svcmpne_n_u8(lanes, svld1_u8(lanes, hits + i), 0);
      // At VL=128 the predicate stores one bit per input byte in two bytes.
      auto* dst =
          reinterpret_cast<unsigned char (*)[2]>(bytes + ((start + i) >> 3));
      asm("str %1, %0" : "=Q"(*dst) : "Upa"(mask));
    }
  }
#endif
  for (; i < size; ++i) {
    bits::setBit(result, start + i, hits[i] != 0);
  }
}

class MightContainBatchFunction : public exec::VectorFunction {
 public:
  MightContainBatchFunction(
      VectorPtr filter,
      std::optional<BloomFilterView> view,
      std::shared_ptr<const std::string> sharedFilter,
      bool sve)
      : filter_(std::move(filter)),
        view_(view),
        sharedFilter_(std::move(sharedFilter)),
        sve_(sve),
        memoEnabled_(FLAGS_velox_might_contain_memo) {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    context.ensureWritable(rows, BOOLEAN(), result);
    auto* flat = result->asUnchecked<FlatVector<bool>>();
    auto* rawResult = flat->mutableRawValues<uint64_t>();
    if (!view_.has_value()) {
      rows.applyToSelected(
          [&](vector_size_t row) { bits::setBit(rawResult, row, false); });
      return;
    }
    exec::LocalDecodedVector decodedValues(context, *args[1], rows);
    const auto* values = decodedValues.get();
    const uint64_t* bloom = view_->bitsData();
    const auto bloomSize = static_cast<uint32_t>(view_->size());
    auto probeBloom = [&](const int64_t* input, int32_t size, uint8_t* out) {
      if (sve_) {
        bloom_probe::probeSve(bloom, bloomSize, input, size, out);
      } else {
        bloom_probe::probeGeneric(bloom, bloomSize, input, size, out);
      }
    };
    auto probe = [&](const int64_t* input, int32_t size, uint8_t* out) {
      if (memoEnabled_) {
        probeWithMemo(input, size, out, probeBloom);
      } else {
        probeBloom(input, size, out);
      }
    };
    constexpr int32_t kBatch = kMaxProbeSize;
    std::array<uint8_t, kBatch> hits;
    if (values->isIdentityMapping() && rows.isAllSelected()) {
      const auto* raw = values->data<int64_t>();
      for (vector_size_t start = rows.begin(); start < rows.end();
           start += kBatch) {
        const int32_t size = std::min<int32_t>(kBatch, rows.end() - start);
#if VELOX_ARM_SVE_BLOOM_PROBE_AVAILABLE
        if (sve_ && !memoEnabled_ && size >= 16 && (start & 7) == 0 &&
            svcntb() == 16) {
          bloom_probe::probeSveImpl<true>(
              bloom,
              bloomSize,
              raw + start,
              size,
              reinterpret_cast<uint8_t*>(rawResult) + start / 8);
          continue;
        }
#endif
        probe(raw + start, size, hits.data());
        writeDenseHits(rawResult, start, hits.data(), size, sve_);
      }
      return;
    }
    std::array<int64_t, kBatch> input;
    std::array<vector_size_t, kBatch> positions;
    int32_t size = 0;
    auto flush = [&]() {
      probe(input.data(), size, hits.data());
      for (int32_t i = 0; i < size; ++i) {
        bits::setBit(rawResult, positions[i], hits[i] != 0);
      }
      size = 0;
    };
    rows.applyToSelected([&](vector_size_t row) {
      input[size] = values->valueAt<int64_t>(row);
      positions[size] = row;
      if (++size == kBatch) {
        flush();
      }
    });
    if (size > 0) {
      flush();
    }
  }

 private:
  // Answers repeated input values from a small direct-mapped table of earlier
  // results and probes the Bloom filter only for the rest. Runtime filters on
  // low-cardinality keys (dates, regions) then stay in L1/L2 instead of
  // missing on a large, sparsely populated filter. Results are unchanged. The
  // table is dropped once the first kMemoDecisionRows rows show a low hit rate.
  template <typename ProbeBloom>
  void probeWithMemo(
      const int64_t* input,
      int32_t size,
      uint8_t* out,
      ProbeBloom& probeBloom) const {
    if (memoKeys_.empty()) {
      memoKeys_.resize(kMemoSize);
      memoStates_.assign(kMemoSize, kMemoEmpty);
    }
    std::array<int64_t, kMaxProbeSize> missValues;
    std::array<int32_t, kMaxProbeSize> missPositions;
    std::array<uint8_t, kMaxProbeSize> missHits;
    int32_t numMisses = 0;
    for (int32_t i = 0; i < size; ++i) {
      // Inputs are xxhash64 values, so their low bits index the table well.
      const auto slot = static_cast<uint64_t>(input[i]) & (kMemoSize - 1);
      if (memoStates_[slot] != kMemoEmpty && memoKeys_[slot] == input[i]) {
        out[i] = memoStates_[slot];
      } else {
        missValues[numMisses] = input[i];
        missPositions[numMisses++] = i;
      }
    }
    if (numMisses > 0) {
      probeBloom(missValues.data(), numMisses, missHits.data());
      for (int32_t i = 0; i < numMisses; ++i) {
        out[missPositions[i]] = missHits[i];
        const auto slot =
            static_cast<uint64_t>(missValues[i]) & (kMemoSize - 1);
        memoKeys_[slot] = missValues[i];
        memoStates_[slot] = missHits[i];
      }
    }
    memoNumProbed_ += size;
    memoNumHits_ += size - numMisses;
    if (memoNumProbed_ >= kMemoDecisionRows &&
        memoNumHits_ * 2 < memoNumProbed_) {
      memoEnabled_ = false;
      memoKeys_.clear();
      memoKeys_.shrink_to_fit();
      memoStates_.clear();
      memoStates_.shrink_to_fit();
    }
  }

  static constexpr int32_t kMaxProbeSize = 1'024;
  static constexpr uint64_t kMemoSize = 4'096;
  static constexpr uint8_t kMemoEmpty = 2;
  static constexpr int64_t kMemoDecisionRows = 1 << 16;

  // Own the constant vector when the view reads its serialized words.
  const VectorPtr filter_;

  // Empty when the filter argument is not constant, matching the simple
  // function, which then returns false for every non-NULL row.
  const std::optional<BloomFilterView> view_;

  // The shared copy 'view_' reads when filters are shared, else null.
  const std::shared_ptr<const std::string> sharedFilter_;

  const bool sve_;

  // Memo state. A function instance belongs to one expression set, which a
  // single driver thread evaluates.
  mutable bool memoEnabled_;
  mutable std::vector<int64_t> memoKeys_;
  // Per slot: the remembered result (0 or 1), or kMemoEmpty.
  mutable std::vector<uint8_t> memoStates_;
  mutable int64_t memoNumProbed_{0};
  mutable int64_t memoNumHits_{0};
};

// Returns the process-wide copy of 'serialized' that every function probing an
// identical filter shares. Each task otherwise holds its own copy, and the
// copies of tasks on the two hardware threads of a core compete for its L2.
std::shared_ptr<const std::string> sharedFilter(StringView serialized) {
  static std::mutex mutex;
  static folly::F14FastMap<size_t, std::vector<std::weak_ptr<const std::string>>>
      filters;
  const std::string_view bytes(serialized.data(), serialized.size());
  const auto hash = std::hash<std::string_view>{}(bytes);
  std::lock_guard<std::mutex> lock(mutex);
  auto& candidates = filters[hash];
  for (auto it = candidates.begin(); it != candidates.end();) {
    auto candidate = it->lock();
    if (candidate == nullptr) {
      it = candidates.erase(it);
      continue;
    }
    if (*candidate == bytes) {
      return candidate;
    }
    ++it;
  }
  auto copy = std::make_shared<const std::string>(bytes);
  candidates.push_back(copy);
  return copy;
}

} // namespace

std::vector<std::shared_ptr<exec::FunctionSignature>> mightContainSignatures() {
  return {exec::FunctionSignatureBuilder()
              .returnType("boolean")
              .argumentType("varbinary")
              .argumentType("bigint")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeMightContainBatch(
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    bool sve) {
  VELOX_CHECK_EQ(inputArgs.size(), 2);
  const auto& constant = inputArgs[0].constantValue;
  std::optional<BloomFilterView> view;
  std::shared_ptr<const std::string> shared;
  // A NULL constant filter makes every row NULL through default NULL handling.
  if (constant != nullptr && !constant->isNullAt(0)) {
    const auto serialized =
        constant->as<ConstantVector<StringView>>()->valueAt(0);
    VELOX_USER_CHECK_GE(
        serialized.size(),
        BloomFilterView::kSerializedHeaderSize,
        "Serialized BloomFilter is too small: {}",
        serialized.size());
    if (FLAGS_velox_might_contain_share_filters) {
      shared = sharedFilter(serialized);
      view.emplace(shared->data());
    } else {
      view.emplace(serialized.data());
    }
    const auto availableWords =
        (serialized.size() - BloomFilterView::kSerializedHeaderSize) /
        sizeof(uint64_t);
    VELOX_USER_CHECK_LE(
        static_cast<size_t>(view->size()),
        availableWords,
        "Truncated serialized BloomFilter");
  }
  VectorPtr filter = shared ? nullptr : constant;
  return std::make_shared<MightContainBatchFunction>(
      std::move(filter), view, std::move(shared), sve);
}

} // namespace facebook::velox::functions::sparksql
