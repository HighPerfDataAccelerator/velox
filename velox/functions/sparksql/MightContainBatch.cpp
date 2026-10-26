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
#include <optional>

#include "velox/common/base/BloomFilter.h"
#include "velox/expression/DecodedArgs.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

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

void probeSve(
    const uint64_t* bloom,
    uint32_t bloomSize,
    const int64_t* values,
    int32_t size,
    uint8_t* out) {
#if VELOX_ARM_SVE_BLOOM_PROBE_AVAILABLE
  // Software pipeline over blocks: vector-hash block k + 1 and prefetch its
  // words before gathering and testing block k, so that two blocks of
  // independent misses overlap. 128-bit vectors alone keep too few loads in
  // flight.
  constexpr int32_t kBlock = 32;
  const uint64_t sizeMask = bloomSize - 1;
  const svuint64_t one = svdup_n_u64(1);
  std::array<uint64_t, 2 * kBlock> hashes;
  auto hashBlock = [&](int64_t start, uint64_t* target) {
    const int64_t end = std::min<int64_t>(size, start + kBlock);
    for (int64_t i = start; i < end; i += svcntd()) {
      const svbool_t active = svwhilelt_b64(i, end);
      svuint64_t key = svreinterpret_u64_s64(svld1_s64(active, values + i));
      // folly::hash::twang_mix64, lane by lane.
      key = svadd_u64_x(
          active, svnot_u64_x(active, key), svlsl_n_u64_x(active, key, 21));
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 24));
      key = svadd_u64_x(
          active,
          svadd_u64_x(active, key, svlsl_n_u64_x(active, key, 3)),
          svlsl_n_u64_x(active, key, 8));
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 14));
      key = svadd_u64_x(
          active,
          svadd_u64_x(active, key, svlsl_n_u64_x(active, key, 2)),
          svlsl_n_u64_x(active, key, 4));
      key = sveor_u64_x(active, key, svlsr_n_u64_x(active, key, 28));
      key = svadd_u64_x(active, key, svlsl_n_u64_x(active, key, 31));
      svst1_u64(active, target + (i - start), key);
      const svuint64_t index =
          svand_n_u64_x(active, svlsr_n_u64_x(active, key, 24), sizeMask);
      svprfd_gather_u64index(active, bloom, index, SV_PLDL1KEEP);
    }
  };
  if (size > 0) {
    hashBlock(0, hashes.data());
  }
  for (int64_t start = 0, slot = 0; start < size; start += kBlock, slot ^= 1) {
    if (start + kBlock < size) {
      hashBlock(start + kBlock, hashes.data() + (slot ^ 1) * kBlock);
    }
    const uint64_t* current = hashes.data() + slot * kBlock;
    const int64_t end = std::min<int64_t>(size, start + kBlock);
    for (int64_t i = start; i < end; i += svcntd()) {
      const svbool_t active = svwhilelt_b64(i, end);
      const svuint64_t key = svld1_u64(active, current + (i - start));
      const svuint64_t index =
          svand_n_u64_x(active, svlsr_n_u64_x(active, key, 24), sizeMask);
      auto bit = [&](uint64_t shift) {
        return svlsl_u64_x(
            active,
            one,
            svand_n_u64_x(active, svlsr_n_u64_x(active, key, shift), 63));
      };
      const svuint64_t bits = svorr_u64_x(
          active,
          svorr_u64_x(active, bit(0), bit(6)),
          svorr_u64_x(active, bit(12), bit(18)));
      const svuint64_t words = svld1_gather_u64index_u64(active, bloom, index);
      const svbool_t hit =
          svcmpeq_u64(active, svand_u64_x(active, words, bits), bits);
      svst1b_u64(active, out + i, svdup_n_u64_z(hit, 1));
    }
  }
#else
  probeGeneric(bloom, bloomSize, values, size, out);
#endif
}

} // namespace bloom_probe

namespace {

class MightContainBatchFunction : public exec::VectorFunction {
 public:
  MightContainBatchFunction(std::optional<BloomFilterView> view, bool sve)
      : view_(view), sve_(sve) {}

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
    auto probe = [&](const int64_t* input, int32_t size, uint8_t* out) {
      if (sve_) {
        bloom_probe::probeSve(bloom, bloomSize, input, size, out);
      } else {
        bloom_probe::probeGeneric(bloom, bloomSize, input, size, out);
      }
    };
    constexpr int32_t kBatch = 1'024;
    std::array<uint8_t, kBatch> hits;
    if (values->isIdentityMapping() && rows.isAllSelected()) {
      const auto* raw = values->data<int64_t>();
      for (vector_size_t start = rows.begin(); start < rows.end();
           start += kBatch) {
        const int32_t size = std::min<int32_t>(kBatch, rows.end() - start);
        probe(raw + start, size, hits.data());
        for (int32_t i = 0; i < size; ++i) {
          bits::setBit(rawResult, start + i, hits[i] != 0);
        }
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
  // Empty when the filter argument is not constant, matching the simple
  // function, which then returns false for every non-NULL row.
  const std::optional<BloomFilterView> view_;

  const bool sve_;
};

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
  // A NULL constant filter makes every row NULL through default NULL handling.
  if (constant != nullptr && !constant->isNullAt(0)) {
    const auto serialized =
        constant->as<ConstantVector<StringView>>()->valueAt(0);
    VELOX_USER_CHECK_GE(
        serialized.size(),
        BloomFilterView::kSerializedHeaderSize,
        "Serialized BloomFilter is too small: {}",
        serialized.size());
    view.emplace(serialized.data());
  }
  return std::make_shared<MightContainBatchFunction>(view, sve);
}

} // namespace facebook::velox::functions::sparksql
