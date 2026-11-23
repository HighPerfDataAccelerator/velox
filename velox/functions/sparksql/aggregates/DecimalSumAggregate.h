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
#pragma once

#if defined(__linux__) && defined(__aarch64__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/type/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {

#if defined(__linux__) && defined(__aarch64__)
inline bool supportsVeraDecimalAdd() {
  static const bool enabled = [] {
    if (!(getauxval(AT_HWCAP) & HWCAP_CPUID)) {
      return false;
    }
    uint64_t midr;
    asm("mrs %0, midr_el1" : "=r"(midr));
    return (midr & 0xff00fff0) == 0x4e000100;
  }();
  return enabled;
}
#endif

/// @tparam TInputType The raw input data type.
/// @tparam TSumType The type of sum in the output of partial aggregation or the
/// final output type of final aggregation.
/// @tparam ResultPrecision The precision of the result type, used for checking
/// overflow.
template <typename TInputType, typename TSumType, uint8_t ResultPrecision>
class DecimalSumAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType = Row</*sum*/ TSumType,
                               /*isEmpty*/ bool>;

  using OutputType = TSumType;

  /// Spark's decimal sum doesn't have the concept of a null group, each group
  /// is initialized with an initial value, where sum = 0 and isEmpty = true.
  /// The final agg may fallback to being executed in Spark, so the meaning of
  /// the intermediate data should be consistent with Spark. Therefore, we need
  /// to use the parameter nonNullGroup in writeIntermediateResult to output a
  /// null group as sum = 0, isEmpty = true. nonNullGroup is only available when
  /// default-null behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  static bool toIntermediate(
      exec::out_type<Row<TSumType, bool>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(static_cast<TSumType>(in.value()), false));
    } else {
      out.copy_from(std::make_tuple(static_cast<TSumType>(0), true));
    }
    return true;
  }

  /// This struct stores the sum of input values, overflow during accumulation,
  /// and a bool value isEmpty used to indicate whether all inputs are null. The
  /// initial value of sum is 0. We need to keep sum unchanged if the input is
  /// null, as sum function ignores null input. If the isEmpty is true, then it
  /// means there were no values to begin with or all the values were null, so
  /// the result will be null. A missing sum with isEmpty false represents
  /// overflow propagated from an intermediate result.
  struct AccumulatorType {
    // Store validity next to the other flags to avoid optional padding.
    // The intermediate row representation is unchanged.
    int128_t sum{0};
    int64_t overflow{0};
    bool hasSum{true};
    bool isEmpty{true};

    static constexpr bool is_aligned_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        DecimalSumAggregate<TInputType, TSumType, ResultPrecision>* /*fn*/) {}

    std::optional<int128_t> computeFinalResult() const {
      if (!hasSum) {
        return std::nullopt;
      }
      auto const adjustedSum = DecimalUtil::adjustSumForOverflow(sum, overflow);
      if (adjustedSum.has_value() &&
          DecimalUtil::valueInPrecisionRange(adjustedSum, ResultPrecision)) {
        return adjustedSum;
      } else {
        // Found overflow during computing adjusted sum.
        return std::nullopt;
      }
    }

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data) {
      if (!data.has_value()) {
        return false;
      }
      return addValue(data.value());
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<TSumType, bool>> other) {
      if (!other.has_value()) {
        return false;
      }
      auto const otherSum = other.value().template at<0>();
      auto const otherIsEmpty = other.value().template at<1>();

      // isEmpty is never null.
      VELOX_CHECK(otherIsEmpty.has_value());
      return combineValue(
          otherSum.has_value() ? otherSum.value() : TSumType{0},
          !otherSum.has_value(),
          otherIsEmpty.value());
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out) {
      if (!nonNullGroup) {
        // If a group is null, all values in this group are null. In Spark, this
        // group will be the initial value, where sum is 0 and isEmpty is true.
        out = std::make_tuple(static_cast<TSumType>(0), true);
      } else {
        auto finalResult = computeFinalResult();
        if (finalResult.has_value()) {
          out = std::make_tuple(
              static_cast<TSumType>(finalResult.value()), isEmpty);
        } else {
          // Sum should be set to null on overflow,
          // and isEmpty should be set to false.
          out.template set_null_at<0>();
          out.template get_writer_at<1>() = false;
        }
      }
      return true;
    }

    /// Same as addInput() for a non-null value. Returns true.
    FOLLY_ALWAYS_INLINE bool addValue(TInputType value) {
      if (!hasSum) {
        // A missing sum represents overflow from a non-empty input.
        VELOX_CHECK(!isEmpty);
        return true;
      }
      int128_t result;
      overflow += DecimalUtil::addWithOverflow(result, value, sum);
      sum = result;
      isEmpty = false;
      return true;
    }

#if defined(__linux__) && defined(__aarch64__)
    // ADDS/ADCS checks the signed 128-bit addition on Vera. Velox also
    // canonicalizes the exact negative boundary with its 127-bit carry scheme.
    FOLLY_ALWAYS_INLINE void addValueWithCarry(TInputType value) {
#if defined(__AARCH64EL__)
      // On little-endian Vera, a narrow addend often changes only the low
      // half. Fall back before writing if the high half or the canonical
      // negative boundary requires handling by the full accumulator.
      uint64_t low;
      std::memcpy(&low, &sum, sizeof(low));
      const auto delta = static_cast<uint64_t>(value);
      uint64_t result;
      const bool carry = __builtin_add_overflow(low, delta, &result);
      const bool negative = value < 0;
      bool narrow = true;
      if constexpr (sizeof(TInputType) > sizeof(uint64_t)) {
        narrow = static_cast<__int128>(static_cast<int64_t>(delta)) == value;
      }
      const bool valid = narrow && (carry == negative);
      if (LIKELY(hasSum && valid && result != 0)) {
        std::memcpy(&sum, &result, sizeof(result));
        isEmpty = false;
        return;
      }
#endif
      int128_t next;
      if (LIKELY(
              hasSum &&
              !__builtin_add_overflow(
                  sum, static_cast<int128_t>(value), &next) &&
              next != std::numeric_limits<int128_t>::min())) {
        sum = next;
        isEmpty = false;
      } else {
        addValue(value);
      }
    }
#endif

    /// Same as combine() for a non-null intermediate row whose sum is
    /// 'otherSum', or null when 'otherSumIsNull' is true.
    FOLLY_ALWAYS_INLINE bool combineValue(
        TSumType otherSum,
        bool otherSumIsNull,
        bool otherIsEmpty) {
      if (isEmpty && otherIsEmpty) {
        // Both accumulators are empty, no need to do the combination.
        return false;
      }
      const bool currentOverflow = !isEmpty && !hasSum;
      const bool otherOverflow = !otherIsEmpty && otherSumIsNull;
      if (currentOverflow || otherOverflow) {
        hasSum = false;
        isEmpty = false;
      } else {
        VELOX_CHECK(!otherSumIsNull);
        int128_t result;
        overflow += DecimalUtil::addWithOverflow(result, otherSum, sum);
        sum = result;
        isEmpty &= otherIsEmpty;
      }
      return true;
    }

    bool writeFinalResult(bool nonNullGroup, exec::out_type<OutputType>& out) {
      if (!nonNullGroup || isEmpty) {
        // If isEmpty is true, we should set null.
        return false;
      }
      auto finalResult = computeFinalResult();
      if (finalResult.has_value()) {
        out = static_cast<TSumType>(finalResult.value());
        return true;
      } else {
        // Sum should be set to null on overflow.
        return false;
      }
    }
  };
};

/// Runs DecimalSumAggregate through SimpleAggregateAdapter, adding typed loops
/// for grouped raw and intermediate input. Inputs those loops do not cover go
/// through the adapter.
template <typename TInputType, typename TSumType, uint8_t ResultPrecision>
class DecimalSumAggregateAdapter
    : public exec::SimpleAggregateAdapter<
          DecimalSumAggregate<TInputType, TSumType, ResultPrecision>> {
  using Base = exec::SimpleAggregateAdapter<
      DecimalSumAggregate<TInputType, TSumType, ResultPrecision>>;
  using Accumulator = typename DecimalSumAggregate<
      TInputType,
      TSumType,
      ResultPrecision>::AccumulatorType;

 public:
  using Base::Base;

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    decodedRaw_.decode(*args[0], rows);
#if defined(__linux__) && defined(__aarch64__)
    if (supportsVeraDecimalAdd()) {
#if defined(__AARCH64EL__)
      const auto count = rows.end() - rows.begin();
      if (count >= 64 && rows.isAllSelected() &&
          decodedRaw_.isIdentityMapping()) {
        // Sample adjacent group pointers before scanning a batch for runs.
        const auto step = count / 8;
        int equal = 0;
        for (int i = 0; i < 8; ++i) {
          const auto row = rows.begin() + i * step;
          equal += groups[row] == groups[row + 1];
        }
        if (equal >= 6) {
          addRuns(groups, rows, decodedRaw_.data<TInputType>());
          return;
        }
      }
#endif
      auto add = [&](char* group, TInputType value) {
        Base::template value<Accumulator>(group)->addValueWithCarry(value);
        Base::clearNull(group);
      };
      if (decodedRaw_.isIdentityMapping() && !decodedRaw_.mayHaveNulls()) {
        const auto* values = decodedRaw_.data<TInputType>();
        rows.applyToSelected(
            [&](vector_size_t row) { add(groups[row], values[row]); });
      } else {
        rows.applyToSelected([&](vector_size_t row) {
          if (!decodedRaw_.isNullAt(row)) {
            add(groups[row], decodedRaw_.valueAt<TInputType>(row));
          }
        });
      }
      return;
    }
#endif
    if (decodedRaw_.isIdentityMapping() && !decodedRaw_.mayHaveNulls()) {
      const auto* values = decodedRaw_.data<TInputType>();
      rows.applyToSelected(
          [&](vector_size_t row) { addValue(groups[row], values[row]); });
      return;
    }
    rows.applyToSelected([&](vector_size_t row) {
      if (!decodedRaw_.isNullAt(row)) {
        addValue(groups[row], decodedRaw_.valueAt<TInputType>(row));
      }
    });
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool mayPushdown) override {
    const auto* input = args[0]->wrappedVector() == args[0].get()
        ? args[0]->as<RowVector>()
        : nullptr;
    const auto* sums = input != nullptr
        ? input->childAt(0)->asFlatVector<TSumType>()
        : nullptr;
    const auto* isEmpty =
        input != nullptr ? input->childAt(1)->asFlatVector<bool>() : nullptr;
    if (sums == nullptr || isEmpty == nullptr || isEmpty->mayHaveNulls()) {
      Base::addIntermediateResults(groups, rows, args, mayPushdown);
      return;
    }
    const uint64_t* rowNulls = input->rawNulls();
    const uint64_t* sumNulls = sums->rawNulls();
    const TSumType* sumValues = sums->rawValues();
    const auto* isEmptyBits = isEmpty->template rawValues<uint64_t>();
    rows.applyToSelected([&](vector_size_t row) {
      if (rowNulls != nullptr && bits::isBitNull(rowNulls, row)) {
        return;
      }
      auto* accumulator =
          Base::template value<Accumulator>(groups[row]);
      if (accumulator->combineValue(
              sumValues[row],
              sumNulls != nullptr && bits::isBitNull(sumNulls, row),
              bits::isBitSet(isEmptyBits, row))) {
        Base::clearNull(groups[row]);
      }
    });
  }

 private:
#if defined(__linux__) && defined(__aarch64__) && defined(__AARCH64EL__)
  // Read four fully selected, non-null values from the same group. If each
  // fits in 61 signed bits, the horizontal reduction fits in signed 64 bits.
  static FOLLY_ALWAYS_INLINE bool addFourBounded(
      Accumulator& accumulator,
      const TInputType* values) {
    // A high word of zero or minus one leaves room for all four
    // intermediate sums, including cancellation within the group.
    const auto high = static_cast<uint64_t>(
        static_cast<unsigned __int128>(accumulator.sum) >> 64);
    if (!accumulator.hasSum || (high != 0 && high != UINT64_MAX)) {
      return false;
    }
    int64x2_t low01, low23;
    uint64x2_t outOfRange = vdupq_n_u64(0);
    if constexpr (sizeof(TInputType) == sizeof(int128_t)) {
      // Load through bytes so the representation never aliases an int64_t
      // array.
      auto a = vreinterpretq_s64_u8(
          vld1q_u8(reinterpret_cast<const uint8_t*>(values)));
      auto b = vreinterpretq_s64_u8(
          vld1q_u8(reinterpret_cast<const uint8_t*>(values + 1)));
      auto c = vreinterpretq_s64_u8(
          vld1q_u8(reinterpret_cast<const uint8_t*>(values + 2)));
      auto d = vreinterpretq_s64_u8(
          vld1q_u8(reinterpret_cast<const uint8_t*>(values + 3)));
      low01 = vuzp1q_s64(a, b);
      low23 = vuzp1q_s64(c, d);
      auto high01 = vuzp2q_s64(a, b), high23 = vuzp2q_s64(c, d);
      outOfRange = vorrq_u64(
          vreinterpretq_u64_s64(veorq_s64(high01, vshrq_n_s64(low01, 63))),
          vreinterpretq_u64_s64(veorq_s64(high23, vshrq_n_s64(low23, 63))));
    } else {
      low01 = vld1q_s64(values);
      low23 = vld1q_s64(values + 2);
    }
    auto excess01 = veorq_s64(vshrq_n_s64(low01, 60), vshrq_n_s64(low01, 63));
    auto excess23 = veorq_s64(vshrq_n_s64(low23, 60), vshrq_n_s64(low23, 63));
    outOfRange = vorrq_u64(
        outOfRange, vreinterpretq_u64_s64(vorrq_s64(excess01, excess23)));
    if (vmaxvq_u32(vreinterpretq_u32_u64(outOfRange))) {
      return false;
    }
    accumulator.sum += vaddvq_s64(vaddq_s64(low01, low23));
    accumulator.isEmpty = false;
    return true;
  }

  FOLLY_NOINLINE void addRuns(
      char** groups,
      const SelectivityVector& rows,
      const TInputType* values) {
    const auto end = rows.end();
    const auto* valid = decodedRaw_.nulls(&rows);
    for (vector_size_t first = rows.begin(); first < end;) {
      if (valid && !bits::isBitSet(valid, first)) {
        ++first;
        continue;
      }
      auto* group = groups[first];
      const auto key = vdupq_n_u64(reinterpret_cast<uintptr_t>(group));
      vector_size_t limit = first + 1;
      while (end - limit >= 2) {
        const auto pair = vreinterpretq_u64_u8(
            vld1q_u8(reinterpret_cast<const uint8_t*>(groups + limit)));
        if (vminvq_u32(vreinterpretq_u32_u64(vceqq_u64(pair, key))) !=
            UINT32_MAX) {
          break;
        }
        limit += 2;
      }
      if (limit < end && groups[limit] == group) {
        ++limit;
      }
      auto* accumulator = Base::template value<Accumulator>(group);
      Accumulator current = *accumulator;
      bool addedValue = false;
      for (auto row = first; row < limit;) {
        // Stay in one bitmap word. The scalar path consumes a boundary or null.
        const bool fourValid = limit - row >= 4 &&
            (!valid ||
             ((row & 63) <= 60 &&
              ((valid[row >> 6] >> (row & 63)) & 15) == 15));
        if (fourValid && addFourBounded(current, values + row)) {
          row += 4;
          addedValue = true;
        } else {
          if (!valid || bits::isBitSet(valid, row)) {
            current.addValueWithCarry(values[row]);
            addedValue = true;
          }
          ++row;
        }
      }
      if (addedValue) {
        *accumulator = current;
        Base::clearNull(group);
      }
      first = limit;
    }
  }
#endif

  FOLLY_ALWAYS_INLINE void addValue(char* group, TInputType value) {
    Base::template value<Accumulator>(group)->addValue(value);
    Base::clearNull(group);
  }

  DecodedVector decodedRaw_;
};

} // namespace facebook::velox::functions::aggregate::sparksql
