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
#include "velox/experimental/ucx-exchange/RangePartitionFunction.h"
#include "velox/vector/FlatVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>

#include <folly/json.h>

#include <sstream>

namespace facebook::velox::ucx_exchange {
namespace {

class UcxOnlyRangePartitionFunction final : public core::PartitionFunction {
 public:
  std::optional<uint32_t> partition(const RowVector&, std::vector<uint32_t>&)
      override {
    VELOX_FAIL(
        "RANGE_PID is supported only by UcxPartitionedOutput; refusing to "
        "silently substitute hash or round-robin partitioning");
  }
};

} // namespace

RowVectorPtr buildRangeBoundaryVector(
    const std::string& boundsJson,
    const RowTypePtr& inputType,
    const std::vector<column_index_t>& keyChannels,
    memory::MemoryPool* pool,
    std::vector<cudf::order>& orders,
    std::vector<cudf::null_order>& nullOrders,
    bool* splitEqualKeys) {
  const auto descriptor = folly::parseJson(boundsJson);
  VELOX_CHECK_EQ(
      descriptor["version"].asInt(),
      1,
      "Unsupported RANGE_PID descriptor version");
  const auto& keys = descriptor["keys"];
  const auto& bounds = descriptor["bounds"];
  VELOX_CHECK(keys.isArray(), "RANGE_PID keys must be an array");
  VELOX_CHECK(bounds.isArray(), "RANGE_PID bounds must be an array");
  VELOX_CHECK_EQ(
      keys.size(),
      keyChannels.size(),
      "RANGE_PID key metadata/channel mismatch");
  if (splitEqualKeys != nullptr) {
    const auto* encoded = descriptor.get_ptr("splitEqualKeys");
    *splitEqualKeys = encoded != nullptr && encoded->asBool();
  }

  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(keyChannels.size());
  types.reserve(keyChannels.size());
  orders.clear();
  nullOrders.clear();
  for (size_t key = 0; key < keyChannels.size(); ++key) {
    const auto channel = keyChannels[key];
    VELOX_CHECK_LT(
        channel, inputType->size(), "RANGE_PID key channel out of bounds");
    names.push_back(inputType->nameOf(channel));
    types.push_back(inputType->childAt(channel));
    orders.push_back(
        keys[key]["ascending"].asBool() ? cudf::order::ASCENDING
                                        : cudf::order::DESCENDING);
    nullOrders.push_back(
        keys[key]["nullsFirst"].asBool() ? cudf::null_order::BEFORE
                                         : cudf::null_order::AFTER);
  }

  auto boundaryType = ROW(std::move(names), std::move(types));
  auto vector = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(boundaryType, bounds.size(), pool));
  VELOX_CHECK_NOT_NULL(vector);

  for (size_t row = 0; row < bounds.size(); ++row) {
    VELOX_CHECK(bounds[row].isArray(), "RANGE_PID boundary must be an array");
    VELOX_CHECK_EQ(
        bounds[row].size(),
        keyChannels.size(),
        "RANGE_PID boundary width mismatch");
    for (size_t key = 0; key < keyChannels.size(); ++key) {
      const auto& encoded = bounds[row][key];
      auto child = vector->childAt(key);
      if (encoded["isNull"].asBool()) {
        child->setNull(row, true);
        continue;
      }

      const auto& value = encoded["value"];
      const auto& type = boundaryType->childAt(key);
      if (type->isDate()) {
        child->asFlatVector<int32_t>()->set(row, value.asInt());
      } else if (type->isTimestamp()) {
        child->asFlatVector<Timestamp>()->set(
            row, Timestamp::fromMicros(value.asInt()));
      } else {
        switch (type->kind()) {
          case TypeKind::BOOLEAN:
            child->asFlatVector<bool>()->set(row, value.asBool());
            break;
          case TypeKind::TINYINT:
            child->asFlatVector<int8_t>()->set(row, value.asInt());
            break;
          case TypeKind::SMALLINT:
            child->asFlatVector<int16_t>()->set(row, value.asInt());
            break;
          case TypeKind::INTEGER:
            child->asFlatVector<int32_t>()->set(row, value.asInt());
            break;
          case TypeKind::BIGINT:
            child->asFlatVector<int64_t>()->set(row, value.asInt());
            break;
          case TypeKind::REAL:
            child->asFlatVector<float>()->set(
                row, static_cast<float>(value.asDouble()));
            break;
          case TypeKind::DOUBLE:
            child->asFlatVector<double>()->set(row, value.asDouble());
            break;
          case TypeKind::VARCHAR: {
            const auto stringValue = value.asString();
            child->asFlatVector<StringView>()->set(
                row, StringView(stringValue));
            break;
          }
          default:
            VELOX_UNSUPPORTED(
                "RANGE_PID D1 does not support native key type {}",
                type->toString());
        }
      }
    }
  }
  return vector;
}

std::unique_ptr<cudf::column> makeRangePartitionIds(
    cudf::table_view boundaries,
    cudf::table_view keys,
    const std::vector<cudf::order>& orders,
    const std::vector<cudf::null_order>& nullOrders,
    bool splitEqualKeys,
    int32_t splitStart,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) {
  auto lower =
      cudf::lower_bound(boundaries, keys, orders, nullOrders, stream, mr);
  if (!splitEqualKeys || keys.num_rows() == 0) {
    return lower;
  }

  // For a key equal to r repeated boundaries, lower_bound and upper_bound
  // differ by r. Stripe only that equality span. A non-boundary key has a
  // zero-width span and therefore keeps its ordinary lower-bound partition.
  auto upper =
      cudf::upper_bound(boundaries, keys, orders, nullOrders, stream, mr);
  const auto int32Type = cudf::data_type{cudf::type_id::INT32};
  auto widths = cudf::binary_operation(
      upper->view(),
      lower->view(),
      cudf::binary_operator::SUB,
      int32Type,
      stream,
      mr);
  cudf::numeric_scalar<int32_t> one(1, true, stream, mr);
  auto hasMultipleDestinations = cudf::binary_operation(
      widths->view(),
      one,
      cudf::binary_operator::GREATER,
      cudf::data_type{cudf::type_id::BOOL8},
      stream,
      mr);
  auto safeWidths = cudf::copy_if_else(
      widths->view(), one, hasMultipleDestinations->view(), stream, mr);
  cudf::numeric_scalar<int32_t> start(splitStart, true, stream, mr);
  auto rowNumbers = cudf::sequence(keys.num_rows(), start, one, stream, mr);
  auto offsets = cudf::binary_operation(
      rowNumbers->view(),
      safeWidths->view(),
      cudf::binary_operator::MOD,
      int32Type,
      stream,
      mr);
  return cudf::binary_operation(
      lower->view(),
      offsets->view(),
      cudf::binary_operator::ADD,
      int32Type,
      stream,
      mr);
}

std::unique_ptr<core::PartitionFunction> RangePartitionFunctionSpec::create(
    int,
    bool) const {
  return std::make_unique<UcxOnlyRangePartitionFunction>();
}

std::string RangePartitionFunctionSpec::toString() const {
  std::ostringstream keys;
  for (size_t i = 0; i < keyChannels_.size(); ++i) {
    if (i > 0) {
      keys << ", ";
    }
    keys << inputType_->nameOf(keyChannels_[i]);
  }
  return fmt::format("RANGE_PID({})", keys.str());
}

folly::dynamic RangePartitionFunctionSpec::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["name"] = "RangePartitionFunctionSpec";
  obj["inputType"] = inputType_->serialize();
  obj["keyChannels"] = ISerializable::serialize(keyChannels_);
  obj["boundsJson"] = boundsJson_;
  return obj;
}

core::PartitionFunctionSpecPtr RangePartitionFunctionSpec::deserialize(
    const folly::dynamic& obj,
    void* context) {
  return std::make_shared<RangePartitionFunctionSpec>(
      ISerializable::deserialize<RowType>(obj["inputType"]),
      ISerializable::deserialize<std::vector<column_index_t>>(
          obj["keyChannels"], context),
      obj["boundsJson"].asString());
}

} // namespace facebook::velox::ucx_exchange
