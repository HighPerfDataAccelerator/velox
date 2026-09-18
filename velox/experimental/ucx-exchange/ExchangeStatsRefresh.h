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

#include <chrono>

namespace facebook::velox::ucx_exchange {

// Driver-local cadence for expensive cumulative exchange snapshots. Counters
// continue accumulating at the source; EOF and close always force a snapshot.
class ExchangeStatsRefresh {
 public:
  using Clock = std::chrono::steady_clock;

  explicit ExchangeStatsRefresh(std::chrono::milliseconds interval)
      : interval_(interval) {}

  bool shouldRefresh(Clock::time_point now, bool force) {
    if (!force && initialized_ && now < next_) {
      return false;
    }
    initialized_ = true;
    next_ = now + interval_;
    return true;
  }

 private:
  const std::chrono::milliseconds interval_;
  Clock::time_point next_{};
  bool initialized_{false};
};

} // namespace facebook::velox::ucx_exchange
