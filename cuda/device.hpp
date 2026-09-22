#pragma once

#include "cuda/error.hpp"

#include <cuda_runtime.h>

#include <expected>
#include <string_view>

namespace qw38::cuda {

class DeviceGuard {
 public:
  DeviceGuard(DeviceGuard&& other) noexcept
      : previous_(other.previous_), restore_(other.restore_) {
    other.restore_ = false;
  }

  DeviceGuard& operator=(DeviceGuard&&) = delete;
  DeviceGuard(DeviceGuard const&) = delete;
  DeviceGuard& operator=(DeviceGuard const&) = delete;

  ~DeviceGuard() {
    if (restore_) {
      (void)cudaSetDevice(previous_);
    }
  }

  [[nodiscard]] static std::expected<DeviceGuard, Error> activate(
      int device, std::string_view operation) {
    int current = 0;
    if (auto st = check(cudaGetDevice(&current), "cudaGetDevice"); !st) {
      return std::unexpected(st.error());
    }
    if (current == device) {
      return DeviceGuard{current, false};
    }
    if (auto st = check(cudaSetDevice(device), operation); !st) {
      return std::unexpected(st.error());
    }
    return DeviceGuard{current, true};
  }

 private:
  DeviceGuard(int previous, bool restore) noexcept
      : previous_(previous), restore_(restore) {}

  int previous_{0};
  bool restore_{false};
};

}  // namespace qw38::cuda
