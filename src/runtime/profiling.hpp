#pragma once

#include <nvtx3/nvToolsExt.h>
#include <cstdio>

namespace qw38::runtime::profiling {

// Profiling is opt-in so ordinary decode does not emit NVTX ranges.
inline thread_local bool enabled = false;

class ScopedRange {
 public:
  explicit ScopedRange(char const* name) noexcept : active_(enabled) {
    if (active_) nvtxRangePushA(name);
  }
  template<class... Args>
  ScopedRange(char const* format, Args... args) noexcept : active_(enabled) {
    if (active_) {
      char name[384];
      std::snprintf(name, sizeof(name), format, args...);
      nvtxRangePushA(name);
    }
  }
  ~ScopedRange() { close(); }
  void close() noexcept {
    if (active_) nvtxRangePop();
    active_ = false;
  }
  ScopedRange(ScopedRange const&) = delete;
  ScopedRange& operator=(ScopedRange const&) = delete;

 private:
  bool active_;
};

}  // namespace qw38::runtime::profiling
