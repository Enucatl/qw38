#pragma once

#include <nvtx3/nvToolsExt.h>

namespace qw38::runtime::profiling {

// Profiling is opt-in so ordinary decode does not emit NVTX ranges.
inline thread_local bool enabled = false;

class ScopedRange {
 public:
  explicit ScopedRange(char const* name) noexcept : active_(enabled) {
    if (active_) nvtxRangePushA(name);
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
