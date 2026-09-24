#pragma once

#include <cstdint>

namespace qw38::cuda {

// Device allocation instrumentation for proving load/session setup is the only
// malloc site. Hot-path session operations must not change these counters.
[[nodiscard]] std::uint64_t malloc_count() noexcept;
[[nodiscard]] std::uint64_t free_count() noexcept;
[[nodiscard]] std::uint64_t live_bytes() noexcept;
[[nodiscard]] std::uint64_t peak_live_bytes() noexcept;

void record_malloc(std::uint64_t bytes) noexcept;
void record_free(std::uint64_t bytes) noexcept;

}  // namespace qw38::cuda
