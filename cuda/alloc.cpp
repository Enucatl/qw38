#include "cuda/alloc.hpp"

#include <atomic>

namespace qw38::cuda {
namespace {

std::atomic<std::uint64_t> g_mallocs{0};
std::atomic<std::uint64_t> g_frees{0};
std::atomic<std::uint64_t> g_live_bytes{0};

}  // namespace

std::uint64_t malloc_count() noexcept {
  return g_mallocs.load(std::memory_order_relaxed);
}

std::uint64_t free_count() noexcept {
  return g_frees.load(std::memory_order_relaxed);
}

std::uint64_t live_bytes() noexcept {
  return g_live_bytes.load(std::memory_order_relaxed);
}

void record_malloc(std::uint64_t bytes) noexcept {
  g_mallocs.fetch_add(1, std::memory_order_relaxed);
  g_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

void record_free(std::uint64_t bytes) noexcept {
  g_frees.fetch_add(1, std::memory_order_relaxed);
  g_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
}

}  // namespace qw38::cuda
