#include "cuda/stream.hpp"

#include <atomic>

namespace qw38::cuda {
namespace {

std::atomic<bool> g_fail_next_stream_sync{false};

}  // namespace

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), device_(other.device_) {
  other.stream_ = nullptr;
  other.device_ = -1;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    destroy();
    stream_ = other.stream_;
    device_ = other.device_;
    other.stream_ = nullptr;
    other.device_ = -1;
  }
  return *this;
}

Stream::~Stream() { destroy(); }

void Stream::destroy() noexcept {
  if (stream_ != nullptr) {
    int previous = 0;
    bool const restore =
        cudaGetDevice(&previous) == cudaSuccess && previous != device_ &&
        cudaSetDevice(device_) == cudaSuccess;
    cudaStreamDestroy(stream_);
    if (restore) {
      (void)cudaSetDevice(previous);
    }
    stream_ = nullptr;
    device_ = -1;
  }
}

std::expected<Stream, Error> Stream::create() {
  int device = 0;
  if (auto st = check(cudaGetDevice(&device), "cudaGetDevice"); !st) {
    return std::unexpected(st.error());
  }
  cudaStream_t stream = nullptr;
  if (auto st = check(cudaStreamCreateWithFlags(&stream, cudaStreamDefault),
                      "cudaStreamCreateWithFlags");
      !st) {
    return std::unexpected(st.error());
  }
  return Stream{stream, device};
}

std::expected<DeviceGuard, Error> Stream::activate() const {
  if (stream_ == nullptr) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
                                      "Stream::activate", "empty stream"));
  }
  return DeviceGuard::activate(device_, "cudaSetDevice(stream)");
}

std::expected<void, Error> Stream::sync() const {
  if (stream_ == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "Stream::sync", "empty stream"));
  }
  auto guard = activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  if (auto st = check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
      !st) {
    return st;
  }
  if (g_fail_next_stream_sync.exchange(false, std::memory_order_relaxed)) {
    return std::unexpected(make_error(
        ErrorCode::Status, "cudaStreamSynchronize",
        "injected deferred stream failure"));
  }
  return {};
}

void testing::fail_next_stream_sync() noexcept {
  g_fail_next_stream_sync.store(true, std::memory_order_relaxed);
}

}  // namespace qw38::cuda
