#pragma once

#include "cuda/error.hpp"

#include <cuda_runtime.h>

#include <expected>

namespace qw38::cuda {

class Stream {
 public:
  Stream() noexcept = default;
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;
  ~Stream();

  Stream(Stream const&) = delete;
  Stream& operator=(Stream const&) = delete;

  // One ordered eager stream (cudaStreamDefault). Not non-blocking, not a graph.
  [[nodiscard]] static std::expected<Stream, Error> create();

  [[nodiscard]] cudaStream_t native() const noexcept { return stream_; }
  [[nodiscard]] bool empty() const noexcept { return stream_ == nullptr; }

  [[nodiscard]] std::expected<void, Error> sync() const;

 private:
  explicit Stream(cudaStream_t stream) noexcept : stream_(stream) {}
  void destroy() noexcept;

  cudaStream_t stream_{nullptr};
};

}  // namespace qw38::cuda
