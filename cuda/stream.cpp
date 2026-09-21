#include "cuda/stream.hpp"

namespace qw38::cuda {

Stream::Stream(Stream&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    destroy();
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

Stream::~Stream() { destroy(); }

void Stream::destroy() noexcept {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

std::expected<Stream, Error> Stream::create() {
  cudaStream_t stream = nullptr;
  if (auto st = check(cudaStreamCreateWithFlags(&stream, cudaStreamDefault),
                      "cudaStreamCreateWithFlags");
      !st) {
    return std::unexpected(st.error());
  }
  return Stream{stream};
}

std::expected<void, Error> Stream::sync() const {
  if (stream_ == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "Stream::sync", "empty stream"));
  }
  return check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace qw38::cuda
