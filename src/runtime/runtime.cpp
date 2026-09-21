#include "runtime/runtime.hpp"

#include <cuda_runtime.h>

namespace qw38::runtime {

std::expected<Runtime, Error> Runtime::create() {
  int device = 0;
  if (auto st = qw38::cuda::check(cudaGetDevice(&device), "cudaGetDevice"); !st) {
    return std::unexpected(from_cuda(st.error()));
  }
  auto stream = qw38::cuda::Stream::create();
  if (!stream) {
    return std::unexpected(from_cuda(stream.error()));
  }
  Runtime rt;
  rt.device_ = device;
  rt.stream_ = std::move(*stream);
  return rt;
}

std::expected<Model, Error> Runtime::load(std::filesystem::path const& path) {
  auto art = qw38::format::Artifact::open(path);
  if (!art) {
    return std::unexpected(from_format(art.error()));
  }
  return upload(*art);
}

std::expected<Model, Error> Runtime::upload(
    qw38::format::Artifact const& artifact) {
  return Model::upload(artifact, stream_);
}

std::expected<Session, Error> Runtime::create_session(Model const& model,
                                                      std::uint64_t kv_capacity) {
  return Session::create(model, stream_, kv_capacity);
}

}  // namespace qw38::runtime
