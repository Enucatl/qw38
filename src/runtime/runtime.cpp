#include "runtime/runtime.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <new>

namespace qw38::runtime {
namespace {

Error closed_error(std::string_view operation) {
  return make_error(ErrorCode::RuntimeClosed, operation,
                    "runtime has been shut down");
}

}  // namespace

std::expected<Runtime, Error> Runtime::create() {
  try {
    int device = 0;
    if (auto st = qw38::cuda::check(cudaGetDevice(&device), "cudaGetDevice");
        !st) {
      return std::unexpected(from_cuda(st.error()));
    }
    auto stream = qw38::cuda::Stream::create();
    if (!stream) {
      return std::unexpected(from_cuda(stream.error()));
    }
    Runtime rt;
    rt.device_ = device;
    rt.stream_ = std::make_shared<qw38::cuda::Stream>(std::move(*stream));
    return rt;
  } catch (std::bad_alloc const&) {
    return std::unexpected(make_error(ErrorCode::AllocationFailed, "runtime.create",
                                      "host allocation failed"));
  }
}

std::expected<void, Error> Runtime::shutdown() {
  if (!stream_) {
    return {};
  }
  if (stream_.use_count() != 1) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "runtime.shutdown",
        "stream is still shared by a Session or moved Runtime"));
  }

  auto synchronized = stream_->sync();
  auto closed = stream_->close();
  stream_.reset();
  if (!synchronized) {
    return std::unexpected(from_cuda(synchronized.error()));
  }
  if (!closed) {
    return std::unexpected(from_cuda(closed.error()));
  }
  return {};
}

std::expected<Model, Error> Runtime::load(std::filesystem::path const& path) {
  if (!stream_) {
    return std::unexpected(closed_error("runtime.load"));
  }
  auto art = qw38::format::Artifact::open(path);
  if (!art) {
    return std::unexpected(from_format(art.error()));
  }
  return upload(*art);
}

std::expected<Model, Error> Runtime::upload(
    qw38::format::Artifact const& artifact) {
  if (!stream_) {
    return std::unexpected(closed_error("runtime.upload"));
  }
  return Model::upload(artifact, *stream_);
}

std::expected<Session, Error> Runtime::create_session(Model const& model,
                                                      std::uint64_t kv_capacity) {
  if (!stream_) {
    return std::unexpected(closed_error("runtime.create_session"));
  }
  if (model.device() != device_) {
    return std::unexpected(make_error(
        ErrorCode::InvalidArgument, "runtime.create_session",
        "model and runtime devices differ"));
  }
  return Session::create(model, stream_, kv_capacity);
}

}  // namespace qw38::runtime
