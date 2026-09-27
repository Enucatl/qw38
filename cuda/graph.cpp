#include "cuda/graph.hpp"
#include <atomic>

namespace qw38::cuda {
namespace { std::atomic<bool> fail_instantiation{false}; }
void testing::fail_next_graph_instantiation() noexcept {
  fail_instantiation.store(true, std::memory_order_relaxed);
}

Graph::~Graph() {
  if (device_ < 0) return;
  auto guard = DeviceGuard::activate(device_, "graph.destroy");
  if (!guard) return;
  if (capturing_) (void)cudaStreamEndCapture(stream_, &graph_);
  if (exec_) (void)cudaGraphExecDestroy(exec_);
  if (graph_) (void)cudaGraphDestroy(graph_);
}

std::expected<void, Error> require_uncaptured(Stream const& stream) {
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  cudaStreamCaptureStatus status{};
  if (auto st = check(cudaStreamIsCapturing(stream.native(), &status),
                      "cudaStreamIsCapturing"); !st) return st;
  if (status != cudaStreamCaptureStatusNone)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,
        "stream.capture", "public completion API does not support capture"));
  return {};
}

std::expected<Graph, Error> Graph::begin(Stream const& stream) {
  if (auto st = require_uncaptured(stream); !st) return std::unexpected(st.error());
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  if (auto st = check(cudaStreamBeginCapture(stream.native(), cudaStreamCaptureModeThreadLocal),
                      "cudaStreamBeginCapture"); !st) return std::unexpected(st.error());
  Graph result;
  result.stream_ = stream.native();
  result.device_ = stream.device();
  result.capturing_ = true;
  return result;
}

std::expected<void, Error> Graph::finish() {
  auto guard = DeviceGuard::activate(device_, "graph.finish");
  if (!guard) return std::unexpected(guard.error());
  capturing_ = false;
  if (auto st = check(cudaStreamEndCapture(stream_, &graph_), "cudaStreamEndCapture"); !st) return st;
  if (fail_instantiation.exchange(false, std::memory_order_relaxed))
    return std::unexpected(make_error(ErrorCode::Status, "cudaGraphInstantiate",
                                      "injected construction failure before execution"));
  if (auto st = check(cudaGraphInstantiate(&exec_, graph_, 0), "cudaGraphInstantiate"); !st) return st;
  auto const source = std::exchange(graph_, nullptr);
  return check(cudaGraphDestroy(source), "cudaGraphDestroy(source)");
}

std::expected<void, Error> Graph::launch(Stream const& stream) const {
  if (!exec_ || stream.native() != stream_ || stream.device() != device_)
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "graph.launch", "graph stream mismatch"));
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  return check(cudaGraphLaunch(exec_, stream.native()), "cudaGraphLaunch");
}

}  // namespace qw38::cuda
