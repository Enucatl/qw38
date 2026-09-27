#pragma once

#include "cuda/stream.hpp"
#include <utility>

namespace qw38::cuda {

// Owns capture until finish(), then one executable. Capture never launches it.
class Graph {
 public:
  Graph() = default;
  Graph(Graph&& other) noexcept { swap(other); }
  Graph& operator=(Graph&& other) noexcept {
    if (this == &other) return *this;
    Graph old;
    old.swap(*this);
    swap(other);
    return *this;
  }
  ~Graph();
  Graph(Graph const&) = delete;
  Graph& operator=(Graph const&) = delete;
  static std::expected<Graph, Error> begin(Stream const&);
  std::expected<void, Error> finish();
  std::expected<void, Error> launch(Stream const&) const;
  bool empty() const noexcept { return exec_ == nullptr; }

 private:
  friend struct GraphTestAccess;
  void swap(Graph& other) noexcept {
    std::swap(stream_, other.stream_);
    std::swap(device_, other.device_);
    std::swap(capturing_, other.capturing_);
    std::swap(graph_, other.graph_);
    std::swap(exec_, other.exec_);
  }
  cudaStream_t stream_{};
  int device_{-1};
  bool capturing_{};
  cudaGraph_t graph_{};
  cudaGraphExec_t exec_{};
};

namespace testing { void fail_next_graph_instantiation() noexcept; }

// Public synchronous APIs reject capture before any launch or mutation.
std::expected<void, Error> require_uncaptured(Stream const&);

}  // namespace qw38::cuda
