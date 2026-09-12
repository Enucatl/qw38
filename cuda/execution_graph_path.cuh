#pragma once

// OPT-055/OPT-096 decode execution-graph selector. Host-includable.
// Production pin stays ffn_only unless OPT-096 acceptance promotes
// decode_segments8.

#include <cstring>

namespace qw38::cuda {

constexpr char kLegalExecutionGraphFfnOnly[] = "ffn_only";
constexpr char kLegalExecutionGraphDecodeSegments8[] = "decode_segments8";

constexpr char kSelectedExecutionGraphPath[] = "ffn_only";

inline constexpr const char* kLegalExecutionGraphPaths[] = {
    kLegalExecutionGraphFfnOnly, kLegalExecutionGraphDecodeSegments8};

constexpr std::size_t kDecodeSegmentLayerCount = 8;
constexpr std::size_t kDecodeSegmentCount = 8;

inline thread_local const char* g_execution_graph_path_override = nullptr;

inline bool legal_execution_graph_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalExecutionGraphFfnOnly) == 0 ||
          std::strcmp(path, kLegalExecutionGraphDecodeSegments8) == 0);
}

inline const char* effective_execution_graph_path() noexcept {
  return g_execution_graph_path_override != nullptr
             ? g_execution_graph_path_override
             : kSelectedExecutionGraphPath;
}

inline bool execution_graph_uses_decode_segments8() noexcept {
  return std::strcmp(effective_execution_graph_path(),
                     kLegalExecutionGraphDecodeSegments8) == 0;
}

inline void set_execution_graph_path_override(const char* path) noexcept {
  g_execution_graph_path_override = path;
}

inline void clear_execution_graph_path_override() noexcept {
  g_execution_graph_path_override = nullptr;
}

struct ExecutionGraphPathScope final {
  explicit ExecutionGraphPathScope(const char* path) noexcept {
    set_execution_graph_path_override(path);
  }
  ~ExecutionGraphPathScope() { clear_execution_graph_path_override(); }
  ExecutionGraphPathScope(const ExecutionGraphPathScope&) = delete;
  ExecutionGraphPathScope& operator=(const ExecutionGraphPathScope&) = delete;
};

}  // namespace qw38::cuda
