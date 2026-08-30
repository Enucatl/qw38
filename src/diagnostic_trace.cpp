#include "diagnostic_trace.h"

#include <cstring>
#include <limits>

namespace qw38::internal {
namespace {

constexpr const char* kTapNames[] = {
    "embedding",
    "input_norm",
    "gdn.packed_qkv",
    "gdn.convolution",
    "gdn.query",
    "gdn.key",
    "gdn.recurrent_output",
    "gdn.recurrent_state",
    "gdn.convolution_state",
    "gdn.output",
    "attention.packed_query_gate",
    "attention.rope_query",
    "attention.rope_key",
    "attention.value",
    "attention.kv_key_row",
    "attention.kv_value_row",
    "attention.output",
    "mixer_residual",
    "ffn.input_norm",
    "ffn.gate",
    "ffn.up",
    "ffn.activated",
    "ffn.correction",
    "layer_residual",
    "final_norm",
    "logits",
};

bool known_tap(const char* tap) noexcept {
  if (tap == nullptr || *tap == '\0') return false;
  for (const char* known : kTapNames) {
    if (std::strcmp(tap, known) == 0) return true;
  }
  return false;
}

}  // namespace

Status validate_trace_filter(const TraceFilter& filter) noexcept {
  if (filter.layer > kTraceAllLayers || filter.tap == nullptr ||
      (std::strcmp(filter.tap, "*") != 0 && !known_tap(filter.tap))) {
    return {StatusCode::kInvalidArgument,
            "diagnostic trace layer or tap filter is invalid"};
  }
  return Status::ok();
}

bool trace_filter_matches(const TraceFilter& filter, std::size_t layer,
                          const char* tap) noexcept {
  return (filter.layer == kTraceAllLayers || filter.layer == layer) &&
         (std::strcmp(filter.tap, "*") == 0 ||
          std::strcmp(filter.tap, tap) == 0);
}

Status emit_trace_tensor(const TraceFilter& filter, TraceSink sink,
                         void* context,
                         const TraceTensorView& tensor) noexcept {
  Status status = validate_trace_filter(filter);
  if (!status.is_ok()) return status;
  if (!known_tap(tensor.name) || tensor.layer > kTraceAllLayers ||
      tensor.values == nullptr || tensor.value_count == 0 || tensor.rank == 0 ||
      tensor.rank > kTraceMaximumRank) {
    return {StatusCode::kInvalidArgument,
            "diagnostic trace tensor view is invalid"};
  }
  std::size_t count = 1;
  for (std::size_t dimension = 0; dimension < tensor.rank; ++dimension) {
    if (tensor.shape[dimension] == 0 ||
        count > std::numeric_limits<std::size_t>::max() /
                    tensor.shape[dimension]) {
      return {StatusCode::kInvalidArgument,
              "diagnostic trace tensor shape is invalid"};
    }
    count *= tensor.shape[dimension];
  }
  if (count != tensor.value_count) {
    return {StatusCode::kInvalidArgument,
            "diagnostic trace tensor shape does not match its values"};
  }
  if (!trace_filter_matches(filter, tensor.layer, tensor.name)) {
    return Status::ok();
  }
  if (sink == nullptr) {
    return {StatusCode::kInvalidArgument,
            "diagnostic trace sink is required for a matching tap"};
  }
  return sink(tensor, context);
}

}  // namespace qw38::internal
