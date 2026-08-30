#ifndef QW38_DIAGNOSTIC_TRACE_H_
#define QW38_DIAGNOSTIC_TRACE_H_

#ifndef QW38_DIAGNOSTIC_TRACE
#error "diagnostic_trace.h is available only in diagnostic builds"
#endif

#include <array>
#include <cstddef>

#include "qw38/status.h"

namespace qw38::internal {

constexpr std::size_t kTraceAllLayers = 64;
constexpr std::size_t kTraceMaximumRank = 3;

struct TraceFilter final {
  std::size_t layer;
  const char* tap;
};

struct TraceTensorView final {
  const char* name;
  std::size_t layer;
  const float* values;
  std::size_t value_count;
  std::array<std::size_t, kTraceMaximumRank> shape;
  std::size_t rank;
};

using TraceSink = Status (*)(const TraceTensorView&, void*) noexcept;

Status validate_trace_filter(const TraceFilter& filter) noexcept;
bool trace_filter_matches(const TraceFilter& filter, std::size_t layer,
                          const char* tap) noexcept;
Status emit_trace_tensor(const TraceFilter& filter, TraceSink sink,
                         void* context,
                         const TraceTensorView& tensor) noexcept;

}  // namespace qw38::internal

#endif  // QW38_DIAGNOSTIC_TRACE_H_
