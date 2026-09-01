#ifndef QW38_SERVER_CORE_H_
#define QW38_SERVER_CORE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <condition_variable>

#include "qw38/status.h"

namespace qw38::server {

struct QueueTiming final {
  std::uint64_t microseconds = 0;
  std::size_t depth_on_arrival = 0;
};

class SingleFlightGate final {
 public:
  SingleFlightGate() noexcept = default;
  ~SingleFlightGate() = default;
  SingleFlightGate(const SingleFlightGate&) = delete;
  SingleFlightGate& operator=(const SingleFlightGate&) = delete;

  Status acquire(const std::atomic<bool>* cancelled,
                 QueueTiming* timing) noexcept;
  Status release() noexcept;
  void shutdown() noexcept;
  std::size_t waiting() const noexcept;
  bool active() const noexcept;

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::uint64_t> waiting_;
  std::uint64_t next_ticket_ = 0;
  bool active_ = false;
  bool shutdown_ = false;
};

}  // namespace qw38::server

#endif  // QW38_SERVER_CORE_H_
