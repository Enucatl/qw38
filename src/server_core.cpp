#include "server_core.h"

#include <algorithm>
#include <chrono>

namespace qw38::server {

Status SingleFlightGate::acquire(const std::atomic<bool>* cancelled,
                                 QueueTiming* timing) noexcept {
  if (timing == nullptr) {
    return {StatusCode::kInvalidArgument, "queue timing output is required"};
  }
  const auto started = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  if (shutdown_) {
    return {StatusCode::kCancelled, "server queue is shutting down"};
  }
  const std::uint64_t ticket = next_ticket_++;
  timing->depth_on_arrival = waiting_.size() + (active_ ? 1U : 0U);
  waiting_.push_back(ticket);
  for (;;) {
    if (shutdown_ ||
        (cancelled != nullptr && cancelled->load(std::memory_order_relaxed))) {
      const auto position = std::find(waiting_.begin(), waiting_.end(), ticket);
      if (position != waiting_.end()) waiting_.erase(position);
      changed_.notify_all();
      return {StatusCode::kCancelled,
              shutdown_ ? "server queue is shutting down"
                        : "request cancelled while queued"};
    }
    if (!active_ && !waiting_.empty() && waiting_.front() == ticket) {
      waiting_.pop_front();
      active_ = true;
      timing->microseconds = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - started)
              .count());
      return Status::ok();
    }
    changed_.wait_for(lock, std::chrono::milliseconds(5));
  }
}

Status SingleFlightGate::release() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!active_) {
    return {StatusCode::kInvalidArgument,
            "single-flight queue has no active owner"};
  }
  active_ = false;
  changed_.notify_all();
  return Status::ok();
}

void SingleFlightGate::shutdown() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  changed_.notify_all();
}

std::size_t SingleFlightGate::waiting() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return waiting_.size();
}

bool SingleFlightGate::active() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return active_;
}

}  // namespace qw38::server
