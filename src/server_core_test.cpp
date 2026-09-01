#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "qw38/status.h"
#include "server_core.h"

namespace {

bool wait_for_depth(const qw38::server::SingleFlightGate& gate,
                    std::size_t depth) {
  for (int attempt = 0; attempt < 200; ++attempt) {
    if (gate.waiting() == depth) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

}  // namespace

int main() {
  qw38::server::SingleFlightGate gate;
  std::atomic<bool> never_cancel{false};
  qw38::server::QueueTiming first;
  qw38::Status status = gate.acquire(&never_cancel, &first);
  if (!status.is_ok() || first.depth_on_arrival != 0 || !gate.active()) return 1;

  qw38::server::QueueTiming second;
  qw38::Status second_status;
  std::atomic<int> active{1};
  std::atomic<int> maximum_active{1};
  std::thread second_thread([&] {
    second_status = gate.acquire(&never_cancel, &second);
    if (!second_status.is_ok()) return;
    const int now = active.fetch_add(1) + 1;
    int maximum = maximum_active.load();
    while (now > maximum &&
           !maximum_active.compare_exchange_weak(maximum, now)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    active.fetch_sub(1);
    gate.release();
  });
  const bool second_queued = wait_for_depth(gate, 1);

  std::atomic<bool> cancel_third{false};
  qw38::server::QueueTiming third;
  qw38::Status third_status;
  std::thread third_thread(
      [&] { third_status = gate.acquire(&cancel_third, &third); });
  const bool third_queued = wait_for_depth(gate, 2);
  cancel_third.store(true);
  third_thread.join();
  const bool third_removed = wait_for_depth(gate, 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  active.fetch_sub(1);
  status = gate.release();
  second_thread.join();
  if (!second_queued || !third_queued || !third_removed || !status.is_ok() ||
      !second_status.is_ok() ||
      third_status.code() != qw38::StatusCode::kCancelled ||
      maximum_active.load() != 1 || second.depth_on_arrival != 1 ||
      second.microseconds < 30000 || gate.active() || gate.waiting() != 0) {
    return 1;
  }

  qw38::server::QueueTiming held;
  status = gate.acquire(nullptr, &held);
  if (!status.is_ok()) return 1;
  qw38::server::QueueTiming shutdown_wait;
  qw38::Status shutdown_status;
  std::thread shutdown_thread(
      [&] { shutdown_status = gate.acquire(nullptr, &shutdown_wait); });
  const bool shutdown_queued = wait_for_depth(gate, 1);
  gate.shutdown();
  shutdown_thread.join();
  status = gate.release();
  qw38::server::QueueTiming stopped;
  const qw38::Status stopped_status = gate.acquire(nullptr, &stopped);
  if (!shutdown_queued || !status.is_ok() ||
      shutdown_status.code() != qw38::StatusCode::kCancelled ||
      stopped_status.code() != qw38::StatusCode::kCancelled) {
    return 1;
  }

  std::cout << "queue_case=fifo_single_flight depth=" << second.depth_on_arrival
            << " wait_us=" << second.microseconds
            << " max_active=" << maximum_active.load() << " passed=true\n";
  std::cout << "queue_case=cancelled_waiter status="
            << qw38::status_code_name(third_status.code())
            << " removed=true passed=true\n";
  std::cout << "queue_case=shutdown status="
            << qw38::status_code_name(shutdown_status.code())
            << " woke_waiter=true passed=true\nstatus=passed\n";
  return 0;
}
