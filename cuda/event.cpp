#include "cuda/event.hpp"

#include "cuda/device.hpp"

namespace qw38::cuda {

Event::Event(Event&& other) noexcept
    : event_(other.event_), device_(other.device_) {
  other.event_ = nullptr;
  other.device_ = -1;
}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    destroy();
    event_ = other.event_;
    device_ = other.device_;
    other.event_ = nullptr;
    other.device_ = -1;
  }
  return *this;
}

Event::~Event() { destroy(); }

void Event::destroy() noexcept {
  if (event_ != nullptr) {
    int previous = 0;
    bool const restore =
        cudaGetDevice(&previous) == cudaSuccess && previous != device_ &&
        cudaSetDevice(device_) == cudaSuccess;
    cudaEventDestroy(event_);
    if (restore) {
      (void)cudaSetDevice(previous);
    }
    event_ = nullptr;
    device_ = -1;
  }
}

std::expected<Event, Error> Event::create() {
  int device = 0;
  if (auto st = check(cudaGetDevice(&device), "cudaGetDevice"); !st) {
    return std::unexpected(st.error());
  }
  cudaEvent_t event = nullptr;
  if (auto st = check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                      "cudaEventCreateWithFlags");
      !st) {
    return std::unexpected(st.error());
  }
  return Event{event, device};
}

std::expected<Event, Error> Event::create_timing() {
  int device = 0;
  if (auto st = check(cudaGetDevice(&device), "cudaGetDevice"); !st) {
    return std::unexpected(st.error());
  }
  cudaEvent_t event = nullptr;
  if (auto st = check(cudaEventCreate(&event), "cudaEventCreate"); !st) {
    return std::unexpected(st.error());
  }
  return Event{event, device};
}

std::expected<float, Error> elapsed_ms(Event const& start, Event const& end) {
  if (start.empty() || end.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "elapsed_ms",
                                      "empty timing event"));
  }
  if (start.device() != end.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "elapsed_ms",
                                      "event devices differ"));
  }
  auto guard =
      DeviceGuard::activate(start.device(), "cudaSetDevice(event.elapsed)");
  if (!guard) {
    return std::unexpected(guard.error());
  }
  float ms = 0.0f;
  if (auto st = check(cudaEventElapsedTime(&ms, start.native(), end.native()),
                      "cudaEventElapsedTime");
      !st) {
    return std::unexpected(st.error());
  }
  return ms;
}

std::expected<void, Error> Event::record(Stream const& stream) const {
  if (event_ == nullptr || stream.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "Event::record",
                                      "empty event or stream"));
  }
  if (device_ != stream.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "Event::record",
                                      "event and stream devices differ"));
  }
  auto guard = stream.activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  return check(cudaEventRecord(event_, stream.native()), "cudaEventRecord");
}

std::expected<void, Error> Event::wait(Stream const& stream) const {
  if (event_ == nullptr || stream.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "Event::wait",
                                      "empty event or stream"));
  }
  if (device_ != stream.device()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "Event::wait",
                                      "event and stream devices differ"));
  }
  auto guard = stream.activate();
  if (!guard) {
    return std::unexpected(guard.error());
  }
  return check(cudaStreamWaitEvent(stream.native(), event_, 0),
               "cudaStreamWaitEvent");
}

std::expected<void, Error> Event::sync() const {
  if (event_ == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "Event::sync", "empty event"));
  }
  auto guard = DeviceGuard::activate(device_, "cudaSetDevice(event.sync)");
  if (!guard) {
    return std::unexpected(guard.error());
  }
  return check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

}  // namespace qw38::cuda
