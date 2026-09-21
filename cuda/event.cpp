#include "cuda/event.hpp"

namespace qw38::cuda {

Event::Event(Event&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    destroy();
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

Event::~Event() { destroy(); }

void Event::destroy() noexcept {
  if (event_ != nullptr) {
    cudaEventDestroy(event_);
    event_ = nullptr;
  }
}

std::expected<Event, Error> Event::create() {
  cudaEvent_t event = nullptr;
  if (auto st = check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                      "cudaEventCreateWithFlags");
      !st) {
    return std::unexpected(st.error());
  }
  return Event{event};
}

std::expected<Event, Error> Event::create_timing() {
  cudaEvent_t event = nullptr;
  if (auto st = check(cudaEventCreate(&event), "cudaEventCreate"); !st) {
    return std::unexpected(st.error());
  }
  return Event{event};
}

std::expected<float, Error> elapsed_ms(Event const& start, Event const& end) {
  if (start.empty() || end.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "elapsed_ms",
                                      "empty timing event"));
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
  return check(cudaEventRecord(event_, stream.native()), "cudaEventRecord");
}

std::expected<void, Error> Event::wait(Stream const& stream) const {
  if (event_ == nullptr || stream.empty()) {
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "Event::wait",
                                      "empty event or stream"));
  }
  return check(cudaStreamWaitEvent(stream.native(), event_, 0),
               "cudaStreamWaitEvent");
}

std::expected<void, Error> Event::sync() const {
  if (event_ == nullptr) {
    return std::unexpected(
        make_error(ErrorCode::InvalidArgument, "Event::sync", "empty event"));
  }
  return check(cudaEventSynchronize(event_), "cudaEventSynchronize");
}

}  // namespace qw38::cuda
