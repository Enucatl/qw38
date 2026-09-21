#pragma once

#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cuda_runtime.h>

#include <expected>

namespace qw38::cuda {

class Event {
 public:
  Event() noexcept = default;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;
  ~Event();

  Event(Event const&) = delete;
  Event& operator=(Event const&) = delete;

  [[nodiscard]] static std::expected<Event, Error> create();
  [[nodiscard]] static std::expected<Event, Error> create_timing();

  [[nodiscard]] cudaEvent_t native() const noexcept { return event_; }
  [[nodiscard]] bool empty() const noexcept { return event_ == nullptr; }

  [[nodiscard]] std::expected<void, Error> record(Stream const& stream) const;
  [[nodiscard]] std::expected<void, Error> wait(Stream const& stream) const;
  [[nodiscard]] std::expected<void, Error> sync() const;

 private:
  explicit Event(cudaEvent_t event) noexcept : event_(event) {}
  void destroy() noexcept;

  cudaEvent_t event_{nullptr};
};

[[nodiscard]] std::expected<float, Error> elapsed_ms(Event const& start,
                                                     Event const& end);

}  // namespace qw38::cuda
