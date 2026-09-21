#pragma once

#include "cuda/buffer.hpp"
#include "cuda/copy.hpp"
#include "cuda/error.hpp"
#include "cuda/stream.hpp"

#include <cstdint>
#include <expected>
#include <span>

namespace qw38::cuda {

// Allocate a device buffer and copy host bytes onto `stream`. Rejects empty
// payloads so callers can refuse malformed spans before touching the device.
[[nodiscard]] std::expected<DeviceBuffer, Error> upload(
    std::span<std::byte const> host, Stream const& stream);

}  // namespace qw38::cuda
