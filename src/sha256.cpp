#include "sha256.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace qw38::internal {
namespace {

constexpr std::array<std::uint32_t, 64> kRound = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t rotate(std::uint32_t value, std::uint32_t count) noexcept {
  return (value >> count) | (value << (32U - count));
}

class Sha256 final {
 public:
  void update(const unsigned char* data, std::size_t size) noexcept {
    total_bytes_ += size;
    for (std::size_t index = 0; index < size; ++index) {
      block_[block_bytes_++] = data[index];
      if (block_bytes_ == block_.size()) {
        transform();
        block_bytes_ = 0;
      }
    }
  }

  std::array<std::uint32_t, 8> finish() noexcept {
    const std::uint64_t bits = total_bytes_ * 8U;
    block_[block_bytes_++] = 0x80U;
    if (block_bytes_ > 56) {
      while (block_bytes_ < block_.size()) block_[block_bytes_++] = 0;
      transform();
      block_bytes_ = 0;
    }
    while (block_bytes_ < 56) block_[block_bytes_++] = 0;
    for (std::uint32_t index = 0; index < 8; ++index) {
      block_[63U - index] = static_cast<unsigned char>(bits >> (index * 8U));
    }
    transform();
    return state_;
  }

 private:
  void transform() noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::uint32_t index = 0; index < 16; ++index) {
      const std::uint32_t offset = index * 4U;
      words[index] = (static_cast<std::uint32_t>(block_[offset]) << 24U) |
                     (static_cast<std::uint32_t>(block_[offset + 1]) << 16U) |
                     (static_cast<std::uint32_t>(block_[offset + 2]) << 8U) |
                     static_cast<std::uint32_t>(block_[offset + 3]);
    }
    for (std::uint32_t index = 16; index < 64; ++index) {
      const std::uint32_t s0 = rotate(words[index - 15], 7) ^
                               rotate(words[index - 15], 18) ^
                               (words[index - 15] >> 3U);
      const std::uint32_t s1 = rotate(words[index - 2], 17) ^
                               rotate(words[index - 2], 19) ^
                               (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::uint32_t index = 0; index < 64; ++index) {
      const std::uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t first = h + s1 + choice + kRound[index] + words[index];
      const std::uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t second = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {0x6a09e667U, 0xbb67ae85U,
                                          0x3c6ef372U, 0xa54ff53aU,
                                          0x510e527fU, 0x9b05688cU,
                                          0x1f83d9abU, 0x5be0cd19U};
  std::array<unsigned char, 64> block_{};
  std::size_t block_bytes_ = 0;
  std::uint64_t total_bytes_ = 0;
};

std::string hex_digest(const std::array<std::uint32_t, 8>& words) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint32_t word : words) output << std::setw(8) << word;
  return output.str();
}

}  // namespace

Status sha256_file(const std::string& path, std::string* digest) noexcept {
  if (digest == nullptr || path.empty()) {
    return {StatusCode::kInvalidArgument, "SHA-256 path and output are required"};
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {StatusCode::kIoError, "cannot open file for SHA-256"};
  }
  Sha256 hash;
  std::array<unsigned char, 1024 * 1024> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) hash.update(buffer.data(), static_cast<std::size_t>(count));
  }
  if (!input.eof()) {
    return {StatusCode::kIoError, "failed while reading file for SHA-256"};
  }
  *digest = hex_digest(hash.finish());
  return Status::ok();
}

Status sha256_file_prefix(const std::string& path, std::size_t bytes,
                          std::string* digest) noexcept {
  if (digest == nullptr || path.empty()) {
    return {StatusCode::kInvalidArgument,
            "SHA-256 prefix path and output are required"};
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {StatusCode::kIoError, "cannot open file for SHA-256 prefix"};
  }
  Sha256 hash;
  std::array<unsigned char, 1024 * 1024> buffer{};
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const std::size_t request = std::min(remaining, buffer.size());
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(request));
    if (input.gcount() != static_cast<std::streamsize>(request)) {
      return {StatusCode::kIoError, "SHA-256 prefix is shorter than declared"};
    }
    hash.update(buffer.data(), request);
    remaining -= request;
  }
  *digest = hex_digest(hash.finish());
  return Status::ok();
}

Status sha256_bytes(const unsigned char* data, std::size_t size,
                    std::string* digest) noexcept {
  if (digest == nullptr || (data == nullptr && size != 0)) {
    return {StatusCode::kInvalidArgument,
            "SHA-256 bytes and output are required"};
  }
  Sha256 hash;
  hash.update(data, size);
  *digest = hex_digest(hash.finish());
  return Status::ok();
}

}  // namespace qw38::internal
