#include "full_scheduler.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <nvtx3/nvToolsExt.h>

#include "scheduler.h"
#include "sha256.h"

namespace qw38::cuda {
namespace {

constexpr std::array<char, 8> kMagic{'Q', 'W', '3', '8', 'C', 'K', 'P', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 248;
constexpr std::size_t kDigestBytes = 64;
constexpr std::size_t kChunkBytes = 1024 * 1024;
constexpr std::size_t kGdnLayers = 48;
constexpr std::size_t kAttentionLayers = 16;
constexpr const char* kModelSha256 =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLayoutSha256 =
    "b816f20fc25c3de29b71e0404a3b7ef829142e7ad41aad0d67bcfbb92c7550fd";

class PersistenceTiming final {
 public:
  PersistenceTiming(const char* range, RuntimeTimings* timings) noexcept
      : timings_(timings), started_(std::chrono::steady_clock::now()) {
    nvtxRangePushA(range);
  }
  ~PersistenceTiming() {
    nvtxRangePop();
    if (timings_ != nullptr) {
      timings_->persistence = {
          static_cast<float>(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started_)
                                 .count()),
          true};
    }
  }
  PersistenceTiming(const PersistenceTiming&) = delete;
  PersistenceTiming& operator=(const PersistenceTiming&) = delete;

 private:
  RuntimeTimings* timings_;
  std::chrono::steady_clock::time_point started_;
};

void append_u32(std::vector<unsigned char>* output, std::uint32_t value) {
  for (std::size_t byte = 0; byte < 4; ++byte) {
    output->push_back(static_cast<unsigned char>(value >> (byte * 8)));
  }
}

void append_u64(std::vector<unsigned char>* output, std::uint64_t value) {
  for (std::size_t byte = 0; byte < 8; ++byte) {
    output->push_back(static_cast<unsigned char>(value >> (byte * 8)));
  }
}

std::uint32_t float_bits(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(std::uint32_t bits) noexcept {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool read_u32(const std::vector<unsigned char>& input, std::size_t* offset,
              std::uint32_t* value) noexcept {
  if (*offset > input.size() || input.size() - *offset < 4) return false;
  *value = 0;
  for (std::size_t byte = 0; byte < 4; ++byte) {
    *value |= static_cast<std::uint32_t>(input[*offset + byte]) << (byte * 8);
  }
  *offset += 4;
  return true;
}

bool read_u64(const std::vector<unsigned char>& input, std::size_t* offset,
              std::uint64_t* value) noexcept {
  if (*offset > input.size() || input.size() - *offset < 8) return false;
  *value = 0;
  for (std::size_t byte = 0; byte < 8; ++byte) {
    *value |= static_cast<std::uint64_t>(input[*offset + byte]) << (byte * 8);
  }
  *offset += 8;
  return true;
}

Status write_device(std::ofstream* output, const void* device,
                    std::size_t bytes,
                    std::vector<unsigned char>* buffer) noexcept {
  const auto* source = static_cast<const unsigned char*>(device);
  for (std::size_t offset = 0; offset < bytes;) {
    const std::size_t count = std::min(buffer->size(), bytes - offset);
    const cudaError_t error = cudaMemcpy(buffer->data(), source + offset, count,
                                         cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      return {StatusCode::kInternal, "cannot copy CUDA checkpoint state"};
    }
    output->write(reinterpret_cast<const char*>(buffer->data()),
                  static_cast<std::streamsize>(count));
    if (!*output) return {StatusCode::kIoError, "cannot write checkpoint"};
    offset += count;
  }
  return Status::ok();
}

Status read_device(std::ifstream* input, void* device, std::size_t bytes,
                   std::vector<unsigned char>* buffer) noexcept {
  auto* destination = static_cast<unsigned char*>(device);
  for (std::size_t offset = 0; offset < bytes;) {
    const std::size_t count = std::min(buffer->size(), bytes - offset);
    input->read(reinterpret_cast<char*>(buffer->data()),
                static_cast<std::streamsize>(count));
    if (input->gcount() != static_cast<std::streamsize>(count)) {
      return {StatusCode::kIoError, "checkpoint payload is truncated"};
    }
    const cudaError_t error = cudaMemcpy(destination + offset, buffer->data(),
                                         count, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      return {StatusCode::kInternal, "cannot restore CUDA checkpoint state"};
    }
    offset += count;
  }
  return Status::ok();
}

bool add_size(std::size_t value, std::size_t* total) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - *total) return false;
  *total += value;
  return true;
}

bool sync_path(const std::string& path, bool directory) noexcept {
  const int flags = directory ? O_RDONLY | O_DIRECTORY : O_RDONLY;
  const int descriptor = open(path.c_str(), flags);
  if (descriptor < 0) return false;
  const bool synced = fsync(descriptor) == 0;
  const bool closed = close(descriptor) == 0;
  return synced && closed;
}

}  // namespace

Status SchedulerSession::save_checkpoint(const std::string& path,
                                         RuntimeTimings* timings) const noexcept {
  const PersistenceTiming timing("qw38.persistence.save", timings);
  if (capacity_ == 0 || path.empty()) {
    return {StatusCode::kInvalidArgument,
            "initialized session and checkpoint path are required"};
  }
  const std::size_t token_bytes = frontier_ * sizeof(std::uint32_t);
  const std::size_t convolution_bytes =
      kGdnLayers * internal::kGdnConvolutionValues * sizeof(float);
  const std::size_t recurrent_bytes =
      kGdnLayers * internal::kGdnRecurrentStateValues * sizeof(float);
  const std::size_t key_bytes = kAttentionLayers * frontier_ *
                                internal::kAttentionKvWidth *
                                sizeof(__nv_bfloat16);
  const std::size_t value_bytes = key_bytes;
  const std::size_t logits_bytes =
      frontier_ == 0 ? 0 : internal::kVocabularySize * sizeof(float);
  const std::size_t hidden_bytes =
      frontier_ == 0 ? 0 : internal::kResidualWidth * sizeof(float);

  std::vector<unsigned char> header;
  header.reserve(kHeaderBytes);
  header.insert(header.end(), kMagic.begin(), kMagic.end());
  append_u32(&header, kVersion);
  append_u32(&header, kHeaderBytes);
  header.insert(header.end(), kModelSha256, kModelSha256 + 64);
  header.insert(header.end(), kLayoutSha256, kLayoutSha256 + 64);
  append_u64(&header, capacity_);
  append_u64(&header, frontier_);
  append_u32(&header, float_bits(sampler_state_.temperature));
  append_u32(&header, float_bits(sampler_state_.top_p));
  append_u32(&header, sampler_state_.top_k);
  append_u32(&header, 0);
  append_u64(&header, sampler_state_.seed);
  append_u64(&header, sampler_state_.rng_state);
  for (std::size_t bytes : {token_bytes, convolution_bytes, recurrent_bytes,
                            key_bytes, value_bytes, logits_bytes,
                            hidden_bytes}) {
    append_u64(&header, bytes);
  }
  if (header.size() != kHeaderBytes) {
    return {StatusCode::kInternal, "checkpoint header layout is inconsistent"};
  }

  const std::string temporary = path + ".tmp";
  std::remove(temporary.c_str());
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) return {StatusCode::kIoError, "cannot create checkpoint temp"};
  output.write(reinterpret_cast<const char*>(header.data()), header.size());
  for (std::size_t index = 0; output && index < frontier_; ++index) {
    const std::uint32_t token = static_cast<std::uint32_t>(tokens_[index]);
    std::array<unsigned char, 4> encoded{};
    for (std::size_t byte = 0; byte < encoded.size(); ++byte) {
      encoded[byte] = static_cast<unsigned char>(token >> (byte * 8));
    }
    output.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
  }
  std::vector<unsigned char> buffer(kChunkBytes);
  Status status = output ? write_device(&output, gdn_convolution_,
                                        convolution_bytes, &buffer)
                         : Status{StatusCode::kIoError,
                                  "cannot write checkpoint tokens"};
  if (status.is_ok()) {
    status = write_device(&output, gdn_recurrent_, recurrent_bytes, &buffer);
  }
  const std::size_t cache_stride = capacity_ * internal::kAttentionKvWidth;
  const std::size_t layer_bytes =
      frontier_ * internal::kAttentionKvWidth * sizeof(__nv_bfloat16);
  for (std::size_t layer = 0; status.is_ok() && layer < kAttentionLayers;
       ++layer) {
    status = write_device(&output, attention_key_ + layer * cache_stride,
                          layer_bytes, &buffer);
  }
  for (std::size_t layer = 0; status.is_ok() && layer < kAttentionLayers;
       ++layer) {
    status = write_device(&output, attention_value_ + layer * cache_stride,
                          layer_bytes, &buffer);
  }
  if (status.is_ok() && logits_bytes != 0) {
    output.write(reinterpret_cast<const char*>(last_logits_), logits_bytes);
    if (!output) status = {StatusCode::kIoError, "cannot write checkpoint logits"};
  }
  if (status.is_ok() && hidden_bytes != 0) {
    output.write(reinterpret_cast<const char*>(last_hidden_), hidden_bytes);
    if (!output) status = {StatusCode::kIoError, "cannot write checkpoint hidden"};
  }
  output.close();
  if (!status.is_ok() || !output) {
    std::remove(temporary.c_str());
    return status.is_ok() ? Status{StatusCode::kIoError,
                                   "cannot close checkpoint temp"}
                          : status;
  }
  std::size_t authenticated_bytes = kHeaderBytes;
  for (std::size_t bytes : {token_bytes, convolution_bytes, recurrent_bytes,
                            key_bytes, value_bytes, logits_bytes,
                            hidden_bytes}) {
    if (!add_size(bytes, &authenticated_bytes)) {
      std::remove(temporary.c_str());
      return {StatusCode::kResourceExhausted,
              "checkpoint byte count overflows the host"};
    }
  }
  std::string digest;
  status = internal::sha256_file_prefix(temporary, authenticated_bytes, &digest);
  if (status.is_ok()) {
    std::ofstream footer(temporary, std::ios::binary | std::ios::app);
    footer.write(digest.data(), digest.size());
    footer.close();
    if (!footer) status = {StatusCode::kIoError, "cannot append checkpoint digest"};
  }
  if (status.is_ok() && !sync_path(temporary, false)) {
    status = {StatusCode::kIoError, "cannot synchronize checkpoint temp"};
  }
  if (!status.is_ok() || std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::remove(temporary.c_str());
    return status.is_ok()
               ? Status{StatusCode::kIoError, "cannot publish checkpoint"}
               : status;
  }
  std::filesystem::path parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) parent = ".";
  if (!sync_path(parent.string(), true)) {
    return {StatusCode::kIoError,
            "checkpoint published but directory sync failed"};
  }
  return Status::ok();
}

Status SchedulerSession::restore_checkpoint(
    const std::string& path, SchedulerWorkspace* workspace,
    RuntimeTimings* timings) noexcept {
  const PersistenceTiming timing("qw38.persistence.restore", timings);
  if (capacity_ == 0 || path.empty() || workspace == nullptr ||
      workspace->capacity_ != capacity_) {
    return {StatusCode::kInvalidArgument,
            "checkpoint restore session, workspace, and path are required"};
  }
  std::error_code file_error;
  const std::uintmax_t file_bytes = std::filesystem::file_size(path, file_error);
  if (file_error || file_bytes < kHeaderBytes + kDigestBytes ||
      file_bytes > std::numeric_limits<std::size_t>::max()) {
    return {StatusCode::kIoError, "checkpoint file size is invalid"};
  }
  std::ifstream input(path, std::ios::binary);
  std::vector<unsigned char> header(kHeaderBytes);
  input.read(reinterpret_cast<char*>(header.data()), header.size());
  if (input.gcount() != static_cast<std::streamsize>(header.size())) {
    return {StatusCode::kIoError, "checkpoint header is truncated"};
  }
  std::size_t offset = 0;
  if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
    return {StatusCode::kIncompatibleArtifact, "checkpoint magic is invalid"};
  }
  offset += kMagic.size();
  std::uint32_t version = 0;
  std::uint32_t header_bytes = 0;
  if (!read_u32(header, &offset, &version) ||
      !read_u32(header, &offset, &header_bytes) || version != kVersion ||
      header_bytes != kHeaderBytes) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint version or header size is incompatible"};
  }
  if (std::memcmp(header.data() + offset, kModelSha256, 64) != 0) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint model hash is incompatible"};
  }
  offset += 64;
  if (std::memcmp(header.data() + offset, kLayoutSha256, 64) != 0) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint layout hash is incompatible"};
  }
  offset += 64;
  std::uint64_t saved_capacity = 0;
  std::uint64_t frontier = 0;
  std::uint32_t temperature_bits = 0;
  std::uint32_t top_p_bits = 0;
  std::uint32_t top_k = 0;
  std::uint32_t reserved = 0;
  std::uint64_t seed = 0;
  std::uint64_t rng_state = 0;
  if (!read_u64(header, &offset, &saved_capacity) ||
      !read_u64(header, &offset, &frontier) ||
      !read_u32(header, &offset, &temperature_bits) ||
      !read_u32(header, &offset, &top_p_bits) ||
      !read_u32(header, &offset, &top_k) ||
      !read_u32(header, &offset, &reserved) ||
      !read_u64(header, &offset, &seed) ||
      !read_u64(header, &offset, &rng_state) || reserved != 0 ||
      saved_capacity == 0 || saved_capacity > 131072 || frontier > capacity_ ||
      frontier > saved_capacity) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint capacity, frontier, or sampler framing is invalid"};
  }
  std::array<std::uint64_t, 7> sections{};
  for (std::uint64_t& section : sections) {
    if (!read_u64(header, &offset, &section)) {
      return {StatusCode::kIncompatibleArtifact,
              "checkpoint section table is truncated"};
    }
  }
  const std::size_t expected_tokens = frontier * sizeof(std::uint32_t);
  const std::size_t expected_convolution =
      kGdnLayers * internal::kGdnConvolutionValues * sizeof(float);
  const std::size_t expected_recurrent =
      kGdnLayers * internal::kGdnRecurrentStateValues * sizeof(float);
  const std::size_t expected_kv = kAttentionLayers * frontier *
                                  internal::kAttentionKvWidth *
                                  sizeof(__nv_bfloat16);
  const std::size_t expected_logits =
      frontier == 0 ? 0 : internal::kVocabularySize * sizeof(float);
  const std::size_t expected_hidden =
      frontier == 0 ? 0 : internal::kResidualWidth * sizeof(float);
  const std::array<std::size_t, 7> expected{
      expected_tokens, expected_convolution, expected_recurrent, expected_kv,
      expected_kv, expected_logits, expected_hidden};
  std::size_t authenticated_bytes = kHeaderBytes;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (sections[index] != expected[index] ||
        !add_size(expected[index], &authenticated_bytes)) {
      return {StatusCode::kIncompatibleArtifact,
              "checkpoint section sizes are incompatible"};
    }
  }
  if (authenticated_bytes + kDigestBytes != file_bytes) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint total byte size is incompatible"};
  }
  std::string calculated;
  Status status =
      internal::sha256_file_prefix(path, authenticated_bytes, &calculated);
  input.seekg(static_cast<std::streamoff>(authenticated_bytes));
  std::array<char, kDigestBytes> stored_digest{};
  input.read(stored_digest.data(), stored_digest.size());
  if (!status.is_ok()) return status;
  if (input.gcount() != static_cast<std::streamsize>(stored_digest.size()) ||
      std::memcmp(stored_digest.data(), calculated.data(), kDigestBytes) != 0) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint payload digest does not match"};
  }
  const SamplerState restored_sampler{bits_float(temperature_bits),
                                      bits_float(top_p_bits), top_k, seed,
                                      rng_state};
  if (!std::isfinite(restored_sampler.temperature) ||
      restored_sampler.temperature < 0.0F ||
      !std::isfinite(restored_sampler.top_p) ||
      restored_sampler.top_p <= 0.0F || restored_sampler.top_p > 1.0F ||
      restored_sampler.top_k > internal::kVocabularySize) {
    return {StatusCode::kIncompatibleArtifact,
            "checkpoint sampler values are invalid"};
  }

  input.clear();
  input.seekg(kHeaderBytes);
  std::vector<std::size_t> restored_tokens(static_cast<std::size_t>(frontier));
  for (std::size_t index = 0; index < restored_tokens.size(); ++index) {
    std::array<unsigned char, 4> encoded{};
    input.read(reinterpret_cast<char*>(encoded.data()), encoded.size());
    std::uint32_t token = 0;
    for (std::size_t byte = 0; byte < encoded.size(); ++byte) {
      token |= static_cast<std::uint32_t>(encoded[byte]) << (byte * 8);
    }
    if (!input || token >= internal::kVocabularySize) {
      return {StatusCode::kIncompatibleArtifact,
              "checkpoint token history is invalid"};
    }
    restored_tokens[index] = token;
  }
  std::vector<unsigned char> buffer(kChunkBytes);
  status = read_device(&input, workspace->gdn_candidate_convolution_,
                       expected_convolution, &buffer);
  if (status.is_ok()) {
    status = read_device(&input, workspace->gdn_candidate_recurrent_,
                         expected_recurrent, &buffer);
  }
  const std::size_t cache_stride = capacity_ * internal::kAttentionKvWidth;
  const std::size_t layer_bytes =
      frontier * internal::kAttentionKvWidth * sizeof(__nv_bfloat16);
  for (std::size_t layer = 0; status.is_ok() && layer < kAttentionLayers;
       ++layer) {
    status = read_device(&input, attention_key_ + layer * cache_stride,
                         layer_bytes, &buffer);
  }
  for (std::size_t layer = 0; status.is_ok() && layer < kAttentionLayers;
       ++layer) {
    status = read_device(&input, attention_value_ + layer * cache_stride,
                         layer_bytes, &buffer);
  }
  if (status.is_ok() && expected_logits != 0) {
    input.read(reinterpret_cast<char*>(workspace->candidate_logits_host_),
               expected_logits);
    if (!input) status = {StatusCode::kIoError, "cannot restore checkpoint logits"};
  }
  if (status.is_ok() && expected_hidden != 0) {
    input.read(reinterpret_cast<char*>(workspace->candidate_hidden_host_),
               expected_hidden);
    if (!input) status = {StatusCode::kIoError, "cannot restore checkpoint hidden"};
  }
  if (status.is_ok()) {
    const cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      status = {StatusCode::kInternal, "cannot synchronize checkpoint restore"};
    }
  }
  if (!status.is_ok()) return status;
  std::swap(gdn_convolution_, workspace->gdn_candidate_convolution_);
  std::swap(gdn_recurrent_, workspace->gdn_candidate_recurrent_);
  std::copy(restored_tokens.begin(), restored_tokens.end(), tokens_);
  if (frontier != 0) {
    std::memcpy(last_logits_, workspace->candidate_logits_host_, expected_logits);
    std::memcpy(last_hidden_, workspace->candidate_hidden_host_, expected_hidden);
  }
  sampler_state_ = restored_sampler;
  frontier_ = static_cast<std::size_t>(frontier);
  return Status::ok();
}

}  // namespace qw38::cuda
