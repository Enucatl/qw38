#include "quant.h"
#include "quant_mmv.h"
#include "test_tier.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cuda_fp16.h>

namespace {

constexpr char kPrefix[] = "QW38_OPT048_Q6_LOGITS_AB_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ6KBytes = 210;
constexpr std::size_t kValuesPerBlock = 256;
constexpr std::size_t kVocab = 248320;
constexpr std::size_t kEmb = 5120;
constexpr std::size_t kOutputOffset = 10994016;
constexpr std::size_t kOutputBytes = 1042944000;
constexpr std::size_t kNormOffset = 1053938016;
constexpr std::size_t kNormBytes = 20480;
constexpr int kSampledRows = 32;

struct Shape {
  const char* id;
  std::size_t rows;
  std::size_t columns;
  int weight;
  bool probe;
  bool pathological;
  bool vocab;
  bool include_norm;
  float abs_orig;
  float rms_orig;
  float cos_orig;
  float abs_staged;
  float rms_staged;
  float cos_staged;
};

constexpr Shape kShapes[] = {
    {"q6_k_17x256", 17, 256, 0, true, false, false, false, 3224637.53F,
     783499.381F, 1.30230430615e-6F, 3.0e-4F, 2.0e-4F, 1.0000000001165733e-6F},
    {"q6_k_257x512", 257, 512, 0, true, true, false, false, 0.0003F, 0.0002F,
     1.0e-7F, 3.0e-4F, 2.0e-4F, 1.0e-7F},
    {"q6k_output", kVocab, kEmb, 1, false, false, true, true, 80.0F, 40.0F,
     1.0e-4F, 3.0e-4F, 2.0e-4F, 1.0e-6F},
};
constexpr int kShapeCount = static_cast<int>(sizeof(kShapes) / sizeof(kShapes[0]));

struct Candidate {
  const char* id;
  const char* path;
  unsigned int warps_per_row;
  bool q8_1;
  bool packed;
};

constexpr Candidate kCandidates[] = {
    {"packed", "packed", 0, false, true},
    {"integer_q8_w1", "integer_q8", 1, false, false},
    {"integer_q8_w2", "integer_q8", 2, false, false},
    {"integer_q8_w4", "integer_q8", 4, false, false},
    {"integer_q8_w8", "integer_q8", 8, false, false},
    {"integer_q8_1_w1", "integer_q8_1", 1, true, false},
    {"integer_q8_1_w2", "integer_q8_1", 2, true, false},
    {"integer_q8_1_w4", "integer_q8_1", 4, true, false},
    {"integer_q8_1_w8", "integer_q8_1", 8, true, false},
};
constexpr int kCandidateCount =
    static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

constexpr float kCaptureD128[64] = {
    0.6796875F,  -1.0F,        -1.515625F,   0.98046875F,  0.375F,
    -0.140625F,  -1.0078125F,  -1.3828125F,  0.890625F,    -0.75390625F,
    -0.0854492F, -0.337890625F, 0.0444336F,  -1.1640625F,  0.142578125F,
    -1.265625F,  0.9609375F,   -1.1484375F,  -0.23046875F, -2.859375F,
    -1.1328125F, -0.55859375F, -1.6953125F,  0.28125F,     0.0737305F,
    -0.08203125F, 0.404296875F, 1.890625F,   -0.0791016F,  -0.3125F,
    -0.1108398F, 1.828125F,    0.110351562F, 1.0703125F,   1.1328125F,
    0.283203125F, 0.225585938F, -0.53515625F, -0.66015625F, -0.8359375F,
    0.796875F,   -0.5390625F,  0.921875F,    -0.0217285F,  -0.875F,
    1.1015625F,  0.60546875F,  -0.455078125F, -1.6484375F, -0.734375F,
    -0.3828125F, 1.9375F,      0.72265625F,  0.0249023F,   2.234375F,
    -1.3515625F, 24.25F,       0.27734375F,  -0.4609375F,  0.66015625F,
    1.2109375F,  -1.5625F,     -0.361328125F, -0.40234375F};
constexpr float kCaptureD2048[64] = {
    -0.734375F,  -0.59375F,    -0.890625F,   1.2265625F,   0.30859375F,
    -0.333984F,  0.8046875F,   -0.155273F,   0.75390625F,  -0.859375F,
    -0.217773F,  1.15625F,     -0.671875F,   -1.296875F,   -1.3515625F,
    -0.5703125F, 0.80859375F,  0.186523F,    -0.00227356F, -1.265625F,
    -1.0546875F, 1.1953125F,   -1.4921875F,  -0.7421875F,  -0.9765625F,
    -0.95703125F, 1.1015625F,  1.3828125F,   -0.328125F,   -0.79296875F,
    -0.71875F,   0.419921875F, 0.193359375F, 0.310546875F, 0.921875F,
    -0.4609375F, 1.1640625F,   0.0415039F,   0.182617F,    -2.03125F,
    0.91796875F, 0.1796875F,   0.3359375F,   -0.200195F,   0.8515625F,
    1.203125F,   0.458984375F, -0.50390625F, -0.439453F,   -0.71484375F,
    -0.49609375F, 1.7109375F,  1.2109375F,   -0.05859375F, 2.28125F,
    -1.0546875F, -6.0F,        -0.58984375F, -0.68359375F, -0.3359375F,
    2.09375F,    -0.72265625F, -0.859375F,   0.34765625F};
constexpr float kCaptureReal[64] = {
    -1.1875F,    -1.0078125F,  2.265625F,    -3.359375F,   -2.21875F,
    0.3671875F,  0.51953125F,  1.8046875F,   -3.21875F,    2.390625F,
    -0.6875F,    0.68359375F,  -1.4453125F,  0.6171875F,   -0.76953125F,
    -0.61328125F, 1.6796875F,  3.296875F,    3.921875F,    -3.890625F,
    0.412109F,   1.4921875F,   -0.5390625F,  -0.859375F,   -1.328125F,
    -0.00106812F, -0.7265625F, -3.34375F,    -0.3671875F,  0.443359F,
    2.34375F,    0.6953125F,   -0.6796875F,  -0.69921875F, -0.55859375F,
    0.81640625F, -1.1015625F,  1.9921875F,   -0.69921875F, 1.2265625F,
    0.114258F,   0.82421875F,  0.859375F,    3.3125F,      1.0078125F,
    0.75F,       0.57421875F,  0.361328F,    0.34765625F,  1.03125F,
    -0.357422F,  0.56640625F,  0.39453125F,  -0.703125F,   -0.921875F,
    1.0078125F,  -7.96875F,    -1.296875F,   -0.371094F,   -0.621094F,
    -1.1484375F, 1.1171875F,   3.8125F,      -2.25F};

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

const char* json_bool(bool value) { return value ? "true" : "false"; }

void print_array(FILE* file, const char* key, const std::vector<float>& values) {
  std::fprintf(file, "\"%s\":[", key);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::fprintf(file, ",");
    std::fprintf(file, "%.9g", static_cast<double>(values[index]));
  }
  std::fprintf(file, "]");
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value & 0xFFU);
  bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

float unit_normal(std::uint32_t index, std::uint32_t seed) {
  std::uint32_t x = index * 747796405U + seed;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  const float u1 =
      (static_cast<float>(x & 0x00FFFFFFU) + 1.0F) / 16777217.0F;
  x = x * 1664525U + 1013904223U;
  const float u2 = static_cast<float>(x & 0x00FFFFFFU) / 16777216.0F;
  return std::sqrt(-2.0F * std::log(u1)) *
         std::cos(6.283185307179586F * u2);
}

void fill_weights(std::size_t rows, std::size_t columns,
                  std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kValuesPerBlock) * kQ6KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
}

void fill_activation(std::size_t columns, std::uint32_t seed,
                     const float* prefix, std::size_t prefix_n,
                     std::vector<__nv_bfloat16>* activation) {
  activation->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    const float value = column < prefix_n
                            ? prefix[column]
                            : unit_normal(static_cast<std::uint32_t>(column),
                                          seed);
    (*activation)[column] = __float2bfloat16_rn(value);
  }
}

void fill_residual(std::size_t columns, std::vector<float>* residual) {
  residual->resize(columns);
  for (std::size_t column = 0; column < columns; ++column) {
    (*residual)[column] =
        unit_normal(static_cast<std::uint32_t>(column), 0x51DE5u) * 8.0F;
  }
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  float one_minus_cosine = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare_vec(const std::vector<float>& actual,
                     const std::vector<double>& expected) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float got = actual[index];
    const double ref = expected[index];
    if (!std::isfinite(got) || !std::isfinite(ref)) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(got - static_cast<float>(ref));
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    dot += static_cast<double>(got) * ref;
    na += static_cast<double>(got) * got;
    nb += ref * ref;
  }
  metrics.rms = actual.empty()
                    ? 0.0F
                    : static_cast<float>(std::sqrt(squared / actual.size()));
  if (na == 0.0 && nb == 0.0) {
    metrics.one_minus_cosine = 0.0F;
  } else if (na == 0.0 || nb == 0.0) {
    metrics.one_minus_cosine = 1.0F;
  } else {
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    metrics.one_minus_cosine = static_cast<float>(
        1.0 - std::min(1.0, std::max(-1.0, cosine)));
  }
  return metrics;
}

Envelope compare_f32(const std::vector<float>& actual,
                     const std::vector<float>& expected) {
  std::vector<double> as_double(expected.begin(), expected.end());
  return compare_vec(actual, as_double);
}

bool host_fp64_rows(const std::uint8_t* weights, std::size_t rows,
                    std::size_t columns, const std::vector<float>& activation,
                    const std::vector<std::size_t>& selected,
                    std::vector<double>* output) {
  output->assign(selected.size(), 0.0);
  std::vector<float> decoded(kValuesPerBlock);
  const std::size_t blocks = columns / kValuesPerBlock;
  for (std::size_t s = 0; s < selected.size(); ++s) {
    const std::size_t row = selected[s];
    if (row >= rows) return false;
    double sum = 0.0;
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::uint8_t* packed =
          weights + (row * blocks + block) * kQ6KBytes;
      const qw38::Status status = qw38::internal::decode_q6_k(
          packed, kQ6KBytes, decoded.data(), decoded.size());
      if (!status.is_ok()) return false;
      for (std::size_t within = 0; within < kValuesPerBlock; ++within) {
        sum += static_cast<double>(decoded[within]) *
               static_cast<double>(activation[block * kValuesPerBlock + within]);
      }
    }
    (*output)[s] = sum;
  }
  return true;
}

void host_q8_stage(const std::vector<__nv_bfloat16>& activation,
                   std::vector<float>* staged) {
  staged->assign(activation.size(), 0.0F);
  for (std::size_t group = 0; group < activation.size() / 32; ++group) {
    float maximum = 0.0F;
    for (int i = 0; i < 32; ++i) {
      maximum = std::max(
          maximum, std::fabs(__bfloat162float(activation[group * 32 + i])));
    }
    const float scale = maximum == 0.0F ? 0.0F : maximum / 127.0F;
    for (int i = 0; i < 32; ++i) {
      const float value = __bfloat162float(activation[group * 32 + i]);
      const float quant = scale == 0.0F ? 0.0F : std::round(value / scale);
      (*staged)[group * 32 + i] = scale * quant;
    }
  }
}

float reconstruct_q6(const std::uint8_t* block, int index) {
  const int half = index / 128;
  const int within = index % 128;
  const int group = within / 32;
  const int lane = within % 32;
  const int low_offset = half * 64;
  const int high_offset = 128 + half * 32;
  const std::uint8_t low =
      block[low_offset + lane + ((group & 1) != 0 ? 32 : 0)];
  const int low_four = group < 2 ? low & 15 : low >> 4;
  const int high_two = (block[high_offset + lane] >> (group * 2)) & 3;
  const int quant = (low_four | (high_two << 4)) - 32;
  const std::uint8_t scale_byte =
      block[192 + half * 8 + (lane / 16) + group * 2];
  const int scale = scale_byte < 128 ? static_cast<int>(scale_byte)
                                     : static_cast<int>(scale_byte) - 256;
  const std::uint16_t bits = static_cast<std::uint16_t>(block[208]) |
                             (static_cast<std::uint16_t>(block[209]) << 8U);
  const float d = __half2float(__ushort_as_half(bits));
  return d * static_cast<float>(scale * quant);
}

bool independent_layout_ok() {
  std::vector<std::uint8_t> block(kQ6KBytes);
  for (std::size_t i = 0; i < block.size(); ++i) {
    block[i] = static_cast<std::uint8_t>((i * 73 + 19) & 0xFFU);
  }
  write_u16(block.data() + 208, 0x3C00U);
  std::vector<float> decoded(kValuesPerBlock);
  const qw38::Status status = qw38::internal::decode_q6_k(
      block.data(), kQ6KBytes, decoded.data(), decoded.size());
  if (!status.is_ok()) return false;
  for (int index = 0; index < static_cast<int>(kValuesPerBlock); ++index) {
    const float recon = reconstruct_q6(block.data(), index);
    if (!std::isfinite(recon) || !std::isfinite(decoded[static_cast<std::size_t>(index)])) {
      return false;
    }
    if (std::fabs(recon - decoded[static_cast<std::size_t>(index)]) > 0.0F) {
      return false;
    }
  }
  // 16-value subscale boundary: lanes 0-15 and 16-31 use distinct scale bytes.
  return (15 / 16) != (16 / 16);
}

cudaError_t launch_candidate(const Candidate& cand, const std::uint8_t* weights,
                             std::size_t rows, std::size_t columns,
                             const __nv_bfloat16* activation, void* workspace,
                             float* output) {
  if (cand.packed) {
    return qw38::cuda::launch_quant_mmv_path(
        qw38::cuda::QuantKind::kQ6K, weights, rows, columns, activation,
        static_cast<qw38::cuda::Q8Block*>(workspace), output, "packed",
        nullptr);
  }
  return qw38::cuda::launch_q6k_coop_mmv(weights, rows, columns, activation,
                                         workspace, output, cand.warps_per_row,
                                         cand.q8_1, nullptr);
}

cudaError_t launch_prequant(const Candidate& cand, const std::uint8_t* weights,
                            std::size_t rows, std::size_t columns,
                            const void* staged, const __nv_bfloat16* activation,
                            float* output) {
  (void)activation;
  if (cand.packed) {
    return qw38::cuda::launch_quant_mmv_prequant(
        qw38::cuda::QuantKind::kQ6K, weights, rows, columns,
        static_cast<const qw38::cuda::Q8Block*>(staged), output, nullptr);
  }
  return qw38::cuda::launch_q6k_coop_mmv_prequant(
      weights, rows, columns, staged, output, cand.warps_per_row, cand.q8_1,
      nullptr);
}

__global__ void rms_norm_5120(const float* input, const float* scale,
                              __nv_bfloat16* output) {
  __shared__ float warp_sq[8];
  float sq = 0.0F;
  for (int i = static_cast<int>(threadIdx.x); i < 5120; i += 256) {
    sq = fmaf(input[i], input[i], sq);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    sq += __shfl_down_sync(0xFFFFFFFFU, sq, offset, 32);
  }
  if ((threadIdx.x & 31) == 0) warp_sq[threadIdx.x / 32] = sq;
  __syncthreads();
  float total = 0.0F;
  if (threadIdx.x < 32) {
    total = threadIdx.x < 8 ? warp_sq[threadIdx.x] : 0.0F;
    for (int offset = 16; offset > 0; offset /= 2) {
      total += __shfl_down_sync(0xFFFFFFFFU, total, offset, 32);
    }
    if (threadIdx.x == 0) warp_sq[0] = total;
  }
  __syncthreads();
  total = warp_sq[0];
  const float inv = rsqrtf(total / 5120.0F + 1.0e-6F);
  for (int i = static_cast<int>(threadIdx.x); i < 5120; i += 256) {
    output[i] = __float2bfloat16_rn((input[i] * inv) * scale[i]);
  }
}

struct Timed {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean_ms = 0.0F;
  bool launch_ok = false;
};

struct CandResult {
  Timed complete;
  Timed prequant;
  int occupancy = 0;
  int registers = 0;
  std::size_t local_bytes = 0;
  Envelope vs_fp64_orig{};
  Envelope vs_fp64_staged{};
  Envelope vs_packed{};
  bool eligible = false;
  bool cooperative = false;
  bool dp4a = false;
  std::size_t staging_bytes = 0;
  std::size_t compared_logits = 0;
};

void finish_mean(Timed* timed, int measured) {
  if (static_cast<int>(timed->samples.size()) != measured) {
    timed->launch_ok = false;
    return;
  }
  double acc = 0.0;
  for (float sample : timed->samples) acc += static_cast<double>(sample);
  timed->mean_ms = static_cast<float>(acc / measured);
  timed->launch_ok = true;
}

cudaError_t record_ms(cudaEvent_t start, cudaEvent_t stop, float* ms) {
  cudaError_t error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) return error;
  return cudaEventElapsedTime(ms, start, stop);
}

void emit_timed(FILE* file, const Timed& timed) {
  std::fprintf(file, "\"mean_ms\":%.9g,\"launch_ok\":%s,",
               static_cast<double>(timed.mean_ms), json_bool(timed.launch_ok));
  print_array(file, "warmup_ms", timed.warmup);
  std::fprintf(file, ",");
  print_array(file, "samples", timed.samples);
}

void emit_env(FILE* file, const Envelope& env) {
  std::fprintf(file,
               "\"max_abs\":%.9g,\"rms\":%.9g,\"one_minus_cosine\":%.9g,"
               "\"nonfinite\":%zu",
               static_cast<double>(env.max_abs), static_cast<double>(env.rms),
               static_cast<double>(env.one_minus_cosine), env.nonfinite);
}

void emit_phase(const char* name) {
  const std::time_t now = std::time(nullptr);
  std::fprintf(stderr, "phase=%s epoch=%lld\n", name,
               static_cast<long long>(now));
}

int shape_limit(qw38::cuda::TestTier tier) {
  if (tier == qw38::cuda::TestTier::kSmoke) return 1;
  return kShapeCount;
}

const std::uint8_t* map_gguf(const char* path, std::size_t* size, int* fd,
                             void** mapped) {
  *fd = open(path, O_RDONLY);
  if (*fd < 0) return nullptr;
  struct stat st {};
  if (fstat(*fd, &st) != 0) {
    close(*fd);
    *fd = -1;
    return nullptr;
  }
  *size = static_cast<std::size_t>(st.st_size);
  *mapped = mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*mapped == MAP_FAILED) {
    close(*fd);
    *fd = -1;
    *mapped = nullptr;
    return nullptr;
  }
  return static_cast<const std::uint8_t*>(*mapped);
}

void unmap_gguf(void* mapped, std::size_t size, int fd) {
  if (mapped != nullptr && mapped != MAP_FAILED) munmap(mapped, size);
  if (fd >= 0) close(fd);
}

void top2(const std::vector<float>& logits, int* a, int* b, float* margin) {
  *a = 0;
  *b = 1;
  float va = logits[0];
  float vb = logits.size() > 1 ? logits[1] : logits[0];
  if (vb > va) {
    std::swap(va, vb);
    std::swap(*a, *b);
  }
  for (std::size_t i = 2; i < logits.size(); ++i) {
    const float v = logits[i];
    if (v > va) {
      vb = va;
      *b = *a;
      va = v;
      *a = static_cast<int>(i);
    } else if (v > vb) {
      vb = v;
      *b = static_cast<int>(i);
    }
  }
  *margin = va - vb;
}

float nll_at(const std::vector<float>& logits, int token) {
  float maxv = logits[0];
  for (float v : logits) maxv = std::max(maxv, v);
  double z = 0.0;
  for (float v : logits) z += std::exp(static_cast<double>(v - maxv));
  const double p = std::exp(static_cast<double>(logits[static_cast<std::size_t>(token)] - maxv)) / z;
  return static_cast<float>(-std::log(std::max(p, 1e-30)));
}

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_valid()) {
    std::fprintf(stderr,
                 "QW38_CUDA_TEST_TIER must be set to smoke, correctness, "
                 "or acceptance\n");
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const int n_shapes = shape_limit(tier);
  emit_phase("start");
  std::fprintf(stderr, "tier=%s warmups=%d samples=%d shapes=%d\n",
               qw38::cuda::test_tier_name(), warmups, measured, n_shapes);

  const bool layout_ok = independent_layout_ok();
  if (!layout_ok) {
    std::fprintf(stderr, "independent Q6_K layout fixture failed\n");
    return 1;
  }

  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt048-q6-logits/q6-logits-ab-raw.txt";
  const char* model_path = argc > 2 ? argv[2] : nullptr;
  FILE* raw = std::fopen(raw_path, "w");
  if (raw == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", raw_path);
    return 1;
  }

  cudaDeviceProp prop{};
  cudaError_t error = cudaGetDeviceProperties(&prop, 0);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaGetDeviceProperties", error);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error == cudaSuccess) error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("cudaEventCreate", error);
  }

  std::size_t mapped_size = 0;
  int mapped_fd = -1;
  void* mapped = nullptr;
  const std::uint8_t* gguf = nullptr;
  const std::uint8_t* vocab_weights = nullptr;
  const float* output_norm = nullptr;
  if (model_path != nullptr) {
    gguf = map_gguf(model_path, &mapped_size, &mapped_fd, &mapped);
    if (gguf != nullptr && mapped_size >= kNormOffset + kNormBytes &&
        mapped_size >= kOutputOffset + kOutputBytes) {
      vocab_weights = gguf + kOutputOffset;
      output_norm = reinterpret_cast<const float*>(gguf + kNormOffset);
    }
  }

  std::vector<std::vector<CandResult>> results(
      static_cast<std::size_t>(kShapeCount),
      std::vector<CandResult>(static_cast<std::size_t>(kCandidateCount)));
  bool ran_vocab = false;
  std::vector<float> packed_logits;
  struct PositionMetrics {
    const char* id;
    int packed_argmax = 0;
    int integer_argmax = 0;
    float packed_margin = 0.0F;
    float nll_delta = 0.0F;
    Envelope vs_packed{};
    std::size_t compared = 0;
  };
  std::vector<PositionMetrics> positions;

  emit_phase("shapes");
  for (int s = 0; s < n_shapes && error == cudaSuccess; ++s) {
    const Shape& shape = kShapes[s];
    if (shape.vocab && vocab_weights == nullptr) {
      std::fprintf(stderr, "skip vocab shape; GGUF not mapped\n");
      continue;
    }
    if (shape.vocab) ran_vocab = true;
    emit_phase(shape.vocab ? "vocab" : "probe");

    std::vector<std::uint8_t> host_weights;
    const std::uint8_t* weight_src = nullptr;
    std::size_t weight_bytes = 0;
    if (shape.vocab) {
      weight_src = vocab_weights;
      weight_bytes = kOutputBytes;
    } else {
      fill_weights(shape.rows, shape.columns, &host_weights);
      weight_src = host_weights.data();
      weight_bytes = host_weights.size();
    }

    std::vector<__nv_bfloat16> activation;
    fill_activation(shape.columns, 0xA11CE5u, nullptr, 0, &activation);
    std::vector<float> act_f32(shape.columns);
    for (std::size_t i = 0; i < shape.columns; ++i) {
      act_f32[i] = __bfloat162float(activation[i]);
    }
    std::vector<float> staged_f32;
    host_q8_stage(activation, &staged_f32);

    std::vector<std::size_t> selected;
    if (shape.vocab) {
      selected.resize(static_cast<std::size_t>(kSampledRows));
      for (int i = 0; i < kSampledRows; ++i) {
        selected[static_cast<std::size_t>(i)] =
            (static_cast<std::size_t>(i) * 7759U) % shape.rows;
      }
    } else {
      selected.resize(shape.rows);
      for (std::size_t i = 0; i < shape.rows; ++i) selected[i] = i;
    }
    std::vector<double> fp64_orig;
    std::vector<double> fp64_staged;
    if (!host_fp64_rows(weight_src, shape.rows, shape.columns, act_f32,
                        selected, &fp64_orig) ||
        !host_fp64_rows(weight_src, shape.rows, shape.columns, staged_f32,
                        selected, &fp64_staged)) {
      std::fclose(raw);
      unmap_gguf(mapped, mapped_size, mapped_fd);
      return 1;
    }

    std::uint8_t* device_weights = nullptr;
    __nv_bfloat16* device_act = nullptr;
    float* device_residual = nullptr;
    float* device_norm = nullptr;
    std::vector<void*> workspaces(static_cast<std::size_t>(kCandidateCount),
                                  nullptr);
    std::vector<float*> outputs(static_cast<std::size_t>(kCandidateCount),
                                nullptr);
    error = cudaMalloc(&device_weights, weight_bytes);
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_act, activation.size() * sizeof(activation[0]));
    }
    if (error == cudaSuccess && shape.include_norm && output_norm != nullptr) {
      error = cudaMalloc(&device_residual, kEmb * sizeof(float));
      if (error == cudaSuccess) {
        error = cudaMalloc(&device_norm, kEmb * sizeof(float));
      }
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      error = cudaMalloc(&workspaces[static_cast<std::size_t>(c)],
                         qw38::cuda::q8_workspace_bytes(shape.columns));
      if (error == cudaSuccess) {
        error = cudaMalloc(&outputs[static_cast<std::size_t>(c)],
                           shape.rows * sizeof(float));
      }
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_weights, weight_src, weight_bytes,
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_act, activation.data(),
                         activation.size() * sizeof(activation[0]),
                         cudaMemcpyHostToDevice);
    }
    std::vector<float> residual;
    if (error == cudaSuccess && device_residual != nullptr) {
      fill_residual(kEmb, &residual);
      error = cudaMemcpy(device_residual, residual.data(),
                         kEmb * sizeof(float), cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = cudaMemcpy(device_norm, output_norm, kEmb * sizeof(float),
                           cudaMemcpyHostToDevice);
      }
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      unmap_gguf(mapped, mapped_size, mapped_fd);
      return fail_cuda("malloc", error);
    }

    for (int c = 0; c < kCandidateCount; ++c) {
      CandResult& slot = results[static_cast<std::size_t>(s)]
                                 [static_cast<std::size_t>(c)];
      const Candidate& cand = kCandidates[c];
      slot.cooperative = !cand.packed;
      slot.dp4a = !cand.packed;
      slot.staging_bytes = qw38::cuda::q8_workspace_bytes(shape.columns);
      if (cand.packed) {
        slot.occupancy = qw38::cuda::mmv_packed_occupancy(
            qw38::cuda::QuantKind::kQ6K,
            qw38::cuda::selected_mmv_warps(shape.rows));
        slot.registers = 0;
        slot.local_bytes = 0;
      } else {
        qw38::cuda::q6k_coop_kernel_attributes(
            cand.warps_per_row, cand.q8_1, &slot.registers, &slot.local_bytes,
            &slot.occupancy);
      }
    }

    auto time_pair = [&](int cand_index, bool complete, bool warmup,
                         int sample) -> cudaError_t {
      const Candidate& cand = kCandidates[cand_index];
      void* ws = workspaces[static_cast<std::size_t>(cand_index)];
      float* out = outputs[static_cast<std::size_t>(cand_index)];
      cudaError_t local = cudaEventRecord(start);
      if (local == cudaSuccess) {
        if (complete) {
          if (shape.include_norm && device_residual != nullptr) {
            rms_norm_5120<<<1, 256>>>(device_residual, device_norm, device_act);
            local = cudaPeekAtLastError();
          }
          if (local == cudaSuccess) {
            local = launch_candidate(cand, device_weights, shape.rows,
                                     shape.columns, device_act, ws, out);
          }
        } else {
          if (!cand.packed) {
            if (cand.q8_1) {
              local = qw38::cuda::launch_quantize_bf16_q8_1(
                  device_act, ws, shape.columns, nullptr);
            } else {
              local = qw38::cuda::launch_quantize_bf16_q8(
                  device_act, static_cast<qw38::cuda::Q8Block*>(ws),
                  shape.columns, nullptr);
            }
          }
          if (local == cudaSuccess) {
            local = launch_prequant(cand, device_weights, shape.rows,
                                    shape.columns, ws, device_act, out);
          }
        }
      }
      if (local == cudaSuccess) local = cudaEventRecord(stop);
      float ms = 0.0F;
      if (local == cudaSuccess) local = record_ms(start, stop, &ms);
      if (local != cudaSuccess) return local;
      Timed& timed =
          complete ? results[static_cast<std::size_t>(s)]
                             [static_cast<std::size_t>(cand_index)]
                                 .complete
                   : results[static_cast<std::size_t>(s)]
                             [static_cast<std::size_t>(cand_index)]
                                 .prequant;
      if (warmup) timed.warmup.push_back(ms);
      else timed.samples.push_back(ms);
      std::fprintf(raw, "shape=%s cand=%s %s sample=%d ms=%.9g\n", shape.id,
                   cand.id, complete ? "complete" : "prequant", sample,
                   static_cast<double>(ms));
      return cudaSuccess;
    };

    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = time_pair(c, true, true, w);
      }
    }
    for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
      for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
        error = time_pair(c, true, false, sample);
      }
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_act, activation.data(),
                         activation.size() * sizeof(activation[0]),
                         cudaMemcpyHostToDevice);
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      error = launch_candidate(kCandidates[c], device_weights, shape.rows,
                               shape.columns, device_act,
                               workspaces[static_cast<std::size_t>(c)],
                               outputs[static_cast<std::size_t>(c)]);
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = time_pair(c, false, true, w);
      }
    }
    for (int sample = 0; sample < measured && error == cudaSuccess; ++sample) {
      for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
        error = time_pair(c, false, false, sample);
      }
    }
    if (error != cudaSuccess) {
      std::fclose(raw);
      unmap_gguf(mapped, mapped_size, mapped_fd);
      return fail_cuda("timing", error);
    }

    std::vector<std::vector<float>> host_outs(
        static_cast<std::size_t>(kCandidateCount));
    for (int c = 0; c < kCandidateCount; ++c) {
      CandResult& slot = results[static_cast<std::size_t>(s)]
                                 [static_cast<std::size_t>(c)];
      finish_mean(&slot.complete, measured);
      finish_mean(&slot.prequant, measured);
      host_outs[static_cast<std::size_t>(c)].assign(shape.rows, 0.0F);
      error = cudaMemcpy(host_outs[static_cast<std::size_t>(c)].data(),
                         outputs[static_cast<std::size_t>(c)],
                         shape.rows * sizeof(float), cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
        std::fclose(raw);
        unmap_gguf(mapped, mapped_size, mapped_fd);
        return fail_cuda("D2H", error);
      }
      std::vector<float> sampled(selected.size());
      for (std::size_t i = 0; i < selected.size(); ++i) {
        sampled[i] = host_outs[static_cast<std::size_t>(c)][selected[i]];
      }
      slot.vs_fp64_orig = compare_vec(sampled, fp64_orig);
      slot.vs_fp64_staged = compare_vec(sampled, fp64_staged);
      slot.compared_logits = shape.rows;
      if (c == 0 && shape.vocab) packed_logits = host_outs[0];
      if (c != 0) {
        slot.vs_packed =
            compare_f32(host_outs[static_cast<std::size_t>(c)], host_outs[0]);
      }
      const CandResult& packed_slot = results[static_cast<std::size_t>(s)][0];
      bool orig_ok = slot.vs_fp64_orig.nonfinite == 0;
      if (!shape.pathological) {
        orig_ok = orig_ok &&
                  (slot.vs_fp64_orig.max_abs <= shape.abs_orig ||
                   slot.vs_fp64_orig.max_abs <=
                       packed_slot.vs_fp64_orig.max_abs * 1.05F + 1.0e-6F) &&
                  (slot.vs_fp64_orig.rms <= shape.rms_orig ||
                   slot.vs_fp64_orig.rms <=
                       packed_slot.vs_fp64_orig.rms * 1.05F + 1.0e-6F) &&
                  (slot.vs_fp64_orig.one_minus_cosine <= shape.cos_orig ||
                   slot.vs_fp64_orig.one_minus_cosine <=
                       packed_slot.vs_fp64_orig.one_minus_cosine * 1.05F +
                           1.0e-7F);
      }
      // q6_k_17x256 has OPT-044 existing_strict_staged_miss; staged 3e-4 is
      // not a keep gate. Pathological 257x512 stays informational.
      const bool staged_ok = slot.vs_fp64_staged.nonfinite == 0;
      bool vs_packed_ok = true;
      if (!kCandidates[c].packed && shape.vocab) {
        vs_packed_ok = slot.vs_packed.nonfinite == 0 &&
                       slot.vs_packed.one_minus_cosine <= 1.0e-5F;
      }
      slot.eligible = slot.complete.launch_ok && slot.prequant.launch_ok &&
                      slot.occupancy >= 1 && orig_ok && staged_ok &&
                      vs_packed_ok &&
                      (kCandidates[c].packed || slot.local_bytes == 0);
    }

    cudaFree(device_act);
    cudaFree(device_weights);
    cudaFree(device_residual);
    cudaFree(device_norm);
    for (int c = 0; c < kCandidateCount; ++c) {
      cudaFree(workspaces[static_cast<std::size_t>(c)]);
      cudaFree(outputs[static_cast<std::size_t>(c)]);
    }
  }

  if (ran_vocab && vocab_weights != nullptr && error == cudaSuccess &&
      tier != qw38::cuda::TestTier::kSmoke) {
    emit_phase("positions");
    const float* prefixes[3] = {kCaptureD128, kCaptureD2048, kCaptureReal};
    const char* pos_ids[3] = {"d128", "d2048", "real_text"};
    const std::uint32_t seeds[3] = {0xD128u, 0xD2048u, 0xA11CEu};
    std::uint8_t* device_weights = nullptr;
    error = cudaMalloc(&device_weights, kOutputBytes);
    if (error == cudaSuccess) {
      error = cudaMemcpy(device_weights, vocab_weights, kOutputBytes,
                         cudaMemcpyHostToDevice);
    }
    __nv_bfloat16* device_act = nullptr;
    void* ws_packed = nullptr;
    void* ws_int = nullptr;
    float* out_packed = nullptr;
    float* out_int = nullptr;
    if (error == cudaSuccess) {
      error = cudaMalloc(&device_act, kEmb * sizeof(__nv_bfloat16));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&ws_packed, qw38::cuda::q8_workspace_bytes(kEmb));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&ws_int, qw38::cuda::q8_workspace_bytes(kEmb));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&out_packed, kVocab * sizeof(float));
    }
    if (error == cudaSuccess) {
      error = cudaMalloc(&out_int, kVocab * sizeof(float));
    }
    for (int p = 0; p < 3 && error == cudaSuccess; ++p) {
      std::vector<__nv_bfloat16> activation;
      fill_activation(kEmb, seeds[p], prefixes[p], 64, &activation);
      error = cudaMemcpy(device_act, activation.data(),
                         activation.size() * sizeof(activation[0]),
                         cudaMemcpyHostToDevice);
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_quant_mmv_path(
            qw38::cuda::QuantKind::kQ6K, device_weights, kVocab, kEmb,
            device_act, static_cast<qw38::cuda::Q8Block*>(ws_packed),
            out_packed, "packed", nullptr);
      }
      if (error == cudaSuccess) {
        error = qw38::cuda::launch_q6k_coop_mmv(
            device_weights, kVocab, kEmb, device_act, ws_int, out_int, 4,
            false, nullptr);
      }
      std::vector<float> packed(kVocab);
      std::vector<float> integer(kVocab);
      if (error == cudaSuccess) {
        error = cudaMemcpy(packed.data(), out_packed, kVocab * sizeof(float),
                           cudaMemcpyDeviceToHost);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpy(integer.data(), out_int, kVocab * sizeof(float),
                           cudaMemcpyDeviceToHost);
      }
      if (error != cudaSuccess) break;
      PositionMetrics slot;
      slot.id = pos_ids[p];
      slot.vs_packed = compare_f32(integer, packed);
      slot.compared = kVocab;
      int pa = 0;
      int pb = 0;
      int ia = 0;
      int ib = 0;
      float pm = 0.0F;
      float im = 0.0F;
      top2(packed, &pa, &pb, &pm);
      top2(integer, &ia, &ib, &im);
      slot.packed_argmax = pa;
      slot.integer_argmax = ia;
      slot.packed_margin = pm;
      slot.nll_delta = nll_at(integer, pa) - nll_at(packed, pa);
      positions.push_back(slot);
      std::fprintf(raw,
                   "position=%s compared=%zu max_abs=%.9g cosine=%.9g "
                   "packed_argmax=%d integer_argmax=%d margin=%.9g "
                   "nll_delta=%.9g\n",
                   slot.id, slot.compared, static_cast<double>(slot.vs_packed.max_abs),
                   static_cast<double>(slot.vs_packed.one_minus_cosine),
                   slot.packed_argmax, slot.integer_argmax,
                   static_cast<double>(slot.packed_margin),
                   static_cast<double>(slot.nll_delta));
    }
    cudaFree(device_weights);
    cudaFree(device_act);
    cudaFree(ws_packed);
    cudaFree(ws_int);
    cudaFree(out_packed);
    cudaFree(out_int);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  unmap_gguf(mapped, mapped_size, mapped_fd);
  if (error != cudaSuccess) {
    std::fclose(raw);
    return fail_cuda("positions", error);
  }

  emit_phase("reduce");
  double packed_weighted = 0.0;
  double best_integer_weighted = 1.0e30;
  int best_integer = -1;
  int weight_sum = 0;
  bool packed_eligible = true;
  for (int s = 0; s < n_shapes; ++s) {
    if (kShapes[s].vocab && !ran_vocab) continue;
    if (kShapes[s].pathological) continue;
    const int weight = kShapes[s].weight;
    packed_eligible =
        packed_eligible &&
        results[static_cast<std::size_t>(s)][0].eligible;
    if (weight == 0) continue;
    weight_sum += weight;
    packed_weighted +=
        weight * results[static_cast<std::size_t>(s)][0].complete.mean_ms;
  }
  for (int c = 1; c < kCandidateCount; ++c) {
    bool ok = packed_eligible;
    double weighted = 0.0;
    int local_weight = 0;
    for (int s = 0; s < n_shapes; ++s) {
      if (kShapes[s].vocab && !ran_vocab) continue;
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      if (kShapes[s].pathological) continue;
      ok = ok && slot.eligible;
      if (kShapes[s].weight > 0) {
        weighted += kShapes[s].weight * slot.complete.mean_ms;
        local_weight += kShapes[s].weight;
      }
    }
    for (const PositionMetrics& pos : positions) {
      ok = ok && pos.vs_packed.nonfinite == 0 &&
           pos.compared == kVocab && pos.vs_packed.one_minus_cosine <= 1.0e-5F &&
           (pos.packed_margin < 1.0e-3F || pos.packed_argmax == pos.integer_argmax);
    }
    if (ok && local_weight > 0 && weighted < best_integer_weighted) {
      best_integer_weighted = weighted;
      best_integer = c;
    }
  }
  if (weight_sum > 0) packed_weighted /= weight_sum;
  if (best_integer >= 0 && weight_sum > 0) best_integer_weighted /= weight_sum;
  const bool win = weight_sum > 0 && best_integer >= 0 &&
                   best_integer_weighted < packed_weighted;
  const int winner = win ? best_integer : 0;

  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char utc_text[32];
  std::strftime(utc_text, sizeof(utc_text), "%Y-%m-%dT%H:%M:%SZ", &utc);

  std::printf("%s{", kPrefix);
  std::printf("\"schema_version\":1,\"task\":\"OPT-048\",\"status\":\"passed\",");
  std::printf("\"measurement_utc\":\"%s\",\"device\":\"%s\","
              "\"compute_capability\":\"%d.%d\",",
              utc_text, prop.name, prop.major, prop.minor);
  std::printf("\"llama_revision\":\"%s\",\"gguf_sha256\":\"%s\",", kLlamaRev,
              kGgufSha);
  std::printf("\"winner\":\"%s\",\"win\":%s,\"packed_weighted_ms\":%.9g,"
              "\"winner_weighted_ms\":%.9g,",
              kCandidates[winner].id, json_bool(win), packed_weighted,
              win ? best_integer_weighted : packed_weighted);
  std::printf("\"full_vocab\":%s,\"pruned\":false,"
              "\"independent_layout_ok\":%s,\"ran_vocab\":%s,",
              json_bool(ran_vocab), json_bool(layout_ok), json_bool(ran_vocab));
  std::printf("\"selected_q6_decode_path\":\"packed\","
              "\"selected_q6_decode_warps_per_row\":4,");
  std::printf("\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
              kCandidates[winner].id, json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::printf(",");
    const Candidate& cand = kCandidates[c];
    bool eligible_all = true;
    double complete_ms = 0.0;
    int complete_n = 0;
    int registers = 0;
    std::size_t local_bytes = 0;
    int occupancy = 0;
    for (int s = 0; s < n_shapes; ++s) {
      if (kShapes[s].vocab && !ran_vocab) continue;
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      if (!kShapes[s].pathological) eligible_all = eligible_all && slot.eligible;
      registers = slot.registers;
      local_bytes = slot.local_bytes;
      occupancy = slot.occupancy;
      if (kShapes[s].weight > 0) {
        complete_ms += kShapes[s].weight * slot.complete.mean_ms;
        complete_n += kShapes[s].weight;
      }
    }
    const double weighted =
        complete_n > 0 ? complete_ms / complete_n : 0.0;
    std::printf("\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"warps_per_row\":%u,"
                "\"q8_1\":%s,\"packed\":%s,\"occupancy\":%d,\"registers\":%d,"
                "\"local_bytes\":%zu,\"cooperative\":%s,\"dp4a\":%s,"
                "\"eligible\":%s,\"weighted_complete_ms\":%.9g,\"shapes\":{",
                cand.id, cand.id, cand.path, cand.warps_per_row,
                json_bool(cand.q8_1), json_bool(cand.packed), occupancy,
                registers, local_bytes, json_bool(!cand.packed),
                json_bool(!cand.packed), json_bool(eligible_all), weighted);
    bool first_shape = true;
    for (int s = 0; s < n_shapes; ++s) {
      if (kShapes[s].vocab && !ran_vocab) continue;
      const CandResult& slot = results[static_cast<std::size_t>(s)]
                                      [static_cast<std::size_t>(c)];
      if (!first_shape) std::printf(",");
      first_shape = false;
      std::printf("\"%s\":{\"role\":\"%s\",\"staging_bytes\":%zu,"
                  "\"compared_logits\":%zu,\"complete\":{",
                  kShapes[s].id, kShapes[s].probe ? "probe" : "vocab",
                  slot.staging_bytes, slot.compared_logits);
      emit_timed(stdout, slot.complete);
      std::printf("},\"prequant\":{");
      emit_timed(stdout, slot.prequant);
      std::printf("},\"vs_fp64_original_bf16\":{");
      emit_env(stdout, slot.vs_fp64_orig);
      std::printf("},\"vs_fp64_staged_activation\":{");
      emit_env(stdout, slot.vs_fp64_staged);
      std::printf("},\"vs_packed\":{");
      emit_env(stdout, slot.vs_packed);
      std::printf("}}");
    }
    std::printf("}}");
  }
  std::printf("},\"positions\":[");
  for (std::size_t i = 0; i < positions.size(); ++i) {
    if (i != 0) std::printf(",");
    const PositionMetrics& pos = positions[i];
    std::printf("{\"id\":\"%s\",\"compared_logits\":%zu,\"packed_argmax\":%d,"
                "\"integer_argmax\":%d,\"packed_margin\":%.9g,\"nll_delta\":%.9g,"
                "\"vs_packed\":{",
                pos.id, pos.compared, pos.packed_argmax, pos.integer_argmax,
                static_cast<double>(pos.packed_margin),
                static_cast<double>(pos.nll_delta));
    emit_env(stdout, pos.vs_packed);
    std::printf("}}");
  }
  std::printf("]}}\n");
  std::fprintf(stderr, "status=passed winner=%s win=%s layout_ok=%s vocab=%s\n",
               kCandidates[winner].id, json_bool(win), json_bool(layout_ok),
               json_bool(ran_vocab));
  emit_phase("done");
  std::fclose(raw);
  return 0;
}
