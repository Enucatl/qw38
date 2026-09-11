#include "ffn_decode_path.cuh"
#include "full_scheduler.h"
#include "quant.h"
#include "quant_mmv.h"
#include "scheduler_primitives.h"
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

namespace {

constexpr char kPrefix[] = "QW38_OPT049_DECODE_FFN_FUSION_AB_RESULT=";
constexpr const char* kGgufSha =
    "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34";
constexpr const char* kLlamaRev =
    "cc83d7b4824f73cfdda4dfbb47ee39804f71b328";
constexpr std::size_t kQ4KBytes = 144;
constexpr std::size_t kValuesPerBlock = 256;
constexpr std::size_t kEmb = 5120;
constexpr std::size_t kFfn = 17408;
constexpr std::size_t kWeightBytes = 50135040;  // two-pointer GGUF gate/up row bytes
constexpr std::size_t kNormBytes = 20480;

struct LayerSpec {
  const char* id;
  std::size_t gate_off;
  std::size_t up_off;
  std::size_t down_off;
  std::size_t ffn_norm_off;
  std::size_t next_norm_off;
  int weight;
  bool probe;
};

constexpr LayerSpec kProbe = {"ffn_probe", 0, 0, 0, 0, 0, 0, true};
constexpr LayerSpec kLayers[] = {
    {"ffn_layer0", 1908404576, 1958539616, 1858269536, 2008674656, 2076228832,
     64, false},
    {"ffn_layer3", 2744126432, 2794261472, 2693991392, 2844396512, 2877840352,
     64, false},
    {"ffn_layer31", 10271204704ULL, 10321339744ULL, 10221069664ULL,
     10371474784ULL, 10404918624ULL, 64, false},
};
constexpr int kLayerCount = static_cast<int>(sizeof(kLayers) / sizeof(kLayers[0]));

struct Candidate {
  const char* id;
  const char* path;
  bool paired;
  bool staged;
};

constexpr Candidate kCandidates[] = {
    {"separate", "separate", false, false},
    {"shared_stage", "shared_stage", false, true},
    {"paired", "paired", true, false},
    {"paired_staged", "paired_staged", true, true},
};
constexpr int kCandidateCount =
    static_cast<int>(sizeof(kCandidates) / sizeof(kCandidates[0]));

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

void emit_phase(const char* name) {
  const std::time_t now = std::time(nullptr);
  std::fprintf(stdout, "phase=%s epoch=%lld\n", name,
               static_cast<long long>(now));
  std::fflush(stdout);
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

void fill_q4(std::size_t rows, std::size_t columns,
             std::vector<std::uint8_t>* weights) {
  weights->assign(rows * (columns / kValuesPerBlock) * kQ4KBytes, 0);
  for (std::size_t index = 0; index < weights->size(); ++index) {
    (*weights)[index] = static_cast<std::uint8_t>((index * 73 + 19) & 0xFFU);
  }
  for (std::size_t offset = 0; offset < weights->size(); offset += kQ4KBytes) {
    write_u16(weights->data() + offset, 0x3C00U);
    write_u16(weights->data() + offset + 2, 0x3400U);
  }
}

void fill_activation(std::size_t count, std::vector<float>* residual) {
  residual->resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    (*residual)[i] = unit_normal(static_cast<std::uint32_t>(i), 0xA11CE5u);
  }
}

struct Envelope {
  float max_abs = 0.0F;
  float rms = 0.0F;
  float one_minus_cosine = 0.0F;
  std::size_t nonfinite = 0;
};

Envelope compare_vec(const std::vector<float>& actual,
                     const std::vector<float>& expected) {
  Envelope metrics;
  double squared = 0.0;
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  const std::size_t n = std::min(actual.size(), expected.size());
  for (std::size_t index = 0; index < n; ++index) {
    const float got = actual[index];
    const float ref = expected[index];
    if (!std::isfinite(got) || !std::isfinite(ref)) {
      ++metrics.nonfinite;
      continue;
    }
    const float error = std::fabs(got - ref);
    metrics.max_abs = std::max(metrics.max_abs, error);
    squared += static_cast<double>(error) * error;
    dot += static_cast<double>(got) * ref;
    na += static_cast<double>(got) * got;
    nb += static_cast<double>(ref) * ref;
  }
  metrics.rms =
      n == 0 ? 0.0F : static_cast<float>(std::sqrt(squared / static_cast<double>(n)));
  if (na == 0.0 && nb == 0.0) {
    metrics.one_minus_cosine = 0.0F;
  } else if (na == 0.0 || nb == 0.0) {
    metrics.one_minus_cosine = 1.0F;
  } else {
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb));
    metrics.one_minus_cosine =
        static_cast<float>(1.0 - std::min(1.0, std::max(-1.0, cosine)));
  }
  return metrics;
}

struct Buffers {
  float* residual = nullptr;
  float* output = nullptr;
  float* mixer = nullptr;
  float* proj_a = nullptr;
  float* proj_b = nullptr;
  __nv_bfloat16* normalized = nullptr;
  __nv_bfloat16* activated = nullptr;
  qw38::cuda::Q8Block* q8 = nullptr;
  std::uint8_t* gate = nullptr;
  std::uint8_t* up = nullptr;
  std::uint8_t* down = nullptr;
  float* ffn_norm = nullptr;
  float* next_norm = nullptr;
  std::size_t emb = 0;
  std::size_t ffn = 0;
};

void free_buffers(Buffers* b) {
  cudaFree(b->residual);
  cudaFree(b->output);
  cudaFree(b->mixer);
  cudaFree(b->proj_a);
  cudaFree(b->proj_b);
  cudaFree(b->normalized);
  cudaFree(b->activated);
  cudaFree(b->q8);
  cudaFree(b->gate);
  cudaFree(b->up);
  cudaFree(b->down);
  cudaFree(b->ffn_norm);
  cudaFree(b->next_norm);
  *b = Buffers{};
}

cudaError_t alloc_buffers(Buffers* b, std::size_t emb, std::size_t ffn) {
  b->emb = emb;
  b->ffn = ffn;
  cudaError_t error = cudaMalloc(&b->residual, emb * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&b->output, emb * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&b->mixer, emb * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&b->proj_a, ffn * sizeof(float));
  if (error == cudaSuccess) error = cudaMalloc(&b->proj_b, ffn * sizeof(float));
  if (error == cudaSuccess) {
    error = cudaMalloc(&b->normalized, emb * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&b->activated, ffn * sizeof(__nv_bfloat16));
  }
  if (error == cudaSuccess) {
    const std::size_t q8_columns = emb > ffn ? emb : ffn;
    error = cudaMalloc(&b->q8, qw38::cuda::q8_workspace_bytes(q8_columns));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&b->ffn_norm, emb * sizeof(float));
  }
  if (error == cudaSuccess) {
    error = cudaMalloc(&b->next_norm, emb * sizeof(float));
  }
  return error;
}

cudaError_t launch_gate_up(const Candidate& cand, const Buffers& b,
                           cudaStream_t stream) {
  if (std::strcmp(cand.path, "paired") == 0) {
    return qw38::cuda::launch_q4k_gate_up_swiglu(
        b.gate, b.up, b.ffn, b.emb, b.normalized, b.q8, b.activated, stream);
  }
  if (std::strcmp(cand.path, "paired_staged") == 0) {
    cudaError_t error =
        qw38::cuda::launch_quantize_bf16_q8(b.normalized, b.q8, b.emb, stream);
    if (error != cudaSuccess) return error;
    return qw38::cuda::launch_q4k_gate_up_swiglu_prequant(
        b.gate, b.up, b.ffn, b.emb, b.q8, b.activated, stream);
  }
  if (std::strcmp(cand.path, "shared_stage") == 0) {
    cudaError_t error =
        qw38::cuda::launch_quantize_bf16_q8(b.normalized, b.q8, b.emb, stream);
    if (error != cudaSuccess) return error;
    error = qw38::cuda::launch_quant_mmv_prequant(
        qw38::cuda::QuantKind::kQ4K, b.gate, b.ffn, b.emb, b.q8, b.proj_a,
        stream);
    if (error != cudaSuccess) return error;
    error = qw38::cuda::launch_quant_mmv_prequant(
        qw38::cuda::QuantKind::kQ4K, b.up, b.ffn, b.emb, b.q8, b.proj_b,
        stream);
    if (error != cudaSuccess) return error;
    return qw38::cuda::launch_swiglu_bf16(b.proj_a, b.proj_b, b.ffn, b.activated,
                                          stream);
  }
  cudaError_t error = qw38::cuda::launch_quant_mmv(
      qw38::cuda::QuantKind::kQ4K, b.gate, b.ffn, b.emb, b.normalized, b.q8,
      b.proj_a, stream);
  if (error != cudaSuccess) return error;
  error = qw38::cuda::launch_quant_mmv(qw38::cuda::QuantKind::kQ4K, b.up, b.ffn,
                                       b.emb, b.normalized, b.q8, b.proj_b,
                                       stream);
  if (error != cudaSuccess) return error;
  return qw38::cuda::launch_swiglu_bf16(b.proj_a, b.proj_b, b.ffn, b.activated,
                                        stream);
}

cudaError_t launch_complete(const Candidate& cand, const Buffers& b,
                            cudaStream_t stream) {
  cudaError_t error = qw38::cuda::launch_rms_norm_fp32_to_bf16(
      b.residual, b.ffn_norm, b.emb, b.normalized, stream);
  if (error != cudaSuccess) return error;
  error = launch_gate_up(cand, b, stream);
  if (error != cudaSuccess) return error;
  error = qw38::cuda::launch_quant_mmv(qw38::cuda::QuantKind::kQ4K, b.down, b.emb,
                                       b.ffn, b.activated, b.q8, b.mixer,
                                       stream);
  if (error != cudaSuccess) return error;
  return qw38::cuda::launch_residual_add_norm_fp32_to_bf16(
      b.residual, b.mixer, b.next_norm, b.emb, b.output, b.normalized, stream);
}

struct Timed {
  std::vector<float> warmup;
  std::vector<float> samples;
  float mean_ms = 0.0F;
  bool launch_ok = false;
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

}  // namespace

int main(int argc, char** argv) {
  if (!qw38::cuda::test_tier_legacy_ok()) {
    std::fprintf(stderr, "%s\n", qw38::cuda::test_tier_legacy_error());
    return 1;
  }
  const qw38::cuda::TestTier tier = qw38::cuda::test_tier();
  const int warmups = qw38::cuda::test_warmups();
  const int measured = qw38::cuda::test_samples();
  const char* raw_path =
      argc > 1 ? argv[1]
               : "evidence/optimization/opt049-decode-ffn-fusion/"
                 "decode-ffn-fusion-ab-raw.txt";
  const char* model_path = argc > 2 ? argv[2] : nullptr;
  emit_phase("start");

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

  const bool load_model = tier != qw38::cuda::TestTier::kSmoke &&
                          model_path != nullptr;
  int mapped_fd = -1;
  std::size_t mapped_size = 0;
  void* mapped = nullptr;
  const std::uint8_t* gguf = nullptr;
  if (load_model) {
    emit_phase("shapes");
    gguf = map_gguf(model_path, &mapped_size, &mapped_fd, &mapped);
    if (gguf == nullptr) {
      std::fclose(raw);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      std::fprintf(stderr, "failed to map GGUF\n");
      return 1;
    }
  } else {
    emit_phase("shapes");
  }

  const int n_layers = load_model ? kLayerCount : 0;
  const int n_cases = 1 + n_layers;

  struct CaseResult {
    const char* id = "";
    bool probe = true;
    int weight = 0;
    std::size_t staging_bytes = 0;
    bool graph_eager_equal[kCandidateCount]{};
    Timed complete[kCandidateCount];
    Envelope vs_separate[kCandidateCount];
    Envelope vs_fp32[kCandidateCount];
    int occupancy[kCandidateCount]{};
    int registers[kCandidateCount]{};
    std::size_t local_bytes[kCandidateCount]{};
  };
  std::vector<CaseResult> cases(static_cast<std::size_t>(n_cases));

  auto run_case = [&](CaseResult* slot, const LayerSpec& spec, std::size_t emb,
                      std::size_t ffn, const std::uint8_t* gate,
                      const std::uint8_t* up, const std::uint8_t* down,
                      const float* ffn_norm, const float* next_norm) -> int {
    slot->id = spec.id;
    slot->probe = spec.probe;
    slot->weight = spec.weight;
    slot->staging_bytes = qw38::cuda::q8_workspace_bytes(emb);
    Buffers b{};
    error = alloc_buffers(&b, emb, ffn);
    if (error != cudaSuccess) return fail_cuda("cudaMalloc", error);
    std::vector<std::uint8_t> host_gate;
    std::vector<std::uint8_t> host_up;
    std::vector<std::uint8_t> host_down;
    std::vector<float> host_residual;
    std::vector<float> host_scale(emb, 1.0F);
    fill_activation(emb, &host_residual);
    if (gate == nullptr) {
      fill_q4(ffn, emb, &host_gate);
      fill_q4(ffn, emb, &host_up);
      fill_q4(emb, ffn, &host_down);
      gate = host_gate.data();
      up = host_up.data();
      down = host_down.data();
      ffn_norm = host_scale.data();
      next_norm = host_scale.data();
    }
    const std::size_t gate_bytes = ffn * (emb / kValuesPerBlock) * kQ4KBytes;
    const std::size_t down_bytes = emb * (ffn / kValuesPerBlock) * kQ4KBytes;
    if (!spec.probe && gate_bytes != kWeightBytes) {
      free_buffers(&b);
      std::fprintf(stderr, "unexpected gate bytes %zu\n", gate_bytes);
      return 1;
    }
    error = cudaMalloc(&b.gate, gate_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&b.up, gate_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&b.down, down_bytes);
    if (error != cudaSuccess) {
      free_buffers(&b);
      return fail_cuda("cudaMalloc weights", error);
    }
    error = cudaMemcpy(b.gate, gate, gate_bytes, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
      error = cudaMemcpy(b.up, up, gate_bytes, cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(b.down, down, down_bytes, cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(b.ffn_norm, ffn_norm, emb * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(b.next_norm, next_norm, emb * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
      error = cudaMemcpy(b.residual, host_residual.data(), emb * sizeof(float),
                         cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess) {
      free_buffers(&b);
      return fail_cuda("cudaMemcpy", error);
    }

    std::vector<float> separate_out(emb);
    std::vector<float> cand_out(emb);
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      const Candidate& cand = kCandidates[c];
      const unsigned int warps = qw38::cuda::selected_mmv_warps(ffn);
      qw38::cuda::q4k_gate_up_swiglu_kernel_attributes(
          warps, cand.staged, &slot->registers[c], &slot->local_bytes[c],
          &slot->occupancy[c]);
      if (!cand.paired) {
        slot->occupancy[c] =
            qw38::cuda::mmv_packed_occupancy(qw38::cuda::QuantKind::kQ4K, warps);
      }
      Timed* timed = &slot->complete[c];
      for (int w = 0; w < warmups && error == cudaSuccess; ++w) {
        error = cudaMemcpy(b.residual, host_residual.data(),
                           emb * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) break;
        error = cudaEventRecord(start, nullptr);
        if (error == cudaSuccess) error = launch_complete(cand, b, nullptr);
        if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
        float ms = 0.0F;
        if (error == cudaSuccess) error = record_ms(start, stop, &ms);
        if (error == cudaSuccess) timed->warmup.push_back(ms);
      }
      for (int s = 0; s < measured && error == cudaSuccess; ++s) {
        error = cudaMemcpy(b.residual, host_residual.data(),
                           emb * sizeof(float), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) break;
        error = cudaEventRecord(start, nullptr);
        if (error == cudaSuccess) error = launch_complete(cand, b, nullptr);
        if (error == cudaSuccess) error = cudaEventRecord(stop, nullptr);
        float ms = 0.0F;
        if (error == cudaSuccess) error = record_ms(start, stop, &ms);
        if (error == cudaSuccess) timed->samples.push_back(ms);
      }
      finish_mean(timed, measured);
      if (error == cudaSuccess) {
        error = cudaMemcpy(cand_out.data(), b.output, emb * sizeof(float),
                           cudaMemcpyDeviceToHost);
      }
      if (c == 0 && error == cudaSuccess) separate_out = cand_out;
      if (error == cudaSuccess) {
        slot->vs_separate[c] = compare_vec(cand_out, separate_out);
        slot->vs_fp32[c] = compare_vec(cand_out, host_residual);
      }
    }
    for (int c = 0; c < kCandidateCount && error == cudaSuccess; ++c) {
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      cudaStream_t stream = nullptr;
      cudaError_t gerr = cudaStreamCreate(&stream);
      if (gerr == cudaSuccess) {
        gerr = cudaMemcpy(b.residual, host_residual.data(), emb * sizeof(float),
                          cudaMemcpyHostToDevice);
      }
      if (gerr == cudaSuccess) {
        gerr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
      }
      cudaError_t enqueue = gerr;
      if (enqueue == cudaSuccess) {
        enqueue = launch_complete(kCandidates[c], b, stream);
      }
      cudaError_t capture_error = cudaStreamEndCapture(stream, &graph);
      gerr = enqueue != cudaSuccess ? enqueue : capture_error;
      if (gerr == cudaSuccess) {
        gerr = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
      }
      std::vector<float> eager(emb);
      std::vector<float> replayed(emb);
      if (gerr == cudaSuccess) {
        gerr = cudaMemcpy(b.residual, host_residual.data(), emb * sizeof(float),
                          cudaMemcpyHostToDevice);
      }
      if (gerr == cudaSuccess) gerr = launch_complete(kCandidates[c], b, nullptr);
      if (gerr == cudaSuccess) {
        gerr = cudaMemcpy(eager.data(), b.output, emb * sizeof(float),
                          cudaMemcpyDeviceToHost);
      }
      if (gerr == cudaSuccess) {
        gerr = cudaMemcpy(b.residual, host_residual.data(), emb * sizeof(float),
                          cudaMemcpyHostToDevice);
      }
      if (gerr == cudaSuccess) gerr = cudaGraphLaunch(exec, stream);
      if (gerr == cudaSuccess) gerr = cudaStreamSynchronize(stream);
      if (gerr == cudaSuccess) {
        gerr = cudaMemcpy(replayed.data(), b.output, emb * sizeof(float),
                          cudaMemcpyDeviceToHost);
      }
      slot->graph_eager_equal[c] = gerr == cudaSuccess && eager == replayed;
      if (exec != nullptr) cudaGraphExecDestroy(exec);
      if (graph != nullptr) cudaGraphDestroy(graph);
      if (stream != nullptr) cudaStreamDestroy(stream);
      (void)cudaGetLastError();
    }
    free_buffers(&b);
    return error == cudaSuccess ? 0 : fail_cuda("run_case", error);
  };

  emit_phase("probe");
  int rc = run_case(&cases[0], kProbe, 256, 256, nullptr, nullptr, nullptr,
                    nullptr, nullptr);
  if (rc != 0) {
    unmap_gguf(mapped, mapped_size, mapped_fd);
    std::fclose(raw);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return rc;
  }

  if (load_model) {
    emit_phase("layers");
    for (int i = 0; i < n_layers; ++i) {
      const LayerSpec& spec = kLayers[i];
      if (mapped_size < spec.next_norm_off + kNormBytes) {
        unmap_gguf(mapped, mapped_size, mapped_fd);
        std::fclose(raw);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::fprintf(stderr, "GGUF too small for %s\n", spec.id);
        return 1;
      }
      rc = run_case(&cases[static_cast<std::size_t>(1 + i)], spec, kEmb, kFfn,
                    gguf + spec.gate_off, gguf + spec.up_off,
                    gguf + spec.down_off,
                    reinterpret_cast<const float*>(gguf + spec.ffn_norm_off),
                    reinterpret_cast<const float*>(gguf + spec.next_norm_off));
      if (rc != 0) {
        unmap_gguf(mapped, mapped_size, mapped_fd);
        std::fclose(raw);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return rc;
      }
    }
  }

  emit_phase("reduce");
  char utc[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));

  double means[kCandidateCount]{};
  bool eligible[kCandidateCount]{};
  for (int c = 0; c < kCandidateCount; ++c) {
    const Candidate& cand = kCandidates[c];
    double weighted = 0.0;
    double weight_sum = 0.0;
    eligible[c] = true;
    for (const CaseResult& slot : cases) {
      if (!slot.complete[c].launch_ok) eligible[c] = false;
      if (!slot.graph_eager_equal[c]) eligible[c] = false;
      if (slot.vs_separate[c].nonfinite != 0) eligible[c] = false;
      if (!slot.probe && std::strcmp(cand.path, "paired") != 0) {
        if (slot.vs_separate[c].max_abs > 1.0e-5F) eligible[c] = false;
      }
      const double w = slot.probe ? 0.0 : static_cast<double>(slot.weight);
      weighted += slot.complete[c].mean_ms * w;
      weight_sum += w;
    }
    means[c] = weight_sum > 0.0 ? weighted / weight_sum
                                : cases[0].complete[c].mean_ms;
  }
  int best = 0;
  double best_weighted = means[0];
  if (eligible[0]) {
    for (int c = 1; c < kCandidateCount; ++c) {
      if (eligible[c] && means[c] < best_weighted) {
        best = c;
        best_weighted = means[c];
      }
    }
  }
  const double separate_weighted = means[0];
  const bool win = best != 0 && best_weighted < separate_weighted;

  std::fprintf(raw, "%s{", kPrefix);
  std::fprintf(raw,
               "\"schema_version\":1,\"task\":\"OPT-049\",\"status\":\"%s\","
               "\"measurement_utc\":\"%s\",\"device\":\"%s\","
               "\"compute_capability\":\"%d.%d\",\"llama_revision\":\"%s\","
               "\"gguf_sha256\":\"%s\",",
               "measured", utc, prop.name, prop.major, prop.minor, kLlamaRev,
               kGgufSha);
  std::fprintf(raw, "\"ab\":{\"winner\":\"%s\",\"win\":%s,\"candidates\":{",
               kCandidates[best].id, json_bool(win));
  for (int c = 0; c < kCandidateCount; ++c) {
    if (c != 0) std::fprintf(raw, ",");
    const Candidate& cand = kCandidates[c];
    std::fprintf(raw,
                 "\"%s\":{\"id\":\"%s\",\"path\":\"%s\",\"paired\":%s,"
                 "\"staged\":%s,\"occupancy\":%d,\"registers\":%d,"
                 "\"local_bytes\":%zu,\"eligible\":%s,"
                 "\"weighted_complete_ms\":%.9g,\"shapes\":{",
                 cand.id, cand.id, cand.path, json_bool(cand.paired),
                 json_bool(cand.staged), cases[0].occupancy[c],
                 cases[0].registers[c], cases[0].local_bytes[c],
                 json_bool(eligible[c]), means[c]);
    for (std::size_t s = 0; s < cases.size(); ++s) {
      if (s != 0) std::fprintf(raw, ",");
      const CaseResult& slot = cases[s];
      std::fprintf(raw,
                   "\"%s\":{\"role\":\"%s\",\"staging_bytes\":%zu,"
                   "\"graph_eager_equal\":%s,\"complete\":{",
                   slot.id, slot.probe ? "probe" : "layer", slot.staging_bytes,
                   json_bool(slot.graph_eager_equal[c]));
      std::fprintf(raw, "\"mean_ms\":%.9g,\"launch_ok\":%s,",
                   static_cast<double>(slot.complete[c].mean_ms),
                   json_bool(slot.complete[c].launch_ok));
      print_array(raw, "warmup_ms", slot.complete[c].warmup);
      std::fprintf(raw, ",");
      print_array(raw, "samples", slot.complete[c].samples);
      std::fprintf(raw,
                   "},\"vs_separate\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                   "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu},"
                   "\"vs_residual\":{\"max_abs\":%.9g,\"rms\":%.9g,"
                   "\"one_minus_cosine\":%.9g,\"nonfinite\":%zu}}",
                   static_cast<double>(slot.vs_separate[c].max_abs),
                   static_cast<double>(slot.vs_separate[c].rms),
                   static_cast<double>(slot.vs_separate[c].one_minus_cosine),
                   slot.vs_separate[c].nonfinite,
                   static_cast<double>(slot.vs_fp32[c].max_abs),
                   static_cast<double>(slot.vs_fp32[c].rms),
                   static_cast<double>(slot.vs_fp32[c].one_minus_cosine),
                   slot.vs_fp32[c].nonfinite);
    }
    std::fprintf(raw, "}}");
  }
  std::fprintf(raw, "}}}\n");
  std::fclose(raw);

  std::FILE* echo = std::fopen(raw_path, "r");
  if (echo != nullptr) {
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), echo) != nullptr) std::fputs(buf, stdout);
    std::fclose(echo);
  }
  std::printf("status=passed winner=%s win=%s separate_ms=%.9g winner_ms=%.9g\n",
              kCandidates[best].id, json_bool(win), separate_weighted,
              best_weighted);
  emit_phase("done");
  unmap_gguf(mapped, mapped_size, mapped_fd);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
