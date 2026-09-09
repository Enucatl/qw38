#include "gdn_step.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <vector>

#include <cuda_bf16.h>

#include "gdn.h"

namespace {

constexpr float kMaxAbs = 5.0e-8F;
constexpr float kMaxRms = 5.0e-9F;
constexpr int kWarmups = 3;
constexpr int kSamples = 30;
constexpr std::size_t kFfnWidth = 17408;
constexpr std::size_t kProductionRecurrent = 786432;
constexpr std::uint64_t kFrontier = 7;

const char* json_bool(bool value) { return value ? "true" : "false"; }

int fail_cuda(const char* operation, cudaError_t error) {
  std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
  return 1;
}

bool within_envelope(float max_abs, float rms, std::size_t nonfinite) {
  return nonfinite == 0 && max_abs <= kMaxAbs && rms <= kMaxRms;
}

void accumulate(const std::vector<float>& actual, const std::vector<float>& expected,
                float* max_abs, double* squared, std::size_t* count,
                std::size_t* nonfinite) {
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      ++*nonfinite;
    }
    const float difference = std::fabs(actual[index] - expected[index]);
    *max_abs = std::max(*max_abs, difference);
    *squared += static_cast<double>(difference) * difference;
    ++*count;
  }
}

bool host_chunk(const qw38::cuda::GdnConfig& config, std::size_t token_count,
                const std::vector<float>& input, const std::vector<float>& weights,
                const std::vector<float>& log_decay, const std::vector<float>& beta,
                const std::vector<float>& initial_convolution,
                const std::vector<float>& initial_recurrent,
                std::vector<float>* final_convolution,
                std::vector<float>* final_recurrent,
                std::vector<float>* convolution_output,
                std::vector<float>* recurrent_output) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  const std::size_t query_count =
      static_cast<std::size_t>(config.key_heads) * config.key_width;
  *final_convolution = initial_convolution;
  *final_recurrent = initial_recurrent;
  convolution_output->assign(token_count * channels, 0.0F);
  recurrent_output->assign(token_count * output_values, 0.0F);
  const qw38::internal::GdnShape shape{config.key_heads, config.value_heads,
                                       config.key_width, config.value_width};
  for (std::size_t token = 0; token < token_count; ++token) {
    qw38::Status status = qw38::internal::causal_depthwise_conv_step(
        channels, config.convolution_width, input.data() + token * channels,
        channels, weights.data(), weights.size(), final_convolution->data(),
        final_convolution->size(),
        convolution_output->data() + token * channels, channels);
    if (!status.is_ok()) return false;
    const float* token_convolution =
        convolution_output->data() + token * channels;
    status = qw38::internal::gdn_recurrent_step_precomputed(
        shape, token_convolution, query_count,
        token_convolution + query_count, query_count,
        token_convolution + 2 * query_count, output_values,
        log_decay.data() + token * config.value_heads,
        beta.data() + token * config.value_heads, config.value_heads,
        final_recurrent->data(), final_recurrent->size(),
        recurrent_output->data() + token * output_values, output_values);
    if (!status.is_ok()) return false;
  }
  return true;
}

void fill_synthetic(const qw38::cuda::GdnConfig& config, std::size_t token_count,
                    std::vector<float>* input, std::vector<float>* weights,
                    std::vector<float>* log_decay, std::vector<float>* beta,
                    std::vector<float>* initial_convolution,
                    std::vector<float>* initial_recurrent) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(config);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(config);
  input->resize(token_count * channels);
  weights->resize(convolution_values);
  log_decay->resize(token_count * config.value_heads);
  beta->resize(token_count * config.value_heads);
  initial_convolution->resize(convolution_values);
  initial_recurrent->resize(recurrent_values);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t channel = 0; channel < channels; ++channel) {
      (*input)[token * channels + channel] =
          std::sin(static_cast<float>(channel) * 0.013F +
                   static_cast<float>(token) * 0.071F) *
          0.5F;
    }
    for (std::size_t head = 0; head < config.value_heads; ++head) {
      (*log_decay)[token * config.value_heads + head] =
          -0.001F * static_cast<float>(1 + (token + head) % 31);
      (*beta)[token * config.value_heads + head] =
          0.15F + 0.01F * static_cast<float>((token + head) % 23);
    }
  }
  for (std::size_t index = 0; index < convolution_values; ++index) {
    (*weights)[index] =
        static_cast<float>(static_cast<int>(index % 9) - 4) * 0.03125F;
    (*initial_convolution)[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.015625F;
  }
  for (std::size_t index = 0; index < recurrent_values; ++index) {
    (*initial_recurrent)[index] =
        static_cast<float>(static_cast<int>(index % 23) - 11) * 0.0009765625F;
  }
}

struct DeviceBuffers {
  float* input = nullptr;
  float* weights = nullptr;
  float* log_decay = nullptr;
  float* beta = nullptr;
  float* committed_convolution = nullptr;
  float* committed_recurrent = nullptr;
  float* candidate_convolution = nullptr;
  float* candidate_recurrent = nullptr;
  float* convolution_output = nullptr;
  float* recurrent_output = nullptr;
  float* scratch = nullptr;
  std::uint64_t* frontier = nullptr;
};

void release(DeviceBuffers* buffers) {
  cudaFree(buffers->frontier);
  cudaFree(buffers->scratch);
  cudaFree(buffers->recurrent_output);
  cudaFree(buffers->convolution_output);
  cudaFree(buffers->candidate_recurrent);
  cudaFree(buffers->candidate_convolution);
  cudaFree(buffers->committed_recurrent);
  cudaFree(buffers->committed_convolution);
  cudaFree(buffers->beta);
  cudaFree(buffers->log_decay);
  cudaFree(buffers->weights);
  cudaFree(buffers->input);
}

cudaError_t allocate(DeviceBuffers* buffers, const qw38::cuda::GdnConfig& config,
                    std::size_t token_count) {
  const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
  const std::size_t convolution_values =
      qw38::cuda::gdn_convolution_values(config);
  const std::size_t recurrent_values = qw38::cuda::gdn_recurrent_values(config);
  const std::size_t output_values = qw38::cuda::gdn_output_values(config);
  const std::size_t scratch_floats =
      qw38::cuda::gdn_scan_scratch_floats(config, token_count);
  cudaError_t error =
      cudaMalloc(&buffers->input, token_count * channels * sizeof(float));
#define QW38_ALLOC(field, count)                                               \
  if (error == cudaSuccess)                                                  \
  error = cudaMalloc(&buffers->field, (count) * sizeof(*buffers->field))
  QW38_ALLOC(weights, convolution_values);
  QW38_ALLOC(log_decay, token_count * config.value_heads);
  QW38_ALLOC(beta, token_count * config.value_heads);
  QW38_ALLOC(committed_convolution, convolution_values);
  QW38_ALLOC(committed_recurrent, recurrent_values);
  QW38_ALLOC(candidate_convolution, convolution_values);
  QW38_ALLOC(candidate_recurrent, recurrent_values);
  QW38_ALLOC(convolution_output, token_count * channels);
  QW38_ALLOC(recurrent_output, token_count * output_values);
  QW38_ALLOC(scratch, scratch_floats);
  QW38_ALLOC(frontier, 1);
#undef QW38_ALLOC
  return error;
}

cudaError_t copy_inputs(DeviceBuffers* buffers, const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const std::vector<float>& log_decay,
                         const std::vector<float>& beta,
                         const std::vector<float>& initial_convolution,
                         const std::vector<float>& initial_recurrent) {
  cudaError_t error = cudaSuccess;
#define QW38_COPY(field, source)                                               \
  if (error == cudaSuccess)                                                  \
  error = cudaMemcpy(buffers->field, (source).data(),                         \
                     (source).size() * sizeof((source)[0]),                   \
                     cudaMemcpyHostToDevice)
  QW38_COPY(input, input);
  QW38_COPY(weights, weights);
  QW38_COPY(log_decay, log_decay);
  QW38_COPY(beta, beta);
  QW38_COPY(committed_convolution, initial_convolution);
  QW38_COPY(committed_recurrent, initial_recurrent);
#undef QW38_COPY
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->frontier, &kFrontier, sizeof(kFrontier),
                       cudaMemcpyHostToDevice);
  }
  return error;
}

cudaError_t restore_committed(DeviceBuffers* buffers,
                              const std::vector<float>& initial_convolution,
                              const std::vector<float>& initial_recurrent) {
  cudaError_t error = cudaMemcpy(
      buffers->committed_convolution, initial_convolution.data(),
      initial_convolution.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->committed_recurrent, initial_recurrent.data(),
                       initial_recurrent.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpy(buffers->frontier, &kFrontier, sizeof(kFrontier),
                       cudaMemcpyHostToDevice);
  }
  return error;
}

cudaError_t read_outputs(const DeviceBuffers& buffers,
                         std::vector<float>* candidate_convolution,
                         std::vector<float>* candidate_recurrent,
                         std::vector<float>* convolution_output,
                         std::vector<float>* recurrent_output,
                         std::vector<float>* committed_convolution,
                         std::vector<float>* committed_recurrent,
                         std::uint64_t* frontier) {
  cudaError_t error = cudaSuccess;
#define QW38_READ(destination, field)                                           \
  if (error == cudaSuccess)                                                  \
  error = cudaMemcpy((destination)->data(), buffers.field,                  \
                     (destination)->size() * sizeof((*destination)[0]),       \
                     cudaMemcpyDeviceToHost)
  QW38_READ(candidate_convolution, candidate_convolution);
  QW38_READ(candidate_recurrent, candidate_recurrent);
  QW38_READ(convolution_output, convolution_output);
  QW38_READ(recurrent_output, recurrent_output);
  QW38_READ(committed_convolution, committed_convolution);
  QW38_READ(committed_recurrent, committed_recurrent);
#undef QW38_READ
  if (error == cudaSuccess) {
    error = cudaMemcpy(frontier, buffers.frontier, sizeof(*frontier),
                       cudaMemcpyDeviceToHost);
  }
  return error;
}

struct Envelope {
  const char* name = "";
  std::size_t tokens = 0;
  float max_abs = 0.0F;
  float rms = 0.0F;
  std::size_t nonfinite = 0;
  bool prepare_atomic = false;
  bool tokenwise_equal = false;
  bool conv_memcmp = false;
  bool vs_scalar = false;
  bool passed = false;
};

void compare_four(const std::vector<float>& a_conv, const std::vector<float>& a_rec,
                  const std::vector<float>& a_cconv, const std::vector<float>& a_crec,
                  const std::vector<float>& b_conv, const std::vector<float>& b_rec,
                  const std::vector<float>& b_cconv, const std::vector<float>& b_crec,
                  Envelope* envelope) {
  double squared = 0.0;
  std::size_t count = 0;
  accumulate(a_conv, b_conv, &envelope->max_abs, &squared, &count,
             &envelope->nonfinite);
  accumulate(a_rec, b_rec, &envelope->max_abs, &squared, &count,
             &envelope->nonfinite);
  accumulate(a_cconv, b_cconv, &envelope->max_abs, &squared, &count,
             &envelope->nonfinite);
  accumulate(a_crec, b_crec, &envelope->max_abs, &squared, &count,
             &envelope->nonfinite);
  envelope->rms = static_cast<float>(
      std::sqrt(squared / static_cast<double>(count)));
  envelope->conv_memcmp =
      a_conv == b_conv && a_cconv == b_cconv;
}

qw38::cuda::GdnState committed_state(const DeviceBuffers& buffers) {
  return {buffers.committed_convolution, buffers.committed_recurrent};
}

qw38::cuda::GdnState candidate_state(const DeviceBuffers& buffers) {
  return {buffers.candidate_convolution, buffers.candidate_recurrent};
}

cudaError_t launch_path(const qw38::cuda::GdnConfig& config,
                         const DeviceBuffers& buffers, std::size_t token_count,
                         qw38::cuda::GdnScanPath path, bool tiled,
                         cudaStream_t stream) {
  const std::size_t scratch_floats =
      qw38::cuda::gdn_scan_scratch_floats(config, token_count);
  if (tiled) {
    return qw38::cuda::launch_gdn_prepare_chunk_tiled(
        config, buffers.input, buffers.weights, buffers.log_decay, buffers.beta,
        token_count, committed_state(buffers), candidate_state(buffers),
        buffers.convolution_output, buffers.recurrent_output, stream, path,
        buffers.scratch, scratch_floats);
  }
  return qw38::cuda::launch_gdn_prepare_chunk(
      config, buffers.input, buffers.weights, buffers.log_decay, buffers.beta,
      token_count, committed_state(buffers), candidate_state(buffers),
      buffers.convolution_output, buffers.recurrent_output, stream, path,
      buffers.scratch, scratch_floats);
}

struct KernelNode {
  dim3 grid{};
  dim3 block{};
};

struct Capture {
  bool ok = false;
  std::vector<KernelNode> kernels;
};

template <typename Fn>
Capture capture(Fn launch) {
  Capture info;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaError_t error =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (error == cudaSuccess) {
    error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  }
  if (error == cudaSuccess) error = launch(stream);
  if (error == cudaSuccess) error = cudaStreamEndCapture(stream, &graph);
  std::size_t count = 0;
  if (error == cudaSuccess) error = cudaGraphGetNodes(graph, nullptr, &count);
  std::vector<cudaGraphNode_t> nodes(count);
  if (error == cudaSuccess) {
    error = cudaGraphGetNodes(graph, nodes.data(), &count);
  }
  for (cudaGraphNode_t node : nodes) {
    if (error != cudaSuccess) break;
    cudaGraphNodeType type{};
    if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    if (type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams params{};
    if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
      error = cudaErrorInvalidValue;
      break;
    }
    info.kernels.push_back({params.gridDim, params.blockDim});
  }
  if (graph != nullptr) cudaGraphDestroy(graph);
  if (stream != nullptr) cudaStreamDestroy(stream);
  info.ok = error == cudaSuccess;
  return info;
}

std::size_t count_nodes(const Capture& capture, unsigned gx, unsigned gy,
                        unsigned bx) {
  std::size_t count = 0;
  for (const KernelNode& node : capture.kernels) {
    if (node.grid.x == gx && node.grid.y == gy && node.block.x == bx) ++count;
  }
  return count;
}

bool committed_unchanged(const std::vector<float>& committed_convolution,
                          const std::vector<float>& committed_recurrent,
                          const std::vector<float>& initial_convolution,
                          const std::vector<float>& initial_recurrent,
                          std::uint64_t frontier) {
  return committed_convolution == initial_convolution &&
         committed_recurrent == initial_recurrent && frontier == kFrontier;
}

}  // namespace

int main() {
  const qw38::cuda::GdnConfig small{2, 6, 8, 8, 4};
  const qw38::cuda::GdnConfig production{16, 48, 128, 128, 4};
  std::vector<Envelope> sequential_cases;
  std::vector<Envelope> parallel_cases;
  Envelope tiled_case;
  Envelope boundary_4096;
  bool sequential_ok = true;
  bool parallel_ok = true;
  bool tiled_ok = false;
  bool boundary_ok = false;

  auto run_prepare = [&](const char* name, const qw38::cuda::GdnConfig& config,
                          std::size_t token_count, bool want_tokenwise,
                          bool want_scalar, bool one_window_memcmp,
                          bool tiled) -> Envelope {
    Envelope envelope;
    envelope.name = name;
    envelope.tokens = token_count;
    std::vector<float> input, weights, log_decay, beta, initial_convolution,
        initial_recurrent;
    fill_synthetic(config, token_count, &input, &weights, &log_decay, &beta,
                   &initial_convolution, &initial_recurrent);
    const std::size_t channels = qw38::cuda::gdn_convolution_channels(config);
    const std::size_t convolution_values =
        qw38::cuda::gdn_convolution_values(config);
    const std::size_t recurrent_values =
        qw38::cuda::gdn_recurrent_values(config);
    const std::size_t output_values = qw38::cuda::gdn_output_values(config);
    DeviceBuffers device;
    cudaError_t error = allocate(&device, config, token_count);
    if (error != cudaSuccess) {
      fail_cuda(name, error);
      envelope.passed = false;
      return envelope;
    }
    error = copy_inputs(&device, input, weights, log_decay, beta,
                        initial_convolution, initial_recurrent);
    if (error != cudaSuccess) {
      fail_cuda(name, error);
      release(&device);
      envelope.passed = false;
      return envelope;
    }
    error = launch_path(config, device, token_count,
                         qw38::cuda::GdnScanPath::kSequentialWindows, tiled,
                         nullptr);
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      fail_cuda(name, error);
      release(&device);
      envelope.passed = false;
      return envelope;
    }
    std::vector<float> seq_cconv(convolution_values),
        seq_crec(recurrent_values), seq_conv(token_count * channels),
        seq_rec(token_count * output_values), seq_committed_conv(convolution_values),
        seq_committed_rec(recurrent_values);
    std::uint64_t frontier = 0;
    error = read_outputs(device, &seq_cconv, &seq_crec, &seq_conv, &seq_rec,
                         &seq_committed_conv, &seq_committed_rec, &frontier);
    if (error != cudaSuccess) {
      fail_cuda(name, error);
      release(&device);
      envelope.passed = false;
      return envelope;
    }
    envelope.prepare_atomic = committed_unchanged(
        seq_committed_conv, seq_committed_rec, initial_convolution,
        initial_recurrent, frontier);
    error = restore_committed(&device, initial_convolution, initial_recurrent);
    if (error == cudaSuccess) {
      error = launch_path(config, device, token_count,
                           qw38::cuda::GdnScanPath::kFusedTokenLoop, tiled,
                           nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> fused_cconv(convolution_values),
        fused_crec(recurrent_values), fused_conv(token_count * channels),
        fused_rec(token_count * output_values);
    std::uint64_t fused_frontier = 0;
    std::vector<float> fused_committed_conv(convolution_values),
        fused_committed_rec(recurrent_values);
    if (error == cudaSuccess) {
      error = read_outputs(device, &fused_cconv, &fused_crec, &fused_conv,
                           &fused_rec, &fused_committed_conv,
                           &fused_committed_rec, &fused_frontier);
    }
    Envelope fused_vs_seq;
    if (error == cudaSuccess) {
      compare_four(fused_conv, fused_rec, fused_cconv, fused_crec, seq_conv,
                   seq_rec, seq_cconv, seq_crec, &fused_vs_seq);
      envelope.max_abs = std::max(envelope.max_abs, fused_vs_seq.max_abs);
      envelope.rms = std::max(envelope.rms, fused_vs_seq.rms);
      envelope.nonfinite += fused_vs_seq.nonfinite;
      envelope.prepare_atomic =
          envelope.prepare_atomic &&
          committed_unchanged(fused_committed_conv, fused_committed_rec,
                              initial_convolution, initial_recurrent,
                              fused_frontier);
    } else {
      envelope.passed = false;
      fail_cuda(name, error);
      release(&device);
      return envelope;
    }
    if (want_tokenwise) {
      error = restore_committed(&device, initial_convolution, initial_recurrent);
      for (std::size_t token = 0; token < token_count && error == cudaSuccess;
           ++token) {
        error = qw38::cuda::launch_gdn_prepare(
            config, device.input + token * channels, device.weights,
            device.log_decay + token * config.value_heads,
            device.beta + token * config.value_heads, committed_state(device),
            candidate_state(device),
            device.convolution_output + token * channels,
            device.recurrent_output + token * output_values, nullptr);
        if (error == cudaSuccess) {
          error = qw38::cuda::launch_gdn_commit(
              config, candidate_state(device), committed_state(device),
              kFrontier + token + 1, device.frontier, nullptr);
        }
      }
      if (error == cudaSuccess) error = cudaDeviceSynchronize();
      std::vector<float> tok_cconv(convolution_values),
          tok_crec(recurrent_values), tok_conv(token_count * channels),
          tok_rec(token_count * output_values), tok_committed_conv(convolution_values),
          tok_committed_rec(recurrent_values);
      std::uint64_t tok_frontier = 0;
      if (error == cudaSuccess) {
        error = read_outputs(device, &tok_cconv, &tok_crec, &tok_conv, &tok_rec,
                             &tok_committed_conv, &tok_committed_rec,
                             &tok_frontier);
      }
      envelope.tokenwise_equal =
          error == cudaSuccess && seq_cconv == tok_cconv &&
          seq_crec == tok_crec && seq_conv == tok_conv && seq_rec == tok_rec;
    } else {
      envelope.tokenwise_equal = true;
    }
    std::vector<float> host_cconv, host_crec, host_conv, host_rec;
    if (want_scalar) {
      envelope.vs_scalar = host_chunk(config, token_count, input, weights,
                                       log_decay, beta, initial_convolution,
                                       initial_recurrent, &host_cconv,
                                       &host_crec, &host_conv, &host_rec);
      if (envelope.vs_scalar) {
        Envelope seq_scalar;
        compare_four(seq_conv, seq_rec, seq_cconv, seq_crec, host_conv,
                     host_rec, host_cconv, host_crec, &seq_scalar);
        envelope.max_abs = std::max(envelope.max_abs, seq_scalar.max_abs);
        envelope.rms = std::max(envelope.rms, seq_scalar.rms);
        envelope.nonfinite += seq_scalar.nonfinite;
        envelope.vs_scalar = within_envelope(
            seq_scalar.max_abs, seq_scalar.rms, seq_scalar.nonfinite);
      }
    } else {
      envelope.vs_scalar = true;
    }
    error = restore_committed(&device, initial_convolution, initial_recurrent);
    if (error == cudaSuccess) {
      error = launch_path(config, device, token_count,
                           qw38::cuda::GdnScanPath::kParallelAssociative, tiled,
                           nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> par_cconv(convolution_values),
        par_crec(recurrent_values), par_conv(token_count * channels),
        par_rec(token_count * output_values), par_committed_conv(convolution_values),
        par_committed_rec(recurrent_values);
    std::uint64_t par_frontier = 0;
    if (error == cudaSuccess) {
      error = read_outputs(device, &par_cconv, &par_crec, &par_conv, &par_rec,
                           &par_committed_conv, &par_committed_rec,
                           &par_frontier);
    }
    if (error != cudaSuccess) {
      fail_cuda(name, error);
      release(&device);
      envelope.passed = false;
      return envelope;
    }
    envelope.prepare_atomic =
        envelope.prepare_atomic &&
        committed_unchanged(par_committed_conv, par_committed_rec,
                            initial_convolution, initial_recurrent,
                            par_frontier);
    compare_four(par_conv, par_rec, par_cconv, par_crec, seq_conv, seq_rec,
                 seq_cconv, seq_crec, &envelope);
    if (want_scalar && envelope.vs_scalar) {
      Envelope scalar;
      compare_four(par_conv, par_rec, par_cconv, par_crec, host_conv, host_rec,
                   host_cconv, host_crec, &scalar);
      envelope.max_abs = std::max(envelope.max_abs, scalar.max_abs);
      envelope.rms = std::max(envelope.rms, scalar.rms);
      envelope.nonfinite += scalar.nonfinite;
      envelope.vs_scalar =
          within_envelope(scalar.max_abs, scalar.rms, scalar.nonfinite);
    }
    if (one_window_memcmp) {
      envelope.passed =
          envelope.prepare_atomic && envelope.tokenwise_equal &&
          envelope.conv_memcmp && envelope.vs_scalar &&
          within_envelope(envelope.max_abs, envelope.rms, envelope.nonfinite);
    } else {
      envelope.conv_memcmp = true;
      envelope.passed =
          envelope.prepare_atomic && envelope.tokenwise_equal &&
          envelope.vs_scalar &&
          within_envelope(envelope.max_abs, envelope.rms, envelope.nonfinite);
    }
    release(&device);
    return envelope;
  };

  const struct {
    const char* name;
    qw38::cuda::GdnConfig config;
    std::size_t tokens;
  } sequential_specs[] = {
      {"small_3", small, 3},
      {"small_64", small, 64},
      {"small_65", small, 65},
      {"small_129", small, 129},
      {"production_65", production, 65},
  };
  for (const auto& spec : sequential_specs) {
    Envelope envelope =
        run_prepare(spec.name, spec.config, spec.tokens, true, true,
                    spec.tokens <= 64, false);
    sequential_ok = sequential_ok && envelope.passed;
    sequential_cases.push_back(envelope);
  }

  const struct {
    const char* name;
    qw38::cuda::GdnConfig config;
    std::size_t tokens;
  } parallel_specs[] = {
      {"small_3", small, 3},
      {"small_64", small, 64},
      {"small_65", small, 65},
      {"small_129", small, 129},
      {"production_64", production, 64},
      {"production_65", production, 65},
      {"production_129", production, 129},
      {"production_256", production, 256},
      {"production_4096", production, 4096},
  };
  for (const auto& spec : parallel_specs) {
    const bool one_window = spec.tokens <= 64;
    const bool vs_scalar = spec.tokens <= 129;
    Envelope envelope =
        run_prepare(spec.name, spec.config, spec.tokens, false, vs_scalar,
                    one_window, false);
    parallel_ok = parallel_ok && envelope.passed;
    parallel_cases.push_back(envelope);
  }

  tiled_case = run_prepare("production_65_tiled", production, 65, false, false,
                           false, true);
  tiled_ok = tiled_case.passed;

  {
    std::vector<float> input, weights, log_decay, beta, initial_convolution,
        initial_recurrent;
    fill_synthetic(production, 4096, &input, &weights, &log_decay, &beta,
                   &initial_convolution, &initial_recurrent);
    DeviceBuffers device;
    cudaError_t error = allocate(&device, production, 4096);
    boundary_4096.name = "production_4096_vs_64_windows";
    boundary_4096.tokens = 4096;
    if (error == cudaSuccess) {
      error = copy_inputs(&device, input, weights, log_decay, beta,
                          initial_convolution, initial_recurrent);
    }
    if (error == cudaSuccess) {
      error = launch_path(production, device, 4096,
                            qw38::cuda::GdnScanPath::kParallelAssociative, false,
                            nullptr);
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    const std::size_t channels = qw38::cuda::gdn_convolution_channels(production);
    const std::size_t convolution_values =
        qw38::cuda::gdn_convolution_values(production);
    const std::size_t recurrent_values =
        qw38::cuda::gdn_recurrent_values(production);
    const std::size_t output_values = qw38::cuda::gdn_output_values(production);
    std::vector<float> par_cconv(convolution_values),
        par_crec(recurrent_values), par_conv(4096 * channels),
        par_rec(4096 * output_values), par_committed_conv(convolution_values),
        par_committed_rec(recurrent_values);
    std::uint64_t par_frontier = 0;
    if (error == cudaSuccess) {
      error = read_outputs(device, &par_cconv, &par_crec, &par_conv, &par_rec,
                           &par_committed_conv, &par_committed_rec,
                           &par_frontier);
    }
    boundary_4096.prepare_atomic = error == cudaSuccess &&
        committed_unchanged(par_committed_conv, par_committed_rec,
                            initial_convolution, initial_recurrent,
                            par_frontier);
    DeviceBuffers split;
    if (error == cudaSuccess) error = allocate(&split, production, 4096);
    if (error == cudaSuccess) {
      error = copy_inputs(&split, input, weights, log_decay, beta,
                          initial_convolution, initial_recurrent);
    }
    const std::size_t scratch_floats =
        qw38::cuda::gdn_scan_scratch_floats(production, 64);
    for (std::size_t start = 0; start < 4096 && error == cudaSuccess;
         start += 64) {
      error = qw38::cuda::launch_gdn_prepare_chunk(
          production, split.input + start * channels, split.weights,
          split.log_decay + start * production.value_heads,
          split.beta + start * production.value_heads, 64,
          committed_state(split), candidate_state(split),
          split.convolution_output + start * channels,
          split.recurrent_output + start * output_values, nullptr,
          qw38::cuda::GdnScanPath::kSequentialWindows, split.scratch,
          scratch_floats);
      if (error == cudaSuccess) {
        error = cudaMemcpy(split.committed_convolution, split.candidate_convolution,
                           convolution_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
      if (error == cudaSuccess) {
        error = cudaMemcpy(split.committed_recurrent, split.candidate_recurrent,
                           recurrent_values * sizeof(float),
                           cudaMemcpyDeviceToDevice);
      }
    }
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    std::vector<float> seq_cconv(convolution_values),
        seq_crec(recurrent_values), seq_conv(4096 * channels),
        seq_rec(4096 * output_values), seq_committed_conv(convolution_values),
        seq_committed_rec(recurrent_values);
    std::uint64_t seq_frontier = 0;
    if (error == cudaSuccess) {
      error = read_outputs(split, &seq_cconv, &seq_crec, &seq_conv, &seq_rec,
                           &seq_committed_conv, &seq_committed_rec,
                           &seq_frontier);
    }
    if (error == cudaSuccess) {
      compare_four(par_conv, par_rec, par_cconv, par_crec, seq_conv, seq_rec,
                   seq_cconv, seq_crec, &boundary_4096);
      boundary_4096.conv_memcmp = true;
      boundary_ok =
          boundary_4096.prepare_atomic &&
          within_envelope(boundary_4096.max_abs, boundary_4096.rms,
                           boundary_4096.nonfinite);
      boundary_4096.passed = boundary_ok;
    } else {
      fail_cuda("production_4096_vs_64_windows", error);
      boundary_4096.passed = false;
    }
    release(&split);
    release(&device);
  }

  DeviceBuffers geometry;
  cudaError_t error = allocate(&geometry, production, 4096);
  if (error != cudaSuccess) return fail_cuda("geometry cudaMalloc", error);
  std::vector<float> g_input, g_weights, g_decay, g_beta, g_conv, g_rec;
  fill_synthetic(production, 4096, &g_input, &g_weights, &g_decay, &g_beta,
                 &g_conv, &g_rec);
  error = copy_inputs(&geometry, g_input, g_weights, g_decay, g_beta, g_conv,
                      g_rec);
  if (error != cudaSuccess) {
    release(&geometry);
    return fail_cuda("geometry copy", error);
  }
  const Capture conv64 = capture([&](cudaStream_t stream) {
    return launch_path(production, geometry, 64,
                        qw38::cuda::GdnScanPath::kParallelAssociative, false,
                        stream);
  });
  const Capture conv4096 = capture([&](cudaStream_t stream) {
    return launch_path(production, geometry, 4096,
                       qw38::cuda::GdnScanPath::kParallelAssociative, false,
                       stream);
  });
  const Capture sequential4096 = capture([&](cudaStream_t stream) {
    return launch_path(production, geometry, 4096,
                       qw38::cuda::GdnScanPath::kSequentialWindows, false,
                       stream);
  });
  auto matches = [](const KernelNode& node, unsigned gx, unsigned gy,
                    unsigned bx) {
    return node.grid.x == gx && node.grid.y == gy && node.grid.z == 1 &&
           node.block.x == bx && node.block.y == 1 && node.block.z == 1;
  };
  const bool conv64_ok =
      conv64.ok && count_nodes(conv64, 40, 64, 256) == 1;
  const bool conv4096_ok =
      conv4096.ok && conv4096.kernels.size() == 4 &&
      matches(conv4096.kernels[0], 40, 4096, 256);
  const bool intra_ok =
      conv4096.ok && conv4096.kernels.size() >= 2 &&
      matches(conv4096.kernels[1], 48, 64, 128);
  const bool prefix_ok =
      conv4096.ok && conv4096.kernels.size() >= 3 &&
      matches(conv4096.kernels[2], 48, 1, 128);
  const bool from_state_ok =
      conv4096.ok && conv4096.kernels.size() >= 4 &&
      matches(conv4096.kernels[3], 48, 64, 128);
  const std::size_t sequential_recurrence =
      count_nodes(sequential4096, 48, 1, 128);
  const std::size_t sequential_convolution =
      count_nodes(sequential4096, 80, 1, 128);
  const bool sequential_geometry_ok =
      sequential4096.ok && sequential_recurrence == 64 &&
      sequential_convolution == 64;
  const bool launch_geometry_ok =
      conv64_ok && conv4096_ok && intra_ok && prefix_ok && from_state_ok &&
      sequential_geometry_ok && count_nodes(conv4096, 48, 64, 128) == 2 &&
      count_nodes(conv4096, 48, 1, 128) == 1;
  const std::size_t parallel_conv = count_nodes(conv4096, 40, 4096, 256);
  const std::size_t parallel_intra =
      intra_ok ? static_cast<std::size_t>(1) : count_nodes(conv4096, 48, 64, 128);
  const std::size_t parallel_prefix =
      prefix_ok ? static_cast<std::size_t>(1) : 0;
  const std::size_t parallel_from_state =
      from_state_ok ? static_cast<std::size_t>(1) : 0;
  const bool launch_counts_ok = parallel_conv == 1 && parallel_intra == 1 &&
                                  parallel_prefix == 1 &&
                                  parallel_from_state == 1 &&
                                  sequential_convolution == 64 &&
                                  sequential_recurrence == 64;

  auto fail_closed_case = [&](qw38::cuda::GdnScanPath path, float* scratch,
                              std::size_t scratch_floats, std::size_t tokens,
                              bool alias) {
    std::vector<float> input, weights, log_decay, beta, initial_convolution,
        initial_recurrent;
    fill_synthetic(small, 8, &input, &weights, &log_decay, &beta,
                   &initial_convolution, &initial_recurrent);
    DeviceBuffers device;
    cudaError_t alloc_error = allocate(&device, small, 8);
    if (alloc_error != cudaSuccess) return false;
    alloc_error = copy_inputs(&device, input, weights, log_decay, beta,
                             initial_convolution, initial_recurrent);
    if (alloc_error != cudaSuccess) {
      release(&device);
      return false;
    }
    qw38::cuda::GdnState committed = committed_state(device);
    qw38::cuda::GdnState candidate = candidate_state(device);
    if (alias) candidate.convolution = committed.convolution;
    const cudaError_t launched = qw38::cuda::launch_gdn_prepare_chunk(
        small, device.input, device.weights, device.log_decay, device.beta,
        tokens, committed, candidate, device.convolution_output,
        device.recurrent_output, nullptr, path, scratch, scratch_floats);
    std::vector<float> cconv(initial_convolution.size()),
        crec(initial_recurrent.size()), dummy_conv(8 * qw38::cuda::gdn_convolution_channels(small)),
        dummy_rec(8 * qw38::cuda::gdn_output_values(small));
    std::uint64_t frontier = 0;
    const cudaError_t read = read_outputs(
        device, &cconv, &crec, &dummy_conv, &dummy_rec, &cconv, &crec, &frontier);
    const bool unchanged =
        read == cudaSuccess &&
        committed_unchanged(cconv, crec, initial_convolution, initial_recurrent,
                            frontier);
    release(&device);
    return launched == cudaErrorInvalidValue && unchanged;
  };
  DeviceBuffers fail_buffers;
  error = allocate(&fail_buffers, small, 8);
  if (error != cudaSuccess) {
    release(&geometry);
    return fail_cuda("fail-closed cudaMalloc", error);
  }
  const bool fail_null_scratch = fail_closed_case(
      qw38::cuda::GdnScanPath::kParallelAssociative, nullptr,
      qw38::cuda::gdn_scan_scratch_floats(small, 8), 8, false);
  const bool fail_zero_scratch = fail_closed_case(
      qw38::cuda::GdnScanPath::kParallelAssociative, fail_buffers.scratch, 0, 8,
      false);
  const bool fail_alias = fail_closed_case(
      qw38::cuda::GdnScanPath::kParallelAssociative, fail_buffers.scratch,
      qw38::cuda::gdn_scan_scratch_floats(small, 8), 8, true);
  const bool fail_zero_tokens = fail_closed_case(
      qw38::cuda::GdnScanPath::kParallelAssociative, fail_buffers.scratch,
      qw38::cuda::gdn_scan_scratch_floats(small, 8), 0, false);
  const bool fail_invalid_path = fail_closed_case(
      static_cast<qw38::cuda::GdnScanPath>(3), fail_buffers.scratch,
      qw38::cuda::gdn_scan_scratch_floats(small, 8), 8, false);
  const bool fail_closed_ok = fail_null_scratch && fail_zero_scratch &&
                              fail_alias && fail_zero_tokens &&
                              fail_invalid_path;
  release(&fail_buffers);

  auto w_fit = [](std::size_t rows) {
    const std::size_t pair = 2 * kProductionRecurrent;
    return (rows * kFfnWidth * sizeof(__nv_bfloat16) / sizeof(float)) / pair;
  };
  const std::size_t w_fit_4096 = w_fit(4096);
  const std::size_t w_fit_65 = w_fit(65);
  const bool overlay_ok = w_fit_4096 == 22 && w_fit_65 == 0;

  std::vector<float> sequential_ms(kSamples, 0.0F);
  std::vector<float> parallel_ms(kSamples, 0.0F);
  bool speedup_ok = false;
  float sequential_mean = 0.0F;
  float parallel_mean = 0.0F;
  {
    error = restore_committed(&geometry, g_conv, g_rec);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (error == cudaSuccess) error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    auto time_path = [&](qw38::cuda::GdnScanPath path, float* milliseconds) {
      cudaError_t timed = restore_committed(&geometry, g_conv, g_rec);
      if (timed == cudaSuccess) timed = cudaEventRecord(start);
      if (timed == cudaSuccess) {
        timed = launch_path(production, geometry, 4096, path, false, nullptr);
      }
      if (timed == cudaSuccess) timed = cudaEventRecord(stop);
      if (timed == cudaSuccess) timed = cudaEventSynchronize(stop);
      if (timed == cudaSuccess) {
        timed = cudaEventElapsedTime(milliseconds, start, stop);
      }
      return timed;
    };
    for (int warmup = 0; warmup < kWarmups && error == cudaSuccess; ++warmup) {
      float ignore = 0.0F;
      error = time_path(qw38::cuda::GdnScanPath::kSequentialWindows, &ignore);
      if (error == cudaSuccess) {
        error = time_path(qw38::cuda::GdnScanPath::kParallelAssociative, &ignore);
      }
    }
    for (int sample = 0; sample < kSamples && error == cudaSuccess; ++sample) {
      const bool sequential_first = (sample % 2) == 0;
      float first_ms = 0.0F;
      float second_ms = 0.0F;
      if (sequential_first) {
        error = time_path(qw38::cuda::GdnScanPath::kSequentialWindows, &first_ms);
        if (error == cudaSuccess) {
          error = time_path(qw38::cuda::GdnScanPath::kParallelAssociative, &second_ms);
        }
        sequential_ms[static_cast<std::size_t>(sample)] = first_ms;
        parallel_ms[static_cast<std::size_t>(sample)] = second_ms;
      } else {
        error = time_path(qw38::cuda::GdnScanPath::kParallelAssociative, &first_ms);
        if (error == cudaSuccess) {
          error = time_path(qw38::cuda::GdnScanPath::kSequentialWindows, &second_ms);
        }
        parallel_ms[static_cast<std::size_t>(sample)] = first_ms;
        sequential_ms[static_cast<std::size_t>(sample)] = second_ms;
      }
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (error == cudaSuccess) {
      for (int sample = 0; sample < kSamples; ++sample) {
        sequential_mean += sequential_ms[static_cast<std::size_t>(sample)];
        parallel_mean += parallel_ms[static_cast<std::size_t>(sample)];
      }
      sequential_mean /= static_cast<float>(kSamples);
      parallel_mean /= static_cast<float>(kSamples);
      speedup_ok = parallel_mean < sequential_mean && parallel_mean > 0.0F &&
                   sequential_mean > 0.0F;
    } else {
      fail_cuda("speedup", error);
    }
  }

  float fused_mean = 0.0F;
  float fused_rank2_mean = 0.0F;
  float fused_seq_mean = 0.0F;
  float fused_par_mean = 0.0F;
  bool fused_faster = false;
  const int quality_occupancy = qw38::cuda::gdn_fused_quality_occupancy();
  Envelope opt019_gdn_2048 =
      run_prepare("opt019_gdn_2048", production, 2048, false, false, false,
                  false);
  std::printf("opt019_gdn_occupancy=%d\n", quality_occupancy);
  std::printf("opt019_gdn_2048 max_abs=%.9g rms=%.9g nonfinite=%zu "
              "prepare_atomic=%s passed=%s\n",
              opt019_gdn_2048.max_abs, opt019_gdn_2048.rms,
              opt019_gdn_2048.nonfinite,
              json_bool(opt019_gdn_2048.prepare_atomic),
              json_bool(opt019_gdn_2048.passed));
  {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    error = restore_committed(&geometry, g_conv, g_rec);
    if (error == cudaSuccess) error = cudaEventCreate(&start);
    if (error == cudaSuccess) error = cudaEventCreate(&stop);
    auto time_path2048 = [&](qw38::cuda::GdnScanPath path, float* milliseconds) {
      cudaError_t timed = restore_committed(&geometry, g_conv, g_rec);
      if (timed == cudaSuccess) timed = cudaEventRecord(start);
      if (timed == cudaSuccess) {
        timed = launch_path(production, geometry, 2048, path, false, nullptr);
      }
      if (timed == cudaSuccess) timed = cudaEventRecord(stop);
      if (timed == cudaSuccess) timed = cudaEventSynchronize(stop);
      if (timed == cudaSuccess) {
        timed = cudaEventElapsedTime(milliseconds, start, stop);
      }
      return timed;
    };
    auto time_rank2_2048 = [&](float* milliseconds) {
      cudaError_t timed = restore_committed(&geometry, g_conv, g_rec);
      if (timed == cudaSuccess) timed = cudaEventRecord(start);
      if (timed == cudaSuccess) {
        timed = qw38::cuda::launch_gdn_fused_rank2(
            production, geometry.input, geometry.weights, geometry.log_decay,
            geometry.beta, 2048, committed_state(geometry),
            candidate_state(geometry), geometry.convolution_output,
            geometry.recurrent_output, nullptr, false);
      }
      if (timed == cudaSuccess) timed = cudaEventRecord(stop);
      if (timed == cudaSuccess) timed = cudaEventSynchronize(stop);
      if (timed == cudaSuccess) {
        timed = cudaEventElapsedTime(milliseconds, start, stop);
      }
      return timed;
    };
    for (int sample = 0; sample < 3 && error == cudaSuccess; ++sample) {
      float fused_ms = 0.0F;
      float rank2_ms = 0.0F;
      float seq_ms = 0.0F;
      float par_ms = 0.0F;
      error = time_path2048(qw38::cuda::GdnScanPath::kFusedTokenLoop, &fused_ms);
      if (error == cudaSuccess) error = time_rank2_2048(&rank2_ms);
      if (error == cudaSuccess) {
        error = time_path2048(qw38::cuda::GdnScanPath::kSequentialWindows,
                              &seq_ms);
      }
      if (error == cudaSuccess) {
        error = time_path2048(qw38::cuda::GdnScanPath::kParallelAssociative,
                              &par_ms);
      }
      fused_mean += fused_ms;
      fused_rank2_mean += rank2_ms;
      fused_seq_mean += seq_ms;
      fused_par_mean += par_ms;
      std::printf("opt019_gdn_ab sample=%d quality_ms=%.9g rank2_ms=%.9g "
                  "sequential_ms=%.9g overlay_ms=%.9g\n",
                  sample, fused_ms, rank2_ms, seq_ms, par_ms);
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (error == cudaSuccess) {
      fused_mean /= 3.0F;
      fused_rank2_mean /= 3.0F;
      fused_seq_mean /= 3.0F;
      fused_par_mean /= 3.0F;
      fused_faster = fused_mean > 0.0F && fused_mean < fused_rank2_mean &&
                     fused_mean < fused_seq_mean &&
                     fused_mean < fused_par_mean && quality_occupancy >= 1;
    } else {
      fail_cuda("fused A/B", error);
    }
    std::printf("fused_ab fused_ms=%.9g rank2_ms=%.9g sequential_ms=%.9g "
                "overlay_ms=%.9g occupancy=%d faster=%s\n",
                fused_mean, fused_rank2_mean, fused_seq_mean, fused_par_mean,
                quality_occupancy, fused_faster ? "true" : "false");
  }

  bool opt029_ok = true;
  const char* fuse_ids[4] = {"off", "fuse_conv", "fuse_gate", "fuse_both"};
  float fuse_mean_ms[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  int fuse_occ[4] = {0, 0, 0, 0};
  bool fuse_eligible[4] = {false, false, false, false};
  const char* opt029_winner = "off";
  bool opt029_win = false;
  {
    mkdir("evidence", 0755);
    mkdir("evidence/optimization", 0755);
    mkdir("evidence/optimization/opt029-gdn-fuse", 0755);
    std::FILE* raw = std::fopen(
        "evidence/optimization/opt029-gdn-fuse/gdn-ab-raw.txt", "w");
    if (raw == nullptr) {
      std::fprintf(stderr, "opt029 A/B fopen failed\n");
      opt029_ok = false;
    }
    const std::size_t env_tokens = 256;
    const std::size_t env_heads = production.value_heads;
    const std::size_t env_width = production.value_width;
    const std::size_t env_gate = env_tokens * env_heads * env_width;
    DeviceBuffers env{};
    float* gate = nullptr;
    float* norm = nullptr;
    __nv_bfloat16* fused_bf16 = nullptr;
    __nv_bfloat16* split_bf16 = nullptr;
    cudaError_t env_error = allocate(&env, production, env_tokens);
    if (env_error == cudaSuccess) {
      env_error = cudaMalloc(&gate, env_gate * sizeof(float));
    }
    if (env_error == cudaSuccess) {
      env_error = cudaMalloc(&norm, env_width * sizeof(float));
    }
    if (env_error == cudaSuccess) {
      env_error = cudaMalloc(&fused_bf16, env_gate * sizeof(__nv_bfloat16));
    }
    if (env_error == cudaSuccess) {
      env_error = cudaMalloc(&split_bf16, env_gate * sizeof(__nv_bfloat16));
    }
    std::vector<float> e_input, e_weights, e_decay, e_beta, e_conv, e_rec;
    std::vector<float> host_gate(env_gate), host_norm(env_width);
    if (env_error == cudaSuccess) {
      fill_synthetic(production, env_tokens, &e_input, &e_weights, &e_decay,
                     &e_beta, &e_conv, &e_rec);
      for (std::size_t index = 0; index < env_gate; ++index) {
        host_gate[index] =
            std::sin(static_cast<float>(index) * 0.017F) * 0.5F;
      }
      for (std::size_t index = 0; index < env_width; ++index) {
        host_norm[index] = 0.75F + 0.01F * static_cast<float>(index % 17);
      }
      env_error = copy_inputs(&env, e_input, e_weights, e_decay, e_beta, e_conv,
                              e_rec);
    }
    if (env_error == cudaSuccess) {
      env_error = cudaMemcpy(gate, host_gate.data(), env_gate * sizeof(float),
                             cudaMemcpyHostToDevice);
    }
    if (env_error == cudaSuccess) {
      env_error = cudaMemcpy(norm, host_norm.data(), env_width * sizeof(float),
                             cudaMemcpyHostToDevice);
    }
    const std::size_t rec_values = qw38::cuda::gdn_output_values(production);
    const std::size_t conv_state = qw38::cuda::gdn_convolution_values(production);
    const std::size_t rec_state = qw38::cuda::gdn_recurrent_values(production);
    std::vector<float> seq_rec(env_tokens * rec_values), seq_cconv(conv_state),
        seq_crec(rec_state),
        seq_conv(env_tokens * qw38::cuda::gdn_convolution_channels(production));
    std::vector<float> dummy_cconv(conv_state), dummy_crec(rec_state);
    std::uint64_t env_frontier = 0;
    if (env_error == cudaSuccess) {
      env_error = launch_path(production, env, env_tokens,
                              qw38::cuda::GdnScanPath::kSequentialWindows, true,
                              nullptr);
    }
    if (env_error == cudaSuccess) env_error = cudaDeviceSynchronize();
    if (env_error == cudaSuccess) {
      env_error = read_outputs(env, &seq_cconv, &seq_crec, &seq_conv, &seq_rec,
                               &dummy_cconv, &dummy_crec, &env_frontier);
    }
    const bool seq_atomic =
        env_error == cudaSuccess &&
        committed_unchanged(dummy_cconv, dummy_crec, e_conv, e_rec,
                            env_frontier);
    for (int id = 0; id < 4 && env_error == cudaSuccess; ++id) {
      fuse_occ[id] = qw38::cuda::gdn_fuse_occupancy(fuse_ids[id]);
      env_error = restore_committed(&env, e_conv, e_rec);
      if (env_error == cudaSuccess) {
        env_error = qw38::cuda::launch_gdn_quality_fused(
            production, env.input, env.weights, env.log_decay, env.beta,
            env_tokens, committed_state(env), candidate_state(env),
            env.convolution_output, env.recurrent_output, gate, norm,
            fused_bf16, nullptr, true, fuse_ids[id]);
      }
      if (env_error == cudaSuccess) env_error = cudaDeviceSynchronize();
      std::vector<float> act_rec(env_tokens * rec_values), act_cconv(conv_state),
          act_crec(rec_state), act_conv(seq_conv.size());
      std::uint64_t act_frontier = 0;
      if (env_error == cudaSuccess) {
        env_error = read_outputs(env, &act_cconv, &act_crec, &act_conv, &act_rec,
                                 &dummy_cconv, &dummy_crec, &act_frontier);
      }
      Envelope fuse_env;
      if (env_error == cudaSuccess) {
        double squared = 0.0;
        std::size_t count = 0;
        accumulate(act_rec, seq_rec, &fuse_env.max_abs, &squared, &count,
                   &fuse_env.nonfinite);
        accumulate(act_cconv, seq_cconv, &fuse_env.max_abs, &squared, &count,
                   &fuse_env.nonfinite);
        accumulate(act_crec, seq_crec, &fuse_env.max_abs, &squared, &count,
                   &fuse_env.nonfinite);
        fuse_env.rms = static_cast<float>(
            std::sqrt(squared / static_cast<double>(count)));
        fuse_env.prepare_atomic =
            seq_atomic && committed_unchanged(dummy_cconv, dummy_crec, e_conv,
                                              e_rec, act_frontier);
      }
      bool gate_ok = true;
      const bool fuses_gate = id == 2 || id == 3;
      if (env_error == cudaSuccess && fuses_gate) {
        env_error = qw38::cuda::launch_gdn_gated_output_rows(
            env.recurrent_output, gate, norm, production.key_heads,
            production.value_heads / production.key_heads,
            production.value_width, env_tokens, split_bf16, nullptr);
        if (env_error == cudaSuccess) env_error = cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> host_fused(env_gate), host_split(env_gate);
        if (env_error == cudaSuccess) {
          env_error = cudaMemcpy(host_fused.data(), fused_bf16,
                                 env_gate * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost);
        }
        if (env_error == cudaSuccess) {
          env_error = cudaMemcpy(host_split.data(), split_bf16,
                                 env_gate * sizeof(__nv_bfloat16),
                                 cudaMemcpyDeviceToHost);
        }
        if (env_error == cudaSuccess) {
          double squared = 0.0;
          std::size_t count = 0;
          float max_abs = 0.0F;
          std::size_t nonfinite = 0;
          for (std::size_t index = 0; index < env_gate; ++index) {
            const float actual = __bfloat162float(host_fused[index]);
            const float expected = __bfloat162float(host_split[index]);
            if (!std::isfinite(actual) || !std::isfinite(expected)) ++nonfinite;
            const float difference = std::fabs(actual - expected);
            max_abs = std::max(max_abs, difference);
            squared += static_cast<double>(difference) * difference;
            ++count;
          }
          const float rms = static_cast<float>(
              std::sqrt(squared / static_cast<double>(count)));
          gate_ok = within_envelope(max_abs, rms, nonfinite);
          std::printf("opt029_gate id=%s max_abs=%.9g rms=%.9g nonfinite=%zu "
                      "ok=%s\n",
                      fuse_ids[id], max_abs, rms, nonfinite,
                      json_bool(gate_ok));
        }
      }
      const bool numeric_ok =
          env_error == cudaSuccess && fuse_env.prepare_atomic &&
          within_envelope(fuse_env.max_abs, fuse_env.rms, fuse_env.nonfinite) &&
          gate_ok;
      fuse_eligible[id] = numeric_ok && fuse_occ[id] >= 1;
      std::printf("opt029_env id=%s occupancy=%d max_abs=%.9g rms=%.9g "
                  "nonfinite=%zu prepare_atomic=%s eligible=%s\n",
                  fuse_ids[id], fuse_occ[id], fuse_env.max_abs, fuse_env.rms,
                  fuse_env.nonfinite, json_bool(fuse_env.prepare_atomic),
                  json_bool(fuse_eligible[id]));
      if (raw != nullptr) {
        std::fprintf(raw,
                     "opt029_env id=%s occupancy=%d max_abs=%.9g rms=%.9g "
                     "nonfinite=%zu prepare_atomic=%s eligible=%s\n",
                     fuse_ids[id], fuse_occ[id], fuse_env.max_abs, fuse_env.rms,
                     fuse_env.nonfinite, json_bool(fuse_env.prepare_atomic),
                     json_bool(fuse_eligible[id]));
      }
    }
    cudaFree(split_bf16);
    cudaFree(fused_bf16);
    cudaFree(norm);
    cudaFree(gate);
    release(&env);
    if (env_error != cudaSuccess) {
      fail_cuda("opt029 envelope", env_error);
      opt029_ok = false;
    }

    const std::size_t ab_tokens = 4096;
    const std::size_t ab_gate =
        ab_tokens * production.value_heads * production.value_width;
    float* ab_gate_buf = nullptr;
    float* ab_norm = nullptr;
    __nv_bfloat16* ab_bf16 = nullptr;
    cudaError_t ab_error = cudaSuccess;
    if (opt029_ok) {
      ab_error = cudaMalloc(&ab_gate_buf, ab_gate * sizeof(float));
      if (ab_error == cudaSuccess) {
        ab_error = cudaMalloc(&ab_norm, production.value_width * sizeof(float));
      }
      if (ab_error == cudaSuccess) {
        ab_error = cudaMalloc(&ab_bf16, ab_gate * sizeof(__nv_bfloat16));
      }
    }
    std::vector<float> host_ab_gate(ab_gate), host_ab_norm(production.value_width);
    if (ab_error == cudaSuccess) {
      for (std::size_t index = 0; index < ab_gate; ++index) {
        host_ab_gate[index] =
            std::sin(static_cast<float>(index) * 0.017F) * 0.5F;
      }
      for (std::size_t index = 0; index < production.value_width; ++index) {
        host_ab_norm[index] = 0.75F + 0.01F * static_cast<float>(index % 17);
      }
      ab_error = cudaMemcpy(ab_gate_buf, host_ab_gate.data(),
                            ab_gate * sizeof(float), cudaMemcpyHostToDevice);
      if (ab_error == cudaSuccess) {
        ab_error = cudaMemcpy(ab_norm, host_ab_norm.data(),
                              production.value_width * sizeof(float),
                              cudaMemcpyHostToDevice);
      }
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (ab_error == cudaSuccess) ab_error = cudaEventCreate(&start);
    if (ab_error == cudaSuccess) ab_error = cudaEventCreate(&stop);
    for (int id = 0; id < 4 && ab_error == cudaSuccess && opt029_ok; ++id) {
      if (fuse_occ[id] < 1) continue;
      float samples[3] = {0.0F, 0.0F, 0.0F};
      for (int sample = 0; sample < 3 && ab_error == cudaSuccess; ++sample) {
        ab_error = restore_committed(&geometry, g_conv, g_rec);
        if (ab_error == cudaSuccess) ab_error = cudaEventRecord(start);
        if (ab_error == cudaSuccess) {
          ab_error = qw38::cuda::launch_gdn_quality_fused(
              production, geometry.input, geometry.weights, geometry.log_decay,
              geometry.beta, ab_tokens, committed_state(geometry),
              candidate_state(geometry), geometry.convolution_output,
              geometry.recurrent_output, ab_gate_buf, ab_norm, ab_bf16, nullptr,
              true, fuse_ids[id]);
        }
        if (ab_error == cudaSuccess) ab_error = cudaEventRecord(stop);
        if (ab_error == cudaSuccess) ab_error = cudaEventSynchronize(stop);
        if (ab_error == cudaSuccess) {
          ab_error = cudaEventElapsedTime(&samples[sample], start, stop);
        }
        if (raw != nullptr && ab_error == cudaSuccess) {
          std::fprintf(raw, "gdn_ab id=%s sample=%d ms=%.9g occupancy=%d\n",
                       fuse_ids[id], sample, samples[sample], fuse_occ[id]);
        }
        std::printf("gdn_ab id=%s sample=%d ms=%.9g occupancy=%d\n", fuse_ids[id],
                    sample, samples[sample], fuse_occ[id]);
      }
      if (ab_error == cudaSuccess) {
        fuse_mean_ms[id] = (samples[0] + samples[1] + samples[2]) / 3.0F;
        if (raw != nullptr) {
          std::fprintf(raw,
                       "gdn_ab_mean id=%s mean_ms=%.9g occupancy=%d "
                       "eligible=%s\n",
                       fuse_ids[id], fuse_mean_ms[id], fuse_occ[id],
                       json_bool(fuse_eligible[id]));
        }
        std::printf("gdn_ab_mean id=%s mean_ms=%.9g occupancy=%d eligible=%s\n",
                    fuse_ids[id], fuse_mean_ms[id], fuse_occ[id],
                    json_bool(fuse_eligible[id]));
      }
    }
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    cudaFree(ab_bf16);
    cudaFree(ab_norm);
    cudaFree(ab_gate_buf);
    if (ab_error != cudaSuccess) {
      fail_cuda("opt029 A/B", ab_error);
      opt029_ok = false;
    }
    if (opt029_ok && !fuse_eligible[0]) {
      std::fprintf(stderr, "opt029 baseline off is not GDN-002 eligible\n");
      opt029_ok = false;
    }
    float best_ms = fuse_mean_ms[0];
    opt029_winner = "off";
    opt029_win = false;
    if (opt029_ok && fuse_mean_ms[0] > 0.0F) {
      for (int id = 1; id < 4; ++id) {
        if (!fuse_eligible[id] || fuse_mean_ms[id] <= 0.0F) continue;
        if (fuse_mean_ms[id] < fuse_mean_ms[0] &&
            (!opt029_win || fuse_mean_ms[id] < best_ms)) {
          best_ms = fuse_mean_ms[id];
          opt029_winner = fuse_ids[id];
          opt029_win = true;
        }
      }
    }
    if (raw != nullptr) {
      std::fprintf(raw,
                   "gdn_ab_winner id=%s mean_ms=%.9g baseline_id=off "
                   "baseline_ms=%.9g win=%s selected_path=%s\n",
                   opt029_winner, opt029_win ? best_ms : fuse_mean_ms[0],
                   fuse_mean_ms[0], json_bool(opt029_win),
                   qw38::cuda::selected_gdn_fuse_path());
      std::fclose(raw);
    }
    std::printf("gdn_ab_winner id=%s mean_ms=%.9g baseline_id=off "
                "baseline_ms=%.9g win=%s selected_path=%s\n",
                opt029_winner, opt029_win ? best_ms : fuse_mean_ms[0],
                fuse_mean_ms[0], json_bool(opt029_win),
                qw38::cuda::selected_gdn_fuse_path());
  }
  release(&geometry);

  const bool small_65_ok = parallel_cases.size() > 2 && parallel_cases[2].passed;
  const bool production_65_ok =
      parallel_cases.size() > 5 && parallel_cases[5].passed;
  const bool production_129_ok =
      parallel_cases.size() > 6 && parallel_cases[6].passed;
  const bool chunk_boundaries_ok =
      small_65_ok && production_65_ok && production_129_ok && boundary_ok;

  cudaDeviceProp prop{};
  int device = 0;
  int driver = 0;
  int runtime = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::time_t now = std::time(nullptr);
  char utc[32]{};
  std::tm tm{};
  gmtime_r(&now, &tm);
  std::strftime(utc, sizeof(utc), "%Y-%m-%dT%H:%M:%SZ", &tm);

  auto print_envelope_array = [](const std::vector<Envelope>& cases) {
    std::printf("[");
    for (std::size_t index = 0; index < cases.size(); ++index) {
      const Envelope& envelope = cases[index];
      if (index != 0) std::printf(",");
      std::printf("{\"name\":\"%s\",\"tokens\":%zu,\"max_abs\":%.9g,"
                  "\"rms\":%.9g,\"nonfinite\":%zu,\"prepare_atomic\":%s,"
                  "\"tokenwise_equal\":%s,\"conv_memcmp\":%s,\"vs_scalar\":%s,"
                  "\"passed\":%s}",
                  envelope.name, envelope.tokens, envelope.max_abs,
                  envelope.rms, envelope.nonfinite,
                  json_bool(envelope.prepare_atomic),
                  json_bool(envelope.tokenwise_equal),
                  json_bool(envelope.conv_memcmp), json_bool(envelope.vs_scalar),
                  json_bool(envelope.passed));
    }
    std::printf("]");
  };

  std::printf(
      "QW38_GDN_SCAN_RESULT={\"schema_version\":1,\"task\":\"OPT-013\","
      "\"status\":\"measured\",\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
      "\"driver\":\"%d.%d\",\"runtime\":\"%d.%d\",\"toolkit\":\"CUDA 13.0.2\","
      "\"pinned_image\":\"qw38-cuda:13.0.2\",\"measurement_utc\":\"%s\","
      "\"semantic\":{"
      "\"sequential_regression\":%s,"
      "\"parallel_vs_sequential\":%s,"
      "\"chunk_boundaries\":%s,"
      "\"tiled_layout\":%s,"
      "\"launch_geometry\":%s,"
      "\"fail_closed\":%s,"
      "\"overlay_capacity\":%s,"
      "\"speedup\":%s,"
      "\"launch_counts\":%s},",
      prop.name, prop.major, prop.minor, driver / 1000, (driver % 1000) / 10,
      runtime / 1000, (runtime % 1000) / 10, utc, json_bool(sequential_ok),
      json_bool(parallel_ok), json_bool(chunk_boundaries_ok), json_bool(tiled_ok),
      json_bool(launch_geometry_ok), json_bool(fail_closed_ok),
      json_bool(overlay_ok), json_bool(speedup_ok), json_bool(launch_counts_ok));
  std::printf("\"sequential_regression\":");
  print_envelope_array(sequential_cases);
  std::printf(",\"parallel_vs_sequential\":");
  print_envelope_array(parallel_cases);
  std::printf(
      ",\"chunk_boundaries\":{\"production_65\":%s,\"production_129\":%s,"
      "\"production_4096_vs_64_windows\":{\"max_abs\":%.9g,\"rms\":%.9g,"
      "\"nonfinite\":%zu,\"prepare_atomic\":%s,\"passed\":%s}},"
      "\"tiled\":{\"name\":\"%s\",\"tokens\":%zu,\"max_abs\":%.9g,\"rms\":%.9g,"
      "\"nonfinite\":%zu,\"prepare_atomic\":%s,\"passed\":%s},"
      "\"launch\":{\"conv_64\":{\"kernel_nodes\":%zu,\"grid\":[40,64,1],"
      "\"block\":[256,1,1]},\"conv_4096\":{\"kernel_nodes\":%zu,"
      "\"grid\":[40,4096,1],\"block\":[256,1,1]},"
      "\"intra_4096\":{\"kernel_nodes\":%zu,\"grid\":[48,64,1],"
      "\"block\":[128,1,1]},\"prefix_4096\":{\"kernel_nodes\":%zu,"
      "\"grid\":[48,1,1],\"block\":[128,1,1]},"
      "\"from_state_4096\":{\"kernel_nodes\":%zu,\"grid\":[48,64,1],"
      "\"block\":[128,1,1]},\"sequential_4096_recurrence_nodes\":%zu},"
      "\"fail_closed\":{\"null_scratch\":%s,\"zero_scratch_floats\":%s,"
      "\"aliased\":%s,\"token_count_0\":%s,\"invalid_path\":%s},"
      "\"overlay\":{\"w_fit_4096\":%zu,\"w_fit_65\":%zu},"
      "\"speedup\":{\"warmups\":%d,\"samples\":%d,\"sequential_mean_ms\":%.9g,"
      "\"parallel_mean_ms\":%.9g,\"parallel_below_sequential\":%s,"
      "\"sequential_ms\":[",
      json_bool(production_65_ok), json_bool(production_129_ok),
      boundary_4096.max_abs, boundary_4096.rms, boundary_4096.nonfinite,
      json_bool(boundary_4096.prepare_atomic), json_bool(boundary_4096.passed),
      tiled_case.name, tiled_case.tokens, tiled_case.max_abs, tiled_case.rms,
      tiled_case.nonfinite, json_bool(tiled_case.prepare_atomic),
      json_bool(tiled_case.passed), count_nodes(conv64, 40, 64, 256),
      count_nodes(conv4096, 40, 4096, 256),
      intra_ok ? static_cast<std::size_t>(1) : 0,
      prefix_ok ? static_cast<std::size_t>(1) : 0,
      from_state_ok ? static_cast<std::size_t>(1) : 0, sequential_recurrence,
      json_bool(fail_null_scratch),
      json_bool(fail_zero_scratch), json_bool(fail_alias),
      json_bool(fail_zero_tokens), json_bool(fail_invalid_path), w_fit_4096,
      w_fit_65, kWarmups, kSamples, sequential_mean, parallel_mean,
      json_bool(speedup_ok));
  for (int sample = 0; sample < kSamples; ++sample) {
    if (sample != 0) std::printf(",");
    std::printf("%.9g", sequential_ms[static_cast<std::size_t>(sample)]);
  }
  std::printf("],\"parallel_ms\":[");
  for (int sample = 0; sample < kSamples; ++sample) {
    if (sample != 0) std::printf(",");
    std::printf("%.9g", parallel_ms[static_cast<std::size_t>(sample)]);
  }
  std::printf(
      "]},\"launch_counts\":{\"parallel_convolution\":%zu,\"parallel_intra\":%zu,"
      "\"parallel_prefix\":%zu,\"parallel_from_state\":%zu,"
      "\"sequential_convolution\":%zu,\"sequential_recurrence\":%zu},"
      "\"proof_limit\":\"component-only GDN-prepare evidence; sequential GDN-002 "
      "windows remain the byte-exact reference; Nsight Systems is not claimed; "
      "end-to-end prefill/decode speedup and 128K quality are not claimed\"}\n",
      parallel_conv, parallel_intra, parallel_prefix, parallel_from_state,
      sequential_convolution, sequential_recurrence);

  const bool passed = sequential_ok && parallel_ok && chunk_boundaries_ok &&
                       tiled_ok && launch_geometry_ok && fail_closed_ok &&
                       overlay_ok && speedup_ok && launch_counts_ok &&
                       fused_faster && opt019_gdn_2048.passed &&
                       quality_occupancy >= 1 && opt029_ok;
  return passed ? 0 : 1;
}
