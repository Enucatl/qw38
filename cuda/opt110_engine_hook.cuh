#pragma once

// OPT-110 diagnostic engine hook. Function pointers are null unless the
// llama-flag adapter object is linked. Production TUs never call these.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace qw38::cuda::opt110 {

using EngineGateUpFn = cudaError_t (*)(const std::uint8_t* gate,
                                       const std::uint8_t* up,
                                       std::size_t rows, std::size_t columns,
                                       const void* activation, void* q81,
                                       void* output, cudaStream_t stream);

using EngineDownFn = cudaError_t (*)(const std::uint8_t* weights,
                                     std::size_t rows, std::size_t columns,
                                     const void* activation, void* q81,
                                     float* output, cudaStream_t stream);

inline EngineGateUpFn g_engine_gate_up = nullptr;
inline EngineDownFn g_engine_down = nullptr;

inline bool engine_hooks_ready() noexcept {
  return g_engine_gate_up != nullptr && g_engine_down != nullptr;
}

}  // namespace qw38::cuda::opt110
