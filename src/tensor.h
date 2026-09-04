#ifndef QW38_TENSOR_H_
#define QW38_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "model.h"
#include "qw38/status.h"

namespace qw38::internal {

struct TensorView final {
  const std::uint8_t* data = nullptr;
  std::size_t storage_bytes = 0;
  std::uint32_t type = 0;
  std::size_t columns = 0;
  std::size_t rows = 0;
  std::size_t row_bytes = 0;
};

struct VectorView final {
  const std::uint8_t* data = nullptr;
  std::size_t storage_bytes = 0;
  std::uint32_t type = 0;
  std::size_t count = 0;
};

Status make_tensor_view(const std::uint8_t* data, std::size_t storage_bytes,
                        std::uint32_t type, std::size_t columns,
                        std::size_t rows, TensorView* view) noexcept;
Status bind_tensor_view(const ModelInfo& info, const MappedFile& mapping,
                        const std::string& name, TensorView* view) noexcept;
Status make_vector_view(const std::uint8_t* data, std::size_t storage_bytes,
                        std::uint32_t type, std::size_t count,
                        VectorView* view) noexcept;
Status bind_vector_view(const ModelInfo& info, const MappedFile& mapping,
                        const std::string& name, VectorView* view) noexcept;
Status vector_decode(const VectorView& view, float* output,
                     std::size_t output_count) noexcept;
Status tensor_row_decode(const TensorView& view, std::size_t row, float* output,
                         std::size_t output_count) noexcept;
Status tensor_row_dot(const TensorView& view, std::size_t row,
                      const float* activation, std::size_t activation_count,
                      float* output) noexcept;
Status tensor_matvec(const TensorView& view, const float* activation,
                     std::size_t activation_count, float* output,
                     std::size_t output_count) noexcept;
std::size_t matvec_worker_count() noexcept;

}  // namespace qw38::internal

#endif  // QW38_TENSOR_H_
