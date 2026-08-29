#include "tensor.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "quant.h"

namespace qw38::internal {
namespace {

bool format_layout(std::uint32_t type, std::size_t* block_values,
                   std::size_t* block_bytes) noexcept {
  switch (type) {
    case 0:
      *block_values = 1;
      *block_bytes = 4;
      return true;
    case 8:
      *block_values = kQ80BlockValues;
      *block_bytes = kQ80BlockBytes;
      return true;
    case 12:
      *block_values = kQuantBlockValues;
      *block_bytes = kQ4KBlockBytes;
      return true;
    case 14:
      *block_values = kQuantBlockValues;
      *block_bytes = kQ6KBlockBytes;
      return true;
    default:
      return false;
  }
}

float read_f32_le(const std::uint8_t* bytes) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                             (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                             (static_cast<std::uint32_t>(bytes[3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Status validate_view(const TensorView& view) noexcept {
  if (view.data == nullptr || view.columns == 0 || view.rows == 0 ||
      view.row_bytes == 0) {
    return {StatusCode::kInvalidArgument, "tensor view is empty"};
  }
  if (view.rows > std::numeric_limits<std::size_t>::max() / view.row_bytes ||
      view.rows * view.row_bytes != view.storage_bytes) {
    return {StatusCode::kInvalidArgument,
            "tensor view storage does not equal its complete rows"};
  }
  return Status::ok();
}

}  // namespace

Status make_tensor_view(const std::uint8_t* data, std::size_t storage_bytes,
                        std::uint32_t type, std::size_t columns,
                        std::size_t rows, TensorView* view) noexcept {
  if (view == nullptr) {
    return {StatusCode::kInvalidArgument, "tensor view output is required"};
  }
  std::size_t block_values = 0;
  std::size_t block_bytes = 0;
  if (data == nullptr || columns == 0 || rows == 0 ||
      !format_layout(type, &block_values, &block_bytes) ||
      columns % block_values != 0 ||
      columns / block_values >
          std::numeric_limits<std::size_t>::max() / block_bytes) {
    return {StatusCode::kInvalidArgument,
            "invalid tensor format, dimensions, or block-aligned row width"};
  }
  const std::size_t row_bytes = columns / block_values * block_bytes;
  if (rows > std::numeric_limits<std::size_t>::max() / row_bytes ||
      rows * row_bytes != storage_bytes) {
    return {StatusCode::kInvalidArgument,
            "tensor storage does not contain exactly the declared rows"};
  }
  *view = {data, storage_bytes, type, columns, rows, row_bytes};
  return Status::ok();
}

Status bind_tensor_view(const ModelInfo& info, const MappedFile& mapping,
                        const std::string& name, TensorView* view) noexcept {
  const auto match = std::find_if(
      info.tensors.begin(), info.tensors.end(), [&name](const TensorInfo& tensor) {
        return tensor.name == name;
      });
  if (match == info.tensors.end()) {
    return {StatusCode::kInvalidArgument, "tensor name is not admitted"};
  }
  if (match->dimensions.size() != 2 ||
      match->dimensions[0] > std::numeric_limits<std::size_t>::max() ||
      match->dimensions[1] > std::numeric_limits<std::size_t>::max() ||
      match->storage_bytes > std::numeric_limits<std::size_t>::max() ||
      match->offset > std::numeric_limits<std::size_t>::max() ||
      info.data_offset > std::numeric_limits<std::size_t>::max()) {
    return {StatusCode::kInvalidArgument,
            "tensor is not a supported two-dimensional host matrix"};
  }
  const std::size_t relative = static_cast<std::size_t>(match->offset);
  const std::size_t data_offset = static_cast<std::size_t>(info.data_offset);
  const std::size_t bytes = static_cast<std::size_t>(match->storage_bytes);
  if (relative > std::numeric_limits<std::size_t>::max() - data_offset) {
    return {StatusCode::kInvalidArgument, "tensor absolute offset overflows"};
  }
  const std::size_t absolute = data_offset + relative;
  if (absolute > mapping.size() || bytes > mapping.size() - absolute) {
    return {StatusCode::kInvalidArgument,
            "tensor payload exceeds the mapped artifact"};
  }
  return make_tensor_view(mapping.data() + absolute, bytes, match->type,
                          static_cast<std::size_t>(match->dimensions[0]),
                          static_cast<std::size_t>(match->dimensions[1]), view);
}

Status tensor_row_decode(const TensorView& view, std::size_t row, float* output,
                         std::size_t output_count) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (output == nullptr || output_count != view.columns || row >= view.rows) {
    return {StatusCode::kInvalidArgument,
            "tensor decode row or output count is out of range"};
  }
  const std::uint8_t* row_data = view.data + row * view.row_bytes;
  if (view.type == 0) {
    for (std::size_t column = 0; column < view.columns; ++column) {
      output[column] = read_f32_le(row_data + column * 4);
    }
    return Status::ok();
  }
  const std::size_t block_values =
      view.type == 8 ? kQ80BlockValues : kQuantBlockValues;
  const std::size_t block_bytes = view.type == 8   ? kQ80BlockBytes
                                  : view.type == 12 ? kQ4KBlockBytes
                                                  : kQ6KBlockBytes;
  for (std::size_t column = 0; column < view.columns;
       column += block_values) {
    const std::uint8_t* block =
        row_data + column / block_values * block_bytes;
    if (view.type == 8) {
      status = decode_q8_0(block, block_bytes, output + column, block_values);
    } else if (view.type == 12) {
      status = decode_q4_k(block, block_bytes, output + column, block_values);
    } else {
      status = decode_q6_k(block, block_bytes, output + column, block_values);
    }
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

Status tensor_row_dot(const TensorView& view, std::size_t row,
                      const float* activation, std::size_t activation_count,
                      float* output) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (activation == nullptr || output == nullptr ||
      activation_count != view.columns || row >= view.rows) {
    return {StatusCode::kInvalidArgument,
            "tensor dot row or activation count is out of range"};
  }
  const std::uint8_t* row_data = view.data + row * view.row_bytes;
  float total = 0.0F;
  if (view.type == 0) {
    for (std::size_t column = 0; column < view.columns; ++column) {
      total += read_f32_le(row_data + column * 4) * activation[column];
    }
    *output = total;
    return Status::ok();
  }
  const std::size_t block_values =
      view.type == 8 ? kQ80BlockValues : kQuantBlockValues;
  const std::size_t block_bytes = view.type == 8   ? kQ80BlockBytes
                                  : view.type == 12 ? kQ4KBlockBytes
                                                  : kQ6KBlockBytes;
  for (std::size_t column = 0; column < view.columns;
       column += block_values) {
    const std::uint8_t* block =
        row_data + column / block_values * block_bytes;
    float partial = 0.0F;
    if (view.type == 8) {
      status = dot_q8_0(block, block_bytes, activation + column, block_values,
                        &partial);
    } else if (view.type == 12) {
      status = dot_q4_k(block, block_bytes, activation + column, block_values,
                        &partial);
    } else {
      status = dot_q6_k(block, block_bytes, activation + column, block_values,
                        &partial);
    }
    if (!status.is_ok()) return status;
    total += partial;
  }
  *output = total;
  return Status::ok();
}

Status tensor_matvec(const TensorView& view, const float* activation,
                     std::size_t activation_count, float* output,
                     std::size_t output_count) noexcept {
  Status status = validate_view(view);
  if (!status.is_ok()) return status;
  if (output == nullptr || output_count != view.rows) {
    return {StatusCode::kInvalidArgument,
            "matvec output count does not equal tensor rows"};
  }
  for (std::size_t row = 0; row < view.rows; ++row) {
    status = tensor_row_dot(view, row, activation, activation_count,
                            output + row);
    if (!status.is_ok()) return status;
  }
  return Status::ok();
}

}  // namespace qw38::internal
