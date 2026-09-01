#ifndef QW38_SHA256_H_
#define QW38_SHA256_H_

#include <cstddef>
#include <string>

#include "qw38/status.h"

namespace qw38::internal {

Status sha256_file(const std::string& path, std::string* digest) noexcept;
Status sha256_file_prefix(const std::string& path, std::size_t bytes,
                          std::string* digest) noexcept;
Status sha256_bytes(const unsigned char* data, std::size_t size,
                    std::string* digest) noexcept;
const char* sha256_backend_name() noexcept;

}  // namespace qw38::internal

#endif  // QW38_SHA256_H_
