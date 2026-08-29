#ifndef QW38_SHA256_H_
#define QW38_SHA256_H_

#include <string>

#include "qw38/status.h"

namespace qw38::internal {

Status sha256_file(const std::string& path, std::string* digest) noexcept;

}  // namespace qw38::internal

#endif  // QW38_SHA256_H_
