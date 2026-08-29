#ifndef QW38_STATUS_H_
#define QW38_STATUS_H_

#include <string>

namespace qw38 {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kIoError,
  kIncompatibleArtifact,
  kResourceExhausted,
  kCancelled,
  kUnimplemented,
  kInternal,
};

class Status final {
 public:
  Status() noexcept = default;
  Status(StatusCode code, std::string message) noexcept;

  static Status ok() noexcept;
  bool is_ok() const noexcept;
  StatusCode code() const noexcept;
  const std::string& message() const noexcept;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

const char* status_code_name(StatusCode code) noexcept;

}  // namespace qw38

#endif  // QW38_STATUS_H_
