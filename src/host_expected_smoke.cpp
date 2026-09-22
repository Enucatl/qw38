#include <expected>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

static_assert(__cplusplus >= 202302L, "host compilation must use C++23");

namespace {

enum class SmokeError { non_positive, overflow };

std::expected<int, SmokeError> require_positive(int value) {
  if (value <= 0) {
    return std::unexpected(SmokeError::non_positive);
  }
  return value;
}

std::expected<int, SmokeError> add_checked(int a, int b) {
  const auto left = require_positive(a);
  if (!left) {
    return std::unexpected(left.error());
  }
  const auto right = require_positive(b);
  if (!right) {
    return std::unexpected(right.error());
  }
  if (*left > std::numeric_limits<int>::max() - *right) {
    return std::unexpected(SmokeError::overflow);
  }
  return *left + *right;
}

bool expect_error(std::expected<int, SmokeError> const& result, SmokeError want,
                  std::string_view what) {
  if (result) {
    std::cerr << what << ": expected error, got " << *result << '\n';
    return false;
  }
  if (result.error() != want) {
    std::cerr << what << ": unexpected error enumerator\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  const auto ok = add_checked(20, 22);
  if (!ok || *ok != 42) {
    std::cerr << "std::expected success path failed\n";
    return 1;
  }
  if (!expect_error(add_checked(-1, 2), SmokeError::non_positive, "left") ||
      !expect_error(add_checked(2, 0), SmokeError::non_positive, "right") ||
      !expect_error(add_checked(std::numeric_limits<int>::max(), 1),
                    SmokeError::overflow, "overflow")) {
    return 1;
  }
  std::cout << "host std::expected smoke ok\n";
  return 0;
}
