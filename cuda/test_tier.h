#pragma once

#include <cstdlib>
#include <cstring>

namespace qw38::cuda {

enum class TestTier { kInvalid, kSmoke, kCorrectness, kAcceptance };

inline TestTier test_tier() {
  const char* value = std::getenv("QW38_CUDA_TEST_TIER");
  if (value != nullptr && std::strcmp(value, "acceptance") == 0) {
    return TestTier::kAcceptance;
  }
  if (value != nullptr && std::strcmp(value, "correctness") == 0) {
    return TestTier::kCorrectness;
  }
  if (value != nullptr && std::strcmp(value, "smoke") == 0) {
    return TestTier::kSmoke;
  }
  return TestTier::kInvalid;
}

inline bool test_tier_valid() { return test_tier() != TestTier::kInvalid; }

inline const char* test_tier_name() {
  switch (test_tier()) {
    case TestTier::kInvalid:
      return "invalid";
    case TestTier::kAcceptance:
      return "acceptance";
    case TestTier::kCorrectness:
      return "correctness";
    case TestTier::kSmoke:
      return "smoke";
  }
  return "smoke";
}

inline int test_warmups() {
  switch (test_tier()) {
    case TestTier::kInvalid:
      return 0;
    case TestTier::kAcceptance:
      return 3;
    case TestTier::kCorrectness:
      return 1;
    case TestTier::kSmoke:
      return 0;
  }
  return 0;
}

inline int test_samples() {
  switch (test_tier()) {
    case TestTier::kInvalid:
      return 0;
    case TestTier::kAcceptance:
      return 30;
    case TestTier::kCorrectness:
      return 3;
    case TestTier::kSmoke:
      return 1;
  }
  return 1;
}

}  // namespace qw38::cuda
