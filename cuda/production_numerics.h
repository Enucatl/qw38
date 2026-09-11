#ifndef QW38_CUDA_PRODUCTION_NUMERICS_H_
#define QW38_CUDA_PRODUCTION_NUMERICS_H_

#include <cstddef>
#include <cstring>

namespace qw38::cuda {

// OPT-044/OPT-059: two arithmetic roles plus an engine-wide summary.
// Unrepresented family/shapes stay on the strict path. Per-family dispatch
// is the admission table in quant_mmv.cu, not a global always-false stub.
constexpr char kLegalProductionNumericsPathStrict[] = "strict";
constexpr char kLegalProductionNumericsPathOptimized[] = "optimized";
constexpr char kLegalProductionNumericsPathMixed[] = "mixed";

constexpr char kStagingQ8Fp32[] = "q8_fp32";
constexpr char kStagingQuartzQ81SumQ[] = "quartz_q8_1_sum_q";
constexpr char kStagingLlamaQ81SumX[] = "llama_q8_1_sum_x";
constexpr char kStagingBf16Direct[] = "bf16_direct";

struct ProductionAdmissionEntry final {
  const char* id;
  const char* family;
  std::size_t columns;  // 0 = any
  const char* staging;
  const char* variant;
  const char* production_dispatch;
  bool currently_selected;
  bool v2_admitted;
  bool testing_admitted;
};

inline bool legal_production_numerics_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalProductionNumericsPathStrict) == 0 ||
          std::strcmp(path, kLegalProductionNumericsPathOptimized) == 0);
}

inline bool legal_production_numerics_summary(const char* path) noexcept {
  return legal_production_numerics_path(path) ||
         (path != nullptr &&
          std::strcmp(path, kLegalProductionNumericsPathMixed) == 0);
}

const char* selected_production_numerics_path() noexcept;
const char* unrepresented_production_numerics_path() noexcept;
bool production_numerics_optimized_admitted() noexcept;
const char* production_numerics_path_for_family(const char* family) noexcept;
const char* production_numerics_path_for(const char* family, std::size_t columns,
                                         const char* staging,
                                         const char* variant) noexcept;
bool production_numerics_testing_admitted(const char* family,
                                          std::size_t columns,
                                          const char* staging,
                                          const char* variant) noexcept;
bool production_numerics_v2_admitted(const char* family, std::size_t columns,
                                     const char* staging,
                                     const char* variant) noexcept;
std::size_t production_numerics_admission_count() noexcept;
const ProductionAdmissionEntry* production_numerics_admission_entry(
    std::size_t index) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_PRODUCTION_NUMERICS_H_
