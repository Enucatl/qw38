#ifndef QW38_CUDA_PRODUCTION_NUMERICS_H_
#define QW38_CUDA_PRODUCTION_NUMERICS_H_

#include <cstring>

namespace qw38::cuda {

// OPT-044: two arithmetic roles. Production currently selects the strict path.
// Optimized kernels are not admitted until a later task validates them against
// pins/production_numerics_contract.json. Unrepresented family/shapes stay
// on the strict path.
constexpr char kLegalProductionNumericsPathStrict[] = "strict";
constexpr char kLegalProductionNumericsPathOptimized[] = "optimized";

inline bool legal_production_numerics_path(const char* path) noexcept {
  return path != nullptr &&
         (std::strcmp(path, kLegalProductionNumericsPathStrict) == 0 ||
          std::strcmp(path, kLegalProductionNumericsPathOptimized) == 0);
}

const char* selected_production_numerics_path() noexcept;
bool production_numerics_optimized_admitted() noexcept;
const char* production_numerics_path_for_family(const char* family) noexcept;

}  // namespace qw38::cuda

#endif  // QW38_CUDA_PRODUCTION_NUMERICS_H_
