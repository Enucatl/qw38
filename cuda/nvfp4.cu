#include "cuda/nvfp4.hpp"
#include "cuda/nvfp4_device.cuh"
#include "cuda/decode_mmv.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace qw38::cuda {
namespace {
using namespace cute;
using Operand = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Tile = Shape<_128,_128,_128>;
using Cluster = Shape<_1,_1,_1>;
using Arch = cutlass::arch::Sm120;
using Op = cutlass::arch::OpClassBlockScaledTensorOp;
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    Arch, Op, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, Row, 4, float, Row, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    Arch, Op, Operand, Row, 32, Operand, Col, 32, float, Tile, Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Sf = typename Mainloop::Sm1xxBlkScaledConfig;
std::expected<void, Error> status(cutlass::Status s) {
  if (s == cutlass::Status::kSuccess) return {};
  return std::unexpected(make_error(ErrorCode::Status, "nvfp4.gemm",
                                     cutlass::cutlassGetStatusString(s)));
}
__global__ void pack_kernel(std::uint16_t const* input, std::uint8_t* codes,
                            std::uint8_t* scales, unsigned m, unsigned k, unsigned pk) {
  bool valid = blockIdx.x < m;
  nvfp4_pack_row(valid ? input + static_cast<std::size_t>(blockIdx.x) * k : nullptr,
                 codes, scales, blockIdx.x, k, pk, valid);
}
}
std::expected<void, Error> launch_pack_nvfp4(std::uint16_t const* input,
    std::uint8_t* codes, std::uint8_t* scales, std::uint32_t m, std::uint32_t k,
    Stream const& stream) {
  if (!input || !codes || !scales || !m || m > 1024 || !k || k > 17408 || stream.empty())
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "nvfp4.pack", "invalid operands"));
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  pack_kernel<<<((m + 127) / 128) * 128,128,0,stream.native()>>>(input,codes,scales,m,k,decode_pad_k(k));
  return check(cudaGetLastError(), "nvfp4.pack");
}
std::expected<void, Error> nvfp4_gemm_tile(void const* codes, void const* scales,
    std::uint8_t const* activation, std::uint8_t const* activation_scales,
    std::uint32_t m, std::uint32_t n, std::uint32_t k, std::uint32_t row_start,
    float* output, void* workspace, std::uint64_t workspace_bytes, Stream const& stream) {
  if (!codes || !scales || !activation || !activation_scales || !output ||
      !workspace || !m || m > 1024 || !n || n > 512 || n % 8 || !k || k > 17408 ||
      k % 256 || row_start % 128 || stream.empty())
    return std::unexpected(make_error(ErrorCode::InvalidArgument, "nvfp4.gemm", "invalid tile"));
  auto guard = stream.activate();
  if (!guard) return std::unexpected(guard.error());
  auto shape = make_shape(int(m),int(n),int(k),1);
  auto sa = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {int(m),int(k),1});
  auto sb = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {int(n),int(k),1});
  auto sd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {int(m),int(n),1});
  auto* weight_codes = static_cast<std::uint8_t const*>(codes) + std::uint64_t(row_start)*k/2;
  auto* weight_scales = static_cast<std::uint8_t const*>(scales) + 256 + std::uint64_t(row_start)*k/16;
  Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, shape,
      {reinterpret_cast<cutlass::float_e2m1_t const*>(activation), sa,
       reinterpret_cast<cutlass::float_e2m1_t const*>(weight_codes), sb,
       reinterpret_cast<cutlass::float_ue4m3_t const*>(activation_scales), Sf::tile_atom_to_shape_SFA(shape),
       reinterpret_cast<cutlass::float_ue4m3_t const*>(weight_scales), Sf::tile_atom_to_shape_SFB(shape)},
      {{1.f,0.f}, nullptr,sd,output,sd}};
  args.epilogue.thread.alpha_ptr = static_cast<float const*>(scales);
  Gemm gemm;
  if (auto st = status(gemm.can_implement(args)); !st) return st;
  if (Gemm::get_workspace_size(args) > workspace_bytes)
    return std::unexpected(make_error(ErrorCode::InvalidArgument,"nvfp4.workspace","workspace too small"));
  if (auto st = status(gemm.initialize(args,workspace,stream.native())); !st) return st;
  return status(gemm.run(stream.native()));
}
}  // namespace qw38::cuda
