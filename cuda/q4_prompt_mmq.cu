#include "quant_mmv.h"
#include "quant_mmq_mma.cuh"
#include "pdl_launch.cuh"

// OPT-068: focused TU for production Q4 prompt MMQ instantiations and
// launch wrappers. Selected family from OPT-061 rotating ranking:
// prompt-ffn (699.91 ms) ahead of decode-ffn / decode-mixer. Do not
// include this header from full_scheduler.cu.

QW38_PDL_REGISTER_DEVICE_OPS()
