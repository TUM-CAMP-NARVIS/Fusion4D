#include "volumetric_fusion_GPU.h"

#include "stdafx.h"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "volumetric_fusion.cpp"
}  // namespace Fusion4D_GPU