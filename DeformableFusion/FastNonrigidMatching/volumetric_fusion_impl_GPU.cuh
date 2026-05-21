#ifndef __VOLUMETRIC_FUSION_IMPL_GPU_CUH__
#define __VOLUMETRIC_FUSION_IMPL_GPU_CUH__
#include <stdint.h>

#include "../Common/cuda/CudaHelpers.h"
#include "../Common/cuda/PinnedMemory.h"
#include "CudaGlobalMemory.h"
#include "DeformGraphCudaImpl_GPU.cuh"
#include "cuda_runtime.h"
#include "device_atomic_functions.h"
#include "device_launch_parameters.h"
#include "geometry_types_cuda.h"
#include "utility.h"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "mc_mesh_processing.cuh"
#include "volumetric_fusion_impl.cuh"
}  // namespace Fusion4D_GPU

#endif