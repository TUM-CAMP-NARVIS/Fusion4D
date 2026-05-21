#include "PCGCuda_GPU.cuh"
#include <cuda_runtime.h>
#include "fast_pcg.cuh"
#include "../Common/cuda/PinnedMemory.h"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "PCGCuda.cu"
}