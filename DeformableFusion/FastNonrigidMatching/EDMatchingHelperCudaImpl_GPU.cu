#include "EDMatchingHelperCudaImpl_GPU.cuh"
#include "../Common/cuda/PinnedMemory.h"
#include "../Common/cuda/CudaHelpers.h"

namespace Fusion4D_GPU
{
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
    namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "EDMatchingHelperCudaImpl.cuh"
}
