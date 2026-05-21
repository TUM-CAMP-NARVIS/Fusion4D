// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "DeformGraphCuda_GPU.h"

#include "../Common/cuda/CudaHelpers.h"
#include "../Common/cuda/PinnedMemory.h"
#include "DeformGraph.h"
#include "stdafx.h"

using namespace NonrigidMatching;

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "DeformGraphCuda.cpp"
}  // namespace Fusion4D_GPU