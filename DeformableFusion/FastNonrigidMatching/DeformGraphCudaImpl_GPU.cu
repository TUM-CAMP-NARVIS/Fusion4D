// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "DeformGraphCudaImpl_GPU.cuh"
#include "cuda_math_common.cuh"
#include <cuda_runtime.h>

#include "../Common/cuda/PinnedMemory.h"
#include "../Common/cuda/CudaHelpers.h"

// Include implementation at global scope so method definitions match the
// class declared in DeformGraphCudaImpl.cuh. The GPU header exposes a
// `using ::DeformGraphCudaImpl;` into `Fusion4D_GPU` for namespaced access.
#include "DeformGraphCudaImpl.cu"