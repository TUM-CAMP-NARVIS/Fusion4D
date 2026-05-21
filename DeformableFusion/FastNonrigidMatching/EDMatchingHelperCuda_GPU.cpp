// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "EDMatchingHelperCuda_GPU.h"

#include "color_map.h"
#include "stdafx.h"

namespace Fusion4D_GPU {
#if !defined(FUSION4D_GPU_HAS_CUDA_ALIAS)
namespace cuda = ::cuda;
#define FUSION4D_GPU_HAS_CUDA_ALIAS
#endif
#include "EDMatchingHelperCuda.cpp"
}  // namespace Fusion4D_GPU