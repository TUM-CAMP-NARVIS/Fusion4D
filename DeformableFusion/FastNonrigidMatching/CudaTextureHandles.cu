#include "CudaTextureHandles.h"

#ifndef __CUDA_ARCH__
cudaTextureObject_t tex_depthImgs = 0;
cudaTextureObject_t tex_normalMaps = 0;
cudaSurfaceObject_t surf_visHull = 0;
cudaTextureObject_t tex_visHull = 0;
#endif

// Define device-side texture/surface symbols exactly once
__device__ cudaSurfaceObject_t surf_visHull_dev = 0;
__device__ cudaTextureObject_t tex_visHull_dev = 0;
__device__ cudaTextureObject_t tex_depthImgs_dev = 0;
__device__ cudaTextureObject_t tex_normalMaps_dev = 0;
__device__ int dev_use_depth_top_bit_as_seg = 1;
