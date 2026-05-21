#ifndef __CUDA_TEXTURE_HANDLES_H__
#define __CUDA_TEXTURE_HANDLES_H__

#include <cuda_runtime.h>

// Host-side texture/surface handles. Defined in CudaTextureHandles.cu.
extern cudaTextureObject_t tex_depthImgs;
extern cudaTextureObject_t tex_normalMaps;
extern cudaSurfaceObject_t surf_visHull;
extern cudaTextureObject_t tex_visHull;

// Device-side symbols for texture/surface handles. Defined in one .cu
// (DeformGraphCudaImpl.cu).
extern __device__ cudaSurfaceObject_t surf_ndIds_dev;
extern __device__ cudaTextureObject_t tex_ndIds_dev;
extern __device__ cudaTextureObject_t tex_ndIds_old_dev;

// device-side symbols for commonly used texture/surface objects
extern __device__ cudaSurfaceObject_t surf_visHull_dev;
extern __device__ cudaTextureObject_t tex_visHull_dev;
extern __device__ cudaTextureObject_t tex_depthImgs_dev;
extern __device__ cudaTextureObject_t tex_normalMaps_dev;
extern __device__ int dev_use_depth_top_bit_as_seg;

#if defined(__CUDA_ARCH__)
#define tex_ndIds ::tex_ndIds_dev
#define tex_ndIds_old ::tex_ndIds_old_dev
#define surf_ndIds ::surf_ndIds_dev
#define surf_visHull ::surf_visHull_dev
#define tex_visHull ::tex_visHull_dev
#define tex_depthImgs ::tex_depthImgs_dev
#define tex_normalMaps ::tex_normalMaps_dev
#endif

#endif  // __CUDA_TEXTURE_HANDLES_H__
