# Color/Depth Resolution Coupling

**Status:** limitation documented, not yet addressed.

**Summary:** Fusion4D currently requires the color (RGB) image to have the *same
resolution* as the depth image and to be *pixel-aligned* with it. This is not a
requirement of the reconstruction algorithm — it is a plumbing shortcut. The
color image is treated as if it were captured by the depth camera itself: every
place that needs a color value projects a 3D point with the **depth** camera's
intrinsics/extrinsics and samples the color image at the resulting depth pixel
`(u, v)`. As a consequence, the final texture resolution is capped at the depth
resolution (e.g. 640×480 for the VolumeDeform datasets), even if a much
higher-resolution RGB sensor is available.

A fully decoupled scheme (separate color intrinsics + a depth→color extrinsic,
sampling the color image with its own projection) is feasible and is in fact
**already implemented on the CPU path** in `DDF.cpp` — just not wired into the
GPU pipeline that the demo and `fusion4d_core` actually run.

**Important caveat:** decoupling color resolution is *necessary but not
sufficient* for a high-detail result. Color is also quantized to the **voxel
grid** (one color per voxel, then interpolated to vertices), so there is a
second, independent resolution bottleneck that decoupling does **not** remove.
See §4 below and [`mesh-coloring-and-texturing.md`](mesh-coloring-and-texturing.md).

---

## 1. How color is consumed in reconstruction

There are three distinct consumers of color. Understanding all three matters for
scoping a change, because only the first two are on the live GPU path.

### 1.1 Integration into the TSDF volume (per-voxel color) — GPU, live path

In the fusion kernel, each voxel center `V` is transformed into the depth
camera's space, projected with the depth intrinsics `K` to `(u, v)`, and — if a
valid depth sample exists there — the color image is sampled at the **same**
`(u, v)` and accumulated as a weighted average into the voxel's color.

- Projection with depth `K`, `R|T`:
  `DeformableFusion/FastNonrigidMatching/volumetric_fusion_impl_v1.cu:782-793`
  ```cpp
  cuda_vector_fixed<float,3> X = dev_cam_views[i].cam_pose.R * V + dev_cam_views[i].cam_pose.T;
  cuda_matrix_fixed<float,3,3> const &K = dev_cam_views[i].K;
  int u = ROUND(fx * X[0] / X[2] + cx);
  int v = ROUND(fy * X[1] / X[2] + cy);
  if (u >= 0 && u < depth_width && v >= 0 && v < depth_height) { ... }
  ```
- Color sampled at that **same** `(u, v)` and accumulated (note the BGR→RGB
  channel swap `.x += clr.z … .z += clr.x`, weighted by the depth ray/normal
  weight `weight_cur`):
  `volumetric_fusion_impl_v1.cu:821-825`
  ```cpp
  uchar4 clr = tex2DLayered<uchar4>(tex_colorImgs, (float)u, (float)v, i);
  color_new.x += clr.z * weight_cur;
  color_new.y += clr.y * weight_cur;
  color_new.z += clr.x * weight_cur;
  ```
- Written to the per-voxel color buffer: `volumetric_fusion_impl_v1.cu:845-849`.
- The **deform-enabled** kernel repeats the identical pattern on the deformed
  voxel `V_t`: projection at `:1955-1968`, color fetch at `:2052`, write-back at
  `:2078-2100`.

### 1.2 Read-back onto the mesh in marching cubes — GPU, live path

Vertex colors are **not** re-sampled from the image at mesh time. Marching cubes
loads the stored per-voxel colors and linearly interpolates the two
edge-endpoint voxel colors using the *same* zero-crossing factor `mu` that
positions the vertex:

- `volumetric_fusion_marchingcubes.cu:88, 133` — `sh_color[idx] = volume.colors[pos];`
- `volumetric_fusion_marchingcubes.cu:251-253` — `mu = val_cur / (val_cur - vals[0]);`
- Per-edge color lerp + `/255`:
  `volumetric_fusion_marchingcubes.cu:267-276` (X), `:301-310` (Y), `:335-344` (Z)
- vMesh variant: `volumetric_fusion_marchingcubes_vMesh.cu:320-329, 364-373, 408-417`

Implication: this stage needs **no change** for decoupling — it operates purely
on voxel colors that were already fixed during integration.

### 1.3 Optional photometric term in tracking — CPU/Ceres only

Color can contribute a photometric residual to the non-rigid / rigid alignment,
but only on the CPU path. The GPU fast-matching solver is purely geometric (no
color terms in `EDMatchingHelperCudaImpl*.cu` / `DeformGraphCudaImpl*.cu`).

- Term: `NonRigidMatch/DeformGraphOptDensePtsAndClr.h` — `namespace DenseColorTerm`
  (`:401`), `namespace RigidDenseColorTerm` (`:209`). Projection of the deformed
  vertex uses a precomputed 3×4 projection matrix (K·[R|T]):
  `:471-474`; visibility test against the projected depth map at the same `(u,v)`
  `:506`; color residual `image_color - stored_vertex_color` at `:510-515`.
- Wiring / weights: `NonRigidMatch/LMICP.cpp:227` (build projection matrices),
  `:258-281` (add residual block, gated on `w_color`), with `w_color = 0.0` at
  `:106` and `w_color = 1.0` at `:181`.

Implication: for the GPU demo, **color is texture-only** — it does not influence
geometry or tracking at all. This is important: decoupling color resolution
cannot degrade tracking on the GPU path, because tracking never looks at color
there.

---

## 2. Where the "same resolution" requirement is enforced

The coupling is hard-coded at every layer, and is explicitly flagged as a
shortcut in the source.

### 2.1 Single camera model — one intrinsic, one extrinsic, "depth" only

- GPU per-camera struct carries a single `K` + pose, no color variant:
  `FastNonrigidMatching/geometry_types_cuda.h:377-385`
  ```cpp
  struct CameraViewCuda {
      cuda_matrix_fixed<float,3,3> K;
      RigidTransformCuda cam_pose;
  };
  ```
- Host `GCameraView` also has a single image size + single intrinsics
  (`CameraView/CameraView.h:68-80`).
- Calibration is *always* loaded with `camera_type = "depth"`, even though the
  3DTM JSON schema supports a per-type sub-key (a `"color"` block could exist):
  - Demo: `DeformableFusionDemo/DeformableFusionDemo.cpp:383-384`
  - Core lib: `src/fusion4d_core.cpp:212-213`
  - Loader reads `root["inputCameras"][idx][camera_type]`:
    `CameraView/CameraView.cpp:285`, intrinsics `:333-336`, extrinsics `:344-361`.
- Public API has **no** color intrinsics and **no** color resolution:
  - `include/fusion4d_core/fusion4d_core.h:30-42` — `CameraCalibration` has one
    `intrinsics` / `rotation` / `translation`.
  - `include/fusion4d_core/fusion4d_core.h:53-63` — `SessionConfig` has only
    `depth_width` / `depth_height`; there is no `color_width` / `color_height`.

### 2.2 Color GPU texture allocated at depth dimensions

- `FastNonrigidMatching/volumetric_fusion.h:62-63` — the assumption is written
  down in a `TODO`:
  ```cpp
  //TODO: assume depth and color is aligned and have the same resolution
  this->allocate_cuda_arrays_colors(depth_num, depth_width, depth_height);
  ```
- Allocation uses those dims verbatim (RGBA8 layered array):
  `FastNonrigidMatching/volumetric_fusion_impl.cu:191-194`
  (`cudaMalloc3DArray(..., make_cudaExtent(width, height, tex_num), cudaArrayLayered)`).

### 2.3 Runtime enforcement / silent coercion

- GPU upload asserts color dims == depth dims and copies a depth-sized extent:
  `FastNonrigidMatching/volumetric_fusion.cpp:79-91`
  ```cpp
  cv::Mat img_with_alpha = pad_alpha_for_color_CvMat(colorImgs[i]);
  assert(depth_height_ == img_with_alpha.rows &&
         depth_width_  == img_with_alpha.cols);
  ```
- Core library *throws* on mismatch — color is validated against the **depth**
  size (`config.depth_width/height`):
  `src/fusion4d_core.cpp:26-41` (`validate_image` → `"... dimensions mismatch"`),
  `:49-56` (`wrap_color_image`), `:259-276` (`feed_optional_color` passes depth
  dims as expected color dims).
- Demo silently downsamples color to the depth size at load time — **this is the
  direct cause of the texture-resolution cap**:
  `DeformableFusionDemo/DeformableFusionDemo.cpp:194-203`
  ```cpp
  cv::Mat load_color(const fs::path& path, int width, int height) {
    cv::Mat color = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (color.cols != width || color.rows != height)
      cv::resize(color, color, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    return color;
  }
  ```
  where `width/height` are `depthWidth/depthHeight` from the first depth frame
  (`:501-502`, passed at `:552-553`).

### 2.4 Every resolution-coupling site (color pixel index derived from depth projection)

| # | Projection (depth K,R,T → u,v) | Color fetch at same (u,v) | Path |
|---|---|---|---|
| 1 | `volumetric_fusion_impl_v1.cu:792-793` | `:821` | GPU, no-deform |
| 2 | `volumetric_fusion_impl_v1.cu:1965-1968` | `:2052` | GPU, deform |
| 3 | `TSDF.cpp:374-376` | `:405-407` | CPU TSDF |
| 4 | `DDF.cpp:65-66` | `:136-138` (aligned branch only) | CPU DDF |
| 5 | `DeformGraphOptDensePtsAndClr.h:471-474` | `:510-515` | CPU photometric (non-rigid) |
| 6 | `DeformGraphOptDensePtsAndClr.h:264 / :304` | (same) | CPU photometric (rigid) |
| 7 | `affines_nonrigid_pw_opt.h:578` | `:619` | CPU affine term |

Plus allocation-level coupling: `volumetric_fusion.h:62-63`,
`volumetric_fusion_impl.cu:194`, assert `volumetric_fusion.cpp:86-87`.

---

## 3. The way out — decoupled color camera (already prototyped on CPU)

The reconstruction math imposes **no** requirement that color and depth match.
`DDF.cpp` already contains a working decoupled branch that does exactly what a
proper RGB-D setup needs: it projects each voxel into the **color** camera using
*separate* color intrinsics + extrinsics and bilinearly samples the color image.

`Basics/DDF.cpp:47-49, 142-146`:
```cpp
bool bColorDepthAligned = (cam == cam_clr);        // :47-49
...
if (bColorDepthAligned) {
    // sample color at the depth pixel (u,v)          :134-139
} else {
    vnl_vector_fixed<double,3> X_clr = Rc*X_wld + Tc; // separate color extrinsic
    double u_c = fx_c*X_clr[0]/X_clr[2] + cx_c;       // separate color intrinsic
    double v_c = fy_c*X_clr[1]/X_clr[2] + cy_c;
    pickAColorPixel(img, u_c, v_c, clr_tmp);          // bilinear sample :146
}
```
with `Rc, Tc, Kc` taken from a distinct `cam_clr` (`DDF.cpp:23-31`).

Note: `X_clr = Rc·X_wld + Tc` (a full world→color extrinsic) is mathematically
identical to the "depth extrinsic + depth→color transform" formulation, since
`world→color = (depth→color) ∘ (world→depth)`. Either parameterization works;
pick whichever matches how the calibration is delivered.

### 3.1 Concrete change list to bring this to the GPU path

Scoped, localized, and low-risk (geometry is untouched — only the color lookup
gains its own projection):

1. **Calibration / data model**
   - Add color intrinsics + a color extrinsic (or a depth→color rigid transform)
     per camera to `CameraCalibration` (`include/fusion4d_core/fusion4d_core.h:30-42`)
     and `SessionConfig` (`:53-63`); add `color_width` / `color_height`.
   - Read the `"color"` block from the 3DTM JSON (loader already supports a
     `camera_type` arg — `CameraView.cpp:285`); call it with `"color"` where
     today both call sites pass `"depth"`
     (`DeformableFusionDemo.cpp:383-384`, `fusion4d_core.cpp:212-213`).
   - Extend `CameraViewCuda` (`geometry_types_cuda.h:377-385`) with `Kc` + a
     color pose; upload them in `setup_cameras` (`volumetric_fusion.cpp:108-137`,
     which today copies only `K`, `R`, `T`).

2. **Upload path**
   - Allocate `cu_3dArr_color_` at **color** resolution
     (`volumetric_fusion.h:63`, `volumetric_fusion_impl.cu:191-194`).
   - Drop the same-size assert / depth-sized extent and use color dims
     (`volumetric_fusion.cpp:79-91`).
   - Remove the throw-on-mismatch in the core lib and the `cv::resize` in the
     demo; validate color against `color_width/height` instead
     (`fusion4d_core.cpp:49-56, 259-276`; `DeformableFusionDemo.cpp:194-203`).

3. **Kernel (the actual crux — a few lines in two kernels)**
   - After the existing depth projection yields `(u,v)` for the SDF/depth/normal
     lookups, add a **second** projection into color space:
     `X_c = Rc·V + Tc; u_c = fx_c·X_c.x/X_c.z + cx_c; v_c = …`, bounds-checked
     against `color_width/height`, and sample the color texture there with the
     texture unit's built-in linear filtering (fractional coords are fine).
   - Sites: `volumetric_fusion_impl_v1.cu:821` (no-deform) and `:2052` (deform).
   - For the deform kernel, use the deformed voxel `V_t` for the color projection
     too (consistency with the depth projection at `:1955`).

4. **No change needed:** marching cubes (§1.2) and — on the GPU path — tracking
   (§1.3, color unused there).

### 3.2 Things to watch

- **Occlusion / visibility mismatch.** With decoupled cameras, a voxel visible
  to the depth camera may be occluded in the color view (or vice versa). The
  current code implicitly assumes shared visibility. Consider a color-space
  z-test (e.g. against a color-view depth buffer) or accept minor color bleeding
  at occlusion boundaries as a first cut.
- **Distortion.** A native high-res color frame is more likely to carry
  meaningful lens distortion than the (already-rectified) depth. See
  [`image-distortion-handling.md`](image-distortion-handling.md) — the pipeline
  does **not** undistort, so either feed a rectified color image or add
  distortion to the new color projection.
- **Weighting.** Color is currently averaged with the depth-derived
  `weight_cur`. That weight still makes sense as a confidence, but you may want a
  separate color-visibility weight once the views are decoupled.
- **Two kernels stay in sync.** The no-deform and deform kernels duplicate the
  projection logic; both must be updated identically.

---

## 4. Second resolution bottleneck: voxel-grid color quantization

Even with a full-resolution, correctly-projected color stream, the output color
detail is capped a **second** time — by the voxel grid. Color is never stored
per-pixel or per-triangle; it is stored **per voxel** and then interpolated to
mesh vertices. So the finest color variation the mesh can represent is roughly
one sample per voxel, regardless of how sharp the input image is.

- Per-voxel color is accumulated during fusion (one running-average color per
  voxel): `volumetric_fusion_impl_v1.cu:845-849` / `:2078-2100`;
  CPU `Basics/TSDF.cpp:405-437`.
- Marching cubes interpolates the two edge-endpoint voxel colors onto each vertex
  using the iso-crossing factor `mu` — it does **not** re-sample the source
  image: `volumetric_fusion_marchingcubes.cu:251-253, 267-276, 301-310, 335-344`.
- Voxel size is set by `fusion4d_vxl_res` / `f2f_motion_vxl_res`
  (e.g. `0.3` in `ConfigFileExamples/OfflineFolderFusion_upperbody.cfg:13-14`).

Consequences for planning:

- **Decoupling color resolution (§3) removes the *source* cap but not this one.**
  Feeding a 1920×1080 color image into a 0.3-unit voxel grid still yields
  voxel-grained color.
- Shrinking `*_vxl_res` increases color detail but also multiplies geometry cost
  and GPU memory (the TSDF volume grows cubically) — it is not a free lever.
- The bottleneck-free path to sharp texture is **image-based texture mapping**
  (UV atlas + sampling the original images per triangle), which bypasses the
  voxel grid entirely. That is a separate, larger enhancement — see
  [`mesh-coloring-and-texturing.md`](mesh-coloring-and-texturing.md) §4. It
  composes naturally with §3: decoupling supplies the high-res source that a
  texture-mapping stage then projects onto the mesh.

---

## 5. Quick reference — load-bearing file:line list

- Force-resize color → depth: `DeformableFusionDemo/DeformableFusionDemo.cpp:198-201`
- Positional depth/color pairing: `DeformableFusionDemo.cpp:209-211`, `:221-223`
- Core-lib color validation vs depth dims: `src/fusion4d_core.cpp:26-56, 259-276`
- GPU same-size assert: `FastNonrigidMatching/volumetric_fusion.cpp:86-87`
- Color texture alloc at depth dims: `volumetric_fusion.h:62-63`, `volumetric_fusion_impl.cu:191-194`
- GPU color sample at depth (u,v): `volumetric_fusion_impl_v1.cu:821`, `:2052`
- Single "depth" intrinsic load: `DeformableFusionDemo.cpp:383-384`, `fusion4d_core.cpp:212-213`, `CameraView.cpp:285, 333-361`
- **Decoupled reference implementation (CPU):** `Basics/DDF.cpp:47-49, 142-146`
