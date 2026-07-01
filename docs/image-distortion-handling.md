# Image Distortion Handling

**Status:** documented behaviour; a known input requirement, not a bug.

**Summary:** Fusion4D expects **already-undistorted (rectified)** depth and color
images. Lens distortion (radial `k1,k2,k3` / tangential `p1,p2`) is **not**
modelled anywhere in the reconstruction. Both the forward projection
(voxel → pixel) and the back-projection (pixel → 3D) are pure pinhole using only
`fx, fy, cx, cy`. Distortion coefficients can be *loaded* from a 3DTM
calibration file, but they are never applied to the images fed into fusion and
are never uploaded to the GPU. The caller is responsible for rectifying input.

---

## 1. Reconstruction is pure pinhole (no distortion terms)

- GPU fusion kernel forward projection — only `fx,fy,cx,cy`:
  `FastNonrigidMatching/volumetric_fusion_impl_v1.cu:792-793`
  ```cpp
  int u = ROUND(fx * X[0] / X[2] + cx);
  int v = ROUND(fy * X[1] / X[2] + cy);
  ```
  (deform kernel identical at `:1965-1968`).
- GPU back-projection is the exact inverse pinhole (no distortion undo):
  `volumetric_fusion_impl_v1.cu:812-813`
  ```cpp
  p[0] = (u - cx) * p[2] / fx;
  p[1] = (v - cy) * p[2] / fy;
  ```
- CPU paths are the same pinhole model:
  `Basics/TSDF.cpp:374-376`, `Basics/DDF.cpp:65-66`.

The design contract "inputs carry no lens distortion" is stated explicitly in
many places:
- `Basics/basic_structure.h:86` — `// assume depthMats have not lens distortion`
- `Basics/surface_misc.h:172` — `//no lens distortion` (and `:50`, `:91`)
- `NonRigidMatch/LMICP.cpp:7-8` — parameters `//without len distortion`
- `NonRigidMatch/DeformGraphOptDensePtsAndClr.h:23, 38, 51, 69, 100` —
  `//full image without distortion` / `//without lens distortion`

---

## 2. Distortion coefficients are loaded but never applied to fusion input

This is the part most likely to mislead: the machinery *exists*, so it looks
like distortion might be handled. It is not, on the fusion input path.

- The 3DTM JSON loader **reads** `k1,k2,p1,p2,k3` into `GCameraView::Radial[]`:
  `CameraView/CameraView.cpp:363-367`
  ```cpp
  const char* k_name[] = { "k1", "k2", "p1", "p2", "k3" };
  for (int i = 0; i < 5; i++)
      Radial[i] = camera["intrinsics"][k_name[i]].asDouble();
  ```
- It also **builds** an OpenCV undistort map:
  `CameraView.cpp:382` (`initUndistortMap()`), implemented at `:1174`
  (`cv::initUndistortRectifyMap(... mapX, mapY)`).
- **But the map is never applied to the depth/color images fed into fusion.**
  `GCameraView::UndistortImage()` / `UndistortDepthMap()` are declared
  (`CameraView.h:111-116`) and defined, yet never invoked in the frame-feeding
  path. The only live caller of the point-undistort helper is the *disabled*
  VXL `visual_feature_matching` module
  (`Basics/visual_feature_matching.cpp:130`).
- Distortion is **not uploaded to the GPU** either — `setup_cameras` copies only
  `K`, `R`, `T` into `CameraViewCuda`
  (`FastNonrigidMatching/volumetric_fusion.cpp:113-125`), and `CameraViewCuda`
  has no distortion field (`geometry_types_cuda.h:377-385`).

Net effect: even when a calibration file specifies distortion coefficients, they
have **no effect** on GPU reconstruction.

---

## 3. What this means for the two entry points

- **Demo** (`DeformableFusionDemo`) and **core library** (`fusion4d_core`)
  contain *zero* distortion handling:
  - No undistort call anywhere in their code.
  - `CameraCalibration` has no distortion field at all — only intrinsics /
    rotation / translation (`include/fusion4d_core/fusion4d_core.h:30-42`).
  - The upperbody demo uses the `--fx/--fy/--cx/--cy` fallback, which implies
    zero distortion; the VolumeDeform datasets ship pre-rectified depth, which is
    why the demo looks correct.

**Practical requirement:** rectify (undistort) both depth and color yourself
before calling `submit_frame` / feeding the demo. Nothing downstream compensates.

---

## 4. If in-pipeline distortion is ever wanted

Two options, in rough order of effort:

1. **Undistort on ingest (simplest, CPU).** Populate distortion coefficients on
   `GCameraView` and call `UndistortImage` / `UndistortDepthMap` (which already
   exist and already build `mapX/mapY`) inside the frame-feeding path before the
   images reach `feed_*_textures`. Add a distortion field to `CameraCalibration`
   and wire it through `fusion4d_core`. Caveat: undistorting a *depth* map needs
   care (nearest-neighbour / no interpolation across depth discontinuities).

2. **Distortion in the projection (most accurate, GPU).** Add distortion
   coefficients to `CameraViewCuda` and apply the standard Brown–Conrady model in
   the kernel's forward projection (`volumetric_fusion_impl_v1.cu:792-793` and
   `:1965-1968`). This keeps images untouched but adds per-voxel cost. Most
   relevant for a **native high-resolution color** stream (see
   [`color-depth-resolution-coupling.md`](color-depth-resolution-coupling.md)),
   where a real, unrectified color frame is far more likely to carry meaningful
   distortion than the already-rectified depth. If the decoupled-color work
   happens first, adding distortion to the *color* projection is the natural
   place to do this.
