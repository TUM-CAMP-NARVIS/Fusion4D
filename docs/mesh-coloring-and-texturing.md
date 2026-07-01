# Mesh Coloring & Texturing

**Status:** documented behaviour + enhancement sketch.

**Summary:** Fusion4D does **not** produce a texture map for the reconstructed
mesh. There is no UV parameterization, no texture atlas, no texture image, and no
`.mtl`/material output. The mesh is colored with **per-vertex vertex colors**
only. Those vertex colors come from a color field baked into the TSDF volume
(one color per voxel), interpolated onto vertices during marching cubes. This
means the effective color detail is capped by the **voxel grid**, independent of
image resolution — see also
[`color-depth-resolution-coupling.md`](color-depth-resolution-coupling.md) §5.

---

## 1. Verification — there is no UV texture map

A repo-wide search for `texcoord` / `uv` / `atlas` / `.mtl` / `usemtl` /
`mtllib` finds no texture-mapping code in the reconstruction or export paths
(only unrelated matches: Ceres `parameter` blocks, CUDA `tex2D*` sampling,
OpenGL `textureMat` view matrices).

- **PLY export** writes only per-vertex `red/green/blue` (`uchar`) — no
  `property float s/t`, no external texture reference:
  `CSurface/CSurface.cpp:133-138` (header declaration) and `:153-154` (values,
  written as `color * 255`).
- **OBJ export** calls `glmWriteOBJ(model, file, GLM_SMOOTH)` — **not**
  `GLM_TEXTURE` or `GLM_MATERIAL` — so no `vt` texcoord lines and no `.mtl` are
  emitted: `CSurface/CSurface.cpp:628-633`. `convertToObj()` sets
  `mtllibname = NULL` (`:640`). The bundled GLM library *supports* texcoords and
  materials (`CSurface/glm.h:40, 42, 117-118`), but the exporter never uses them.
- **Core-library readback** (`MeshSnapshot`) exposes `positions`, `normals`,
  `colors`, `indices` — and has **no** UV/texcoord field:
  `include/fusion4d_core/fusion4d_core.h:75-80`.

Conclusion: consumers receive a **vertex-colored triangle mesh**. Any "texture"
appearance is Gouraud-style interpolation of vertex colors by the renderer, not
image-based texturing.

---

## 2. How vertex colors are actually produced (two stages)

### Stage 1 — color baked into the TSDF volume, per voxel

During fusion, each voxel is projected into the (depth) camera and the color
image is sampled at that pixel; the sample is accumulated as a weighted running
average into a per-voxel color field.

- GPU: `FastNonrigidMatching/volumetric_fusion_impl_v1.cu:821-825` (no-deform
  kernel) and `:2052` (deform kernel); per-voxel color stored at `:845-849` /
  `:2078-2100`.
- CPU: `Basics/TSDF.cpp:405-437` (accumulate into `clr_field`).

Because this is one color per voxel, color detail is quantized to the voxel size
(`fusion4d_vxl_res` / `f2f_motion_vxl_res`, e.g. `0.3` in
`ConfigFileExamples/OfflineFolderFusion_upperbody.cfg:13-14`).

### Stage 2 — voxel colors interpolated onto mesh vertices in marching cubes

When a vertex is created on a voxel edge at the iso-crossing factor `mu`, its
color is the linear interpolation of the two endpoint voxels' colors, using the
**same** `mu` that positions the vertex, normalized to `[0,1]`:

- `FastNonrigidMatching/volumetric_fusion_marchingcubes.cu:251-253` (`mu`),
  `:267-276` (X-edge), `:301-310` (Y-edge), `:335-344` (Z-edge).
- vMesh variant: `volumetric_fusion_marchingcubes_vMesh.cu:320-329, 364-373, 408-417`.
- CPU path instead trilinearly samples the color field per vertex:
  `Basics/TSDF.cpp:876-896` (`TSDF::texture_surface`).

Colors are **not** re-sampled from the source image at mesh time — they are read
from the already-colored volume.

---

## 3. Why the result looks low-detail

Two independent resolution bottlenecks stack:

1. **Source cap** — color is force-resized to the depth resolution before it ever
   reaches fusion (see `color-depth-resolution-coupling.md` §2.3).
2. **Volume cap** — even at full source resolution, color is quantized to one
   sample per voxel, so vertex colors cannot carry sub-voxel detail regardless of
   image resolution.

Decoupling color resolution (the other doc) removes bottleneck #1 but **not** #2.
Removing #2 requires image-based texturing (below).

---

## 4. Enhancement sketch — real image-based texture mapping

A true texture map would project the original (full-resolution) color image(s)
directly onto the mesh triangles, bypassing the voxel-grid color quantization
entirely. This is a **separate, larger** effort from resolution-decoupling, and
they compose well (decoupling gives you the high-res source that texturing then
maps). Rough shape:

1. **Generate a UV parameterization / atlas** for the reconstructed mesh
   (e.g. per-triangle chart packing, or a library such as xatlas). The mesh
   currently has no UVs, so this is new.
2. **Assign source pixels to texels.** For each texel, find the mesh surface
   point, choose the best observing camera (visibility + view-angle + resolution),
   and sample that camera's full-resolution color image using its own
   projection — reusing the per-camera projection already available on the CPU
   photometric path (`DeformGraphOptDensePtsAndClr.h`) and the decoupled color
   projection from `DDF.cpp:142-146`.
3. **Handle multi-view blending / seams** across camera boundaries (weight by
   view angle and depth confidence; seam-leveling to avoid visible edges).
4. **Handle occlusion** with a per-camera depth test so hidden triangles are not
   textured from a camera that cannot see them (same concern flagged for
   decoupled per-voxel color).
5. **Export** UVs + a texture image + `.mtl`. The OBJ exporter can already carry
   texcoords/materials via GLM (`GLM_TEXTURE` / `GLM_MATERIAL`,
   `glm.h:40, 42, 117-118`), and PLY would need `s/t` properties added
   (`CSurface.cpp:133-138`); `MeshSnapshot` would need a UV array + texture
   handle (`fusion4d_core.h:75-80`).

Note the non-rigid subtlety: geometry deforms per frame, so a texture atlas is
typically built against a canonical/reference frame (the key-volume reference in
`Fusion4DGPUKeyFrames.cpp`), then either kept stable or re-sampled per frame
depending on the target use case.

---

## 5. Quick reference — file:line list

- PLY per-vertex color, no UV: `CSurface/CSurface.cpp:133-138, 153-154`
- OBJ export, `GLM_SMOOTH` only (no texture/material): `CSurface/CSurface.cpp:628-633, 640`
- GLM texcoord/material support (unused): `CSurface/glm.h:40, 42, 117-118`
- Core readback, no UV field: `include/fusion4d_core/fusion4d_core.h:75-80`
- Per-voxel color bake: `volumetric_fusion_impl_v1.cu:821-825, 2052`; `TSDF.cpp:405-437`
- Voxel→vertex color interpolation (marching cubes): `volumetric_fusion_marchingcubes.cu:251-253, 267-276, 301-310, 335-344`
- CPU per-vertex texturing: `TSDF.cpp:876-896`
- Voxel resolution config: `ConfigFileExamples/OfflineFolderFusion_upperbody.cfg:13-14`
