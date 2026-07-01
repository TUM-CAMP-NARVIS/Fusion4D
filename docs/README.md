# Fusion4D — Design Notes & Investigation Docs

This directory collects investigation notes and design background intended to
give future enhancement work a solid starting point. Each document records
**what the code does today**, **why** it is that way, and **what would have to
change** to lift a given limitation, with concrete `file:line` references.

These are living documents. When you change the behaviour described here, update
the corresponding note (or mark the relevant section as resolved).

## Index

- [`color-depth-resolution-coupling.md`](color-depth-resolution-coupling.md) —
  Why color (RGB) images must currently match the depth resolution, how color is
  consumed in reconstruction, and a concrete plan for decoupling color from depth
  (separate intrinsics + depth→color extrinsic → full-resolution texture).
- [`mesh-coloring-and-texturing.md`](mesh-coloring-and-texturing.md) —
  How the reconstructed mesh is colored. Short answer: **per-vertex vertex
  colors**, not a UV texture map — and color is quantized to the voxel grid.
  Includes a sketch of what real image-based texture mapping would take.
- [`image-distortion-handling.md`](image-distortion-handling.md) —
  Whether the pipeline expects undistorted images or compensates for lens
  distortion. Short answer: it expects **already-undistorted** input; distortion
  is not modelled in the reconstruction.

## Scope / caveats

- Line numbers refer to the tree state at the time of writing (branch `main`,
  around commit `102ed4f`). If code has since shifted, search for the quoted
  snippet rather than trusting the exact line.
- There are two reconstruction backends in the tree: a **GPU** path
  (`FastNonrigidMatching/volumetric_fusion_impl_v1.cu`) and a **CPU** path
  (`Basics/TSDF.cpp`, `Basics/DDF.cpp`). The demo and the `fusion4d_core`
  library exercise the **GPU** path. Where behaviour differs between the two,
  the docs call it out explicitly.
