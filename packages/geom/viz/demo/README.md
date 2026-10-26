# geom-viz demos

Run from `geom/`:

```sh
dune exec viz/demo/static.exe
dune exec viz/demo/movie.exe
dune exec viz/demo/zoo.exe
dune exec viz/demo/arm.exe
dune exec viz/demo/conformal.exe
```

Outputs:

- `out/static.png`, `out/static.svg` — a plane, sphere, circle, points and a
  motor frame, rendered through `hugin.vg.raster` / `hugin.vg.svg`;
- `out/motor.mp4` — a 2 s, 24 fps animation of a sphere and a motor frame
  moving around a circle, rendered frame by frame and encoded with the system
  `ffmpeg`;
- `out/zoo.png`, `out/zoo.svg`, `out/zoo.mp4` — a CGA object zoo built with
  `Cga` (points, sphere, plane, line, circle), with a motor-transformed copy of
  each object animated over the ghost of the original. This exercises the
  `Cga` line/circle extractors and the `Primitive.of_cga_*` adapters;
- `out/arm.png`, `out/arm.svg`, `out/arm.mp4` — a 6-DOF kinematic chain built
  by composing `Motor`s, with links as thick segments, joint spheres and an
  end-effector frame;
- `out/conformal-dilation.*` and `out/conformal-transversion.*` — the same CGA
  objects under the two non-rigid conformal generators. These use dense CGA
  versors (`Cga.dilator`, `Cga.transversion`) and re-extract the transformed
  CGA objects, unlike the rigid motor demos.

`geom-viz` has complete `.mli` files and is still an experimental surface; the
main `geom` library is the stable part to build geometry on.

The demos use the `geom` library directly; `geom-viz` is a separate library in
`viz/lib/`.
