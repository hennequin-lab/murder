# geom-viz — rendering geometric objects and motor movies

Status: **V1–V3 partially implemented**. `viz/lib/` contains the first
`geom_viz` library (primitives, pinhole camera, painter's-algorithm drawing,
PNG/SVG/PDF frames, `ffmpeg` movie export, CGA-to-primitive adapters) and
`viz/demo/` contains a static scene, a two-second motor animation, a CGA object
zoo, a 6-DOF motor chain and dilation/transversion movies. The core library now
implements the algebra, CGA objects including line/circle extractors and the
conformal generators, motors including log/slerp/adjoint, and PGA/E3 basics;
the PGA camera meet path, near-plane clipping, labels and sphere shading
remain.

The mathematics is in [`doc/notes.typ`](doc/notes.typ); the core design is in
[`PLAN.md`](PLAN.md). This document decides the rendering stack, the scene
model, the camera/projection layer, and the movie pipeline.

## 0. Decision in one paragraph

Use **Hugin's vector-graphics layer** — the libraries `hugin.vg`,
`hugin.vg.raster`, `hugin.vg.svg`, `hugin.vg.pdf`, and `hugin.gg` — as the
renderer. It is already part of Raven, it is pure OCaml, it renders
deterministic PNG, SVG and PDF (and RGBA tensors), and it has exactly the
primitives a projected GA scene needs: paths, circles, arcs, polygons, fills,
strokes, clipping, affine transforms, opacity and repeated stamps. Use **PGA
for the camera and projective projection** (the image plane, view rays, meets
and joins), **CGA for the round scene objects** (spheres, circles, point
pairs), and **`Motor` for all motion and interpolation**. Draw with a
depth-sorted painter's algorithm. For movies, render frames as PNG (or raw
RGBA) and encode with the system `ffmpeg`; keep the encoder outside the core
library. Put the visualizer in a **separate `geom-viz` package** so that
`geom` itself stays free of graphics dependencies.

A throwaway probe in this workspace rendered a 640×360 scene with a path,
circle, grid, stroke and dot to PNG and SVG through `hugin.vg.raster` and
`hugin.vg.svg`; the dependency path works from a sibling dune project. The
probe is not committed.

## 1. Why Hugin, and what the alternatives are

I looked at the current OCaml graphics landscape before settling. The short
version: Hugin's `hugin.vg` is a Vg-class vector graphics system that is
already in the Raven tree, so it wins on both capability and integration.

| Option | What it is | Fit for this project | Verdict |
|---|---|---|---|
| **Hugin `hugin.vg`** | Declarative 2D vector pictures (`Picture`), pages (`Renderable`), renderers to RGBA/PNG, SVG, PDF; paths, circles, arcs, polygons, clipping, transforms, opacity, stamps; glyph runs | Exactly the projected-2D rendering model we need; pure OCaml; deterministic; already a Raven package; `Picture.stamp` is ideal for repeated markers | **chosen** |
| Vg | Mature declarative 2D vector graphics (Bünzli); SVG/PDF built in, PNG via Cairo | Excellent, but would add a second vector-graphics abstraction next to Hugin's; PNG needs Cairo2; not a Raven dependency | alternative, not needed |
| Cairo2 | OCaml bindings to Cairo; imperative 2D raster/vector, gradients, fast | Good for gradients and a faster raster path; C dependency; imperative API; no scene model | optional backend later |
| Tsdl / SDL2 | Thin bindings to SDL2: window, input, textures, timing | Real-time preview only; no scene renderer; needs us to draw or upload frames | optional interactive player |
| Bogue | GUI toolkit on SDL2, with animations and an SDL drawing area | Good for sliders/playback controls; heavier than needed; not a renderer | optional UI later |
| Raylib | OCaml bindings to raylib (6.0); 3D models, shapes, shaders, real-time | Great for interactive 3D, but bypasses the projection/GA layer and is not vector/deterministic; C library | optional real-time backend, not v1 |
| Tgls | Thin OpenGL 3/4 bindings | High-performance 3D; lots of plumbing (shaders, buffers, windows) | later only if needed |
| ocaml-ffmpeg | FFmpeg bindings (savonet), requires FFmpeg 7.1+ | In-process video encoding; heavy C dependency and API | not v1; CLI is simpler |
| Prismel / Rays | Metal/SDL3 creative-coding framework | Apple-Silicon/Metal only; not portable | no |

Hugin's `hugin.vg` is close in spirit to Vg but is the right choice here:

- it is part of the same ecosystem as `nx`, `rune` and the rest of Raven;
- it has no C dependency for raster output: `Hugin_vg_raster.render` returns an
  `Nx.uint8_t` and `Hugin_vg_raster.png` returns a PNG string;
- it has SVG and PDF renderers with searchable text and embedded fonts, which
  is what we want for papers and talks;
- `Picture.stamp` paints one marker at many positions efficiently, which is
  exactly what point clouds, grid ticks and trajectory samples need;
- its semantics are precise (coverage-based antialiasing, source-over
  compositing), so small golden tests are meaningful.

What Hugin does **not** give us is a 3D pipeline, shading, or a window. That is
fine: the GA layer projects 3D to 2D itself, and the movie pipeline is
offline. A real-time viewer can come later on top of SDL/raylib without
changing the scene model.

We will use `hugin.vg` directly, not the higher-level `hugin` figure API. The
figure API is for data plots (scales, axes, legends); the visualizer needs an
arbitrary canvas of projected geometry. `hugin.gg` supplies the point, path,
stroke, colour and affine types, and `hugin.font` supplies text runs.

## 2. Scope

In scope:

- a 3D scene of GA objects (points, point pairs, lines, circles, spheres,
  planes, motor frames) with per-object styles;
- a camera with perspective and orthographic modes, implemented over PGA;
- rigid motion animation: a scene is a function of time, with motors
  interpolated by ScLERP;
- frame rendering to RGBA, PNG, SVG and PDF;
- movie rendering to MP4 (and optionally GIF) through the system `ffmpeg`;
- adapters from `Cga.t`, `Pga.t` and `Motor.t` to the scene primitives;
- examples: a transformed object, a moving motor frame, a screw interpolation,
  a robot arm, a camera orbit, a conformal transformation.

Out of scope for v1:

- a real-time interactive 3D engine, shaders, shadows, PBR, mesh import;
- a GUI (sliders, timeline scrubbing); a minimal SDL preview is a later
  optional extra;
- a full 3D rasterizer or ray tracer; the painter's algorithm is the v1
  hidden-surface method;
- a text layout engine; use `hugin.font` / `hugin.text` for labels;
- video encoding inside the OCaml process; the `ffmpeg` CLI is the v1 encoder;
- collision detection, physics, or simulation.

## 3. Rendering model

### 3.1 Neutral primitives, GA adapters

The renderer does not speak CGA or PGA directly. It consumes a small neutral
3D primitive language:

```ocaml
type p3 = { x : float; y : float; z : float }

type primitive =
  | Point of p3
  | Segment of p3 * p3
  | Polyline of p3 list
  | Polygon of p3 list
  | Circle of { center : p3; normal : p3; radius : float }
  | Sphere of { center : p3; radius : float }
  | Plane of { normal : p3; offset : float; patch : p3 list }
  | Frame of { origin : p3; x : p3; y : p3; z : p3 }
```

`Cga` and `Pga` adapters produce these values. `Motor.t` transforms any of
them rigidly: `Primitive.transform : Motor.t -> t -> t`. This keeps the
renderer independent of the algebra model, makes it easy to test with hand-made
scenes, and lets the same renderer draw PGA and CGA objects.

A `Frame` is a motor's coordinate frame (origin and three axis endpoints),
drawn as three coloured arrows.

### 3.2 Camera and projection

The camera is a PGA construction:

- `eye`, `target`, `up`: Euclidean camera parameters;
- a view motor `M_view` mapping world points into camera coordinates
  (`p_cam = M_view · p`);
- camera axes: `+x` right, `+y` up, `+z` toward the viewer; the camera looks
  down `-z`; the image plane is `z = -f`, with focal length `f`;
- a perspective projection is the meet of the **line** through the eye and the
  world point with the **image plane**. In PGA this is a line/plane meet and
  gives a point on the image plane; its `x` and `y` are the projected
  coordinates, and the meet implicitly performs the perspective division;
- a world point is culled when its camera-space `z >= -near` (behind the near
  plane), and clipped for segments and polygons;
- the page is `y`-down (Hugin's convention), so the projected `y` is flipped
  when mapping to page coordinates: `screen_y = h/2 - y_img`.

The PGA route is the reference implementation. The equivalent pinhole formula

```
x_img = f * x_cam / (-z_cam)
y_img = f * y_cam / (-z_cam)
```

is the fast path; tests assert that the two agree. This is the concrete sense
in which the visualizer "uses projective GA": the camera is a PGA point, plane
and line construction, not an ad-hoc 4×4 matrix. In code (provisional names):

```ocaml
let eye = Pga.point (vec3 0. 0. 0.)              (* camera-space origin *)
let image_plane = Pga.plane ~normal:(vec3 0. 0. 1.) ~offset:(-.f)
let ray = Pga.join eye (Pga.point p_cam)         (* line through eye and point *)
let hit = Pga.meet ray image_plane                (* point trivector *)
let x_img, y_img, _ = Pga.point_coords hit
```

In PGA the meet of a line (2-blade) and a plane (1-vector) is a point
(trivector), so the projection is one meet. In CGA the same line/plane
intersection goes through the point-pair/dual construction; that is why the
camera lives on the PGA side while the round objects live on the CGA side. The
view transform itself is a `Motor`, shared with the animation layer. The exact
sign conventions will be moved into `doc/notes.typ` §7 once they are pinned by
tests; until then this section is the provisional source of truth for the
camera.

Orthographic mode is the same camera with `x_img = s * x_cam`,
`y_img = s * y_cam` and no division; it is useful for textbook figures and
exact GA diagrams.

Depth for sorting is `-z_cam` (positive in front). The camera also exposes
`ray : Camera.t -> x:float -> y:float -> Pga.t`, the world-space line through a
pixel; this is the hook for picking and, later, a ray-traced backend.

Projection of the primitive types:

| Primitive | Projection |
|---|---|
| `Point` | project the point; cull if behind near; draw a dot |
| `Segment` | transform endpoints, clip at near, project, stroke |
| `Polyline` | clip each segment, keep the connected pieces, project, stroke |
| `Polygon` | clip against the near plane (Sutherland–Hodgman), project vertices, fill (and optionally stroke) |
| `Circle` | sample the 3D circle into a closed polyline (64–128 points), clip/project, stroke; fill if requested |
| `Sphere` | compute the silhouette circle (the tangent cone from the eye), project it, fill with the sphere style; optionally add a radial-gradient image clipped to the silhouette |
| `Plane` | use the supplied patch (a large quad or a grid); clip/project/fill with low opacity; draw grid lines if requested |
| `Frame` | three segments from the origin, coloured by axis, plus a small dot at the origin |

The sphere silhouette is exact and cheap: for a sphere of centre `c` and radius
`r`, with eye `e` and `d = |c - e|`, the silhouette is the circle on the sphere
centred at `c + (r²/d²)(e - c)` (i.e. at distance `r²/d` from `c` toward the
eye), with radius `r · sqrt(1 - r²/d²)`, in the plane perpendicular to
`e - c`. Project that circle and fill the resulting polygon. This avoids
screen-space ellipse approximations and looks correct off-axis.

### 3.3 Hidden surfaces

V1 uses a **painter's algorithm** with a layer order plus a depth key:

1. background;
2. planes (translucent), then grid lines;
3. spheres, sorted by front depth (`-z_cam - r`);
4. lines, circles and point pairs, sorted by their farthest/midpoint depth;
5. points and motor frames;
6. labels and overlays.

Within a layer, draw far to near. The model is approximate for intersecting
geometry and transparent surfaces, but it is stable and sufficient for the
diagrams and movies this library targets. A z-buffer or a small GA ray tracer
is a later backend, not a v1 requirement. The renderer's `depth` field and
layer index are public so a future backend can reuse the scene.

### 3.4 Styling

Styles are a small record over Hugin's colour and stroke types:

```ocaml
type style = {
  fill : Hugin_gg.Color.t option;
  stroke : Hugin_gg.Color.t option;
  width : float;                (* page points *)
  dash : float list;
  opacity : float;              (* multiplied into fill/stroke alpha *)
}
```

Defaults: a muted colour per object type, dark thin outlines, translucent
planes, and the axis colours red/green/blue for motor frames. A `Theme.t`
holds the background, the object palette and the axis colours; it can be light
or dark. Use `Color.of_hex` for a colorblind-friendly palette (Okabe–Ito).

Sphere shading: v1 uses a flat fill with a darker rim stroke. A later upgrade
generates a small RGBA radial-gradient disk once and draws it with
`Picture.image` clipped to the projected silhouette; `Picture.stamp` can reuse
it at different positions and scales. This stays within `hugin.vg` (no
gradient primitive needed).

### 3.5 Coordinate and unit conventions

- Hugin's page is `y`-down, in points (1/72 inch).
- The renderer's camera is `y`-up; the flip happens at the page mapping.
- A frame is described by a page size in points and a `density` in pixels per
  point: a 640×360-point page at density 2 is 1280×720 pixels. This matches
  Hugin's `Renderable.v` and `Hugin_vg_raster.render ~density`.
- World units are the caller's (metres, say). The camera's focal length is in
  page points; the scene does not assume a unit scale.

## 4. Architecture

```
                    geom (Cga, Pga, Motor)
                              │
             ┌────────────────┴────────────────┐
             │                                 │
        Geom_viz.Primitive              Geom_viz.Cga / Pga
             │                                 │
        Geom_viz.Camera (PGA) ────────► Geom_viz.Project
             │                                 │
        Geom_viz.Scene ──► Geom_viz.Draw ──► Geom_viz.Frame
             │                                 │
        Geom_viz.Anim                     Hugin_vg.Renderable
             │                                 │
        Geom_viz.Movie            hugin.vg.raster / svg / pdf
```

| Module | Role |
|---|---|
| `Primitive` | neutral 3D primitives, `p3`, rigid transforms by a motor |
| `Camera` | eye/target/up, view motor, image plane, PGA projection, rays, perspective/orthographic |
| `Project` | clipping and projection of primitives, depth keys, screen coordinates |
| `Style` / `Theme` | colours, stroke widths, dashes, opacity, palettes |
| `Scene` | entities (primitive + style + layer + time-dependent motor), camera, background |
| `Draw` | evaluate a scene at a time, project, depth-sort, build a `Hugin_vg.Picture.t` |
| `Frame` | page size + density; RGBA, PNG, SVG, PDF output |
| `Anim` | timeline, keyframes, ScLERP, sampling |
| `Movie` | frame sequence, `ffmpeg` invocation, MP4/GIF |
| `Cga` | `Cga.t` → `Primitive.t` adapters (point, sphere, plane, line, circle, point pair) and motor frames |
| `Pga` | `Pga.t` → `Primitive.t` adapters and motor frames |

The package is `geom-viz` (library `geom_viz`), separate from `geom`, so the
core algebra does not depend on Hugin. `geom-viz` depends on `geom`, `hugin.vg`,
`hugin.vg.raster`, `hugin.vg.svg`, `hugin.vg.pdf`, `hugin.gg`, `hugin.font` and
`nx` (transitively `nx.io` for PNG). The movie encoder uses `bos` or `unix`
and the external `ffmpeg`; it can live in an optional `geom-viz.movie`
sublibrary so that the pure frame renderer has no process dependency.

## 5. API sketch (provisional)

```ocaml
(* primitive.ml *)
type p3 = { x : float; y : float; z : float }
type box3 = { min : p3; max : p3 }
type t =
  | Point of p3
  | Segment of p3 * p3
  | Polyline of p3 list
  | Polygon of p3 list
  | Circle of { center : p3; normal : p3; radius : float }
  | Sphere of { center : p3; radius : float }
  | Plane of { normal : p3; offset : float; patch : p3 list }
  | Frame of { origin : p3; x : p3; y : p3; z : p3 }

val transform : Motor.t -> t -> t
val bounds : t -> box3
val of_cga_point : Cga.t -> t
val of_cga_sphere : Cga.t -> t
val of_cga_plane : Cga.t -> t
val of_cga_line : Cga.t -> t
val of_cga_circle : Cga.t -> t
val of_pga_point : Pga.t -> t
val of_pga_plane : Pga.t -> t
val of_pga_line : Pga.t -> t

(* camera.ml *)
type t
val create :
  ?orthographic:bool ->
  eye:p3 -> target:p3 -> up:p3 ->
  fov_y:float -> near:float -> far:float ->
  width:float -> height:float ->
  t
val view : t -> Motor.t
val image_plane : t -> Pga.t
val project : t -> p3 -> (float * float * float) option
  (* page x, page y, depth; None if behind the near plane *)
val ray : t -> x:float -> y:float -> Pga.t
val orbit : t -> yaw:float -> pitch:float -> distance:float -> t
val look_at : p3 -> p3 -> p3 -> t

(* style.ml *)
type t = {
  fill : Hugin_gg.Color.t option;
  stroke : Hugin_gg.Color.t option;
  width : float;
  dash : float list;
  opacity : float;
}
val default : t
val solid : ?alpha:float -> Hugin_gg.Color.t -> t
val outline : ?width:float -> Hugin_gg.Color.t -> t

(* scene.ml *)
type entity = {
  primitive : Primitive.t;
  style : Style.t;
  layer : int;
  motion : float -> Motor.t;
}
type t = {
  camera : Camera.t;
  background : Hugin_gg.Color.t;
  entities : entity list;
}
val static : ?layer:int -> ?style:Style.t -> Primitive.t -> entity
val moving : ?layer:int -> ?style:Style.t -> motion:(float -> Motor.t) -> Primitive.t -> entity
val at : t -> float -> entity list
val add : entity -> t -> t
val with_camera : Camera.t -> t -> t

(* draw.ml *)
val picture : Scene.t -> time:float -> Hugin_vg.Picture.t
val render : Scene.t -> time:float -> Hugin_vg.Renderable.t

(* frame.ml *)
type t = {
  page : Hugin_vg.Renderable.t;
  density : float;
}
val create : ?density:float -> width:float -> height:float -> Hugin_vg.Picture.t -> t
val of_scene : Scene.t -> time:float -> t
val rgba : t -> Nx.uint8_t
val png : t -> string
val svg : t -> string
val pdf : t -> string
val save_png : t -> string -> unit
val save_svg : t -> string -> unit
val save_pdf : t -> string -> unit

(* anim.ml *)
val keyframes : (float * Motor.t) list -> float -> Motor.t
val constant_screw : Motor.t -> Motor.t -> float -> Motor.t
val timeline : duration:float -> fps:int -> (float -> Scene.t) -> Scene.t list

(* movie.ml *)
type spec = {
  duration : float;
  fps : int;
  size : int * int;             (* pixels, or points + density *)
  density : float;
  scene : float -> Scene.t;
}
val frames : spec -> Frame.t Seq.t
val write_png_sequence : spec -> dir:string -> prefix:string -> unit
val to_mp4 : ?crf:int -> spec -> path:string -> unit
val to_gif : ?fps:int -> spec -> path:string -> unit
```

The first `.mli` pass will settle the exact types (float vs `Nx` coordinates,
`Seq.t` vs arrays, labelled arguments) and whether `Frame` and `Movie` live in
separate sublibraries.

## 6. Animation and movies

### 6.1 Scene as a function of time

A `Scene.entity` carries `motion : float -> Motor.t`. A robot arm example
computes joint motors from a trajectory and composes them; a simple animation
uses a keyframe interpolator. `Scene.at scene t` applies each motion to its
primitive and returns the static entities to draw.

Keyframes interpolate motors with the group exponential:

```
M(λ) = exp(λ log(M0⁻¹ M1)) M0
```

which is ScLERP. `Anim.keyframes` wraps this over a sorted list, with optional
smoothstep easing between keys. If the core `Motor.slerp` is implemented with
the same formula, the two share tests.

A robot arm can be modelled with a chain of `Motor` products; the visualizer
only needs the frames and the link geometries (capsules/segments/spheres). The
first examples use a small 2-DOF or 6-DOF chain, not a full robot model.

### 6.2 Frame loop

```
for frame = 0 to n_frames - 1:
  t = frame / fps
  let scene_t = scene t
  let page = Draw.render scene_t
  write frame (Frame.png (Frame.create page))
```

The renderer is pure and immutable, so frames can be rendered in parallel with
`Domain.spawn` later. Encoding is sequential. The first implementation is
sequential; the benchmark decides whether parallel frame rendering is worth
it.

### 6.3 Encoding

The system `ffmpeg` is installed in this environment (9.0.2). Use the CLI:

- PNG sequence:
  ```
  ffmpeg -y -framerate 60 -i frame-%06d.png \
    -c:v libx264 -pix_fmt yuv420p -crf 18 movie.mp4
  ```
- raw RGBA pipe (later, avoids PNG encode/decode):
  ```
  ffmpeg -y -f rawvideo -pixel_format rgba -video_size 1280x720 \
    -framerate 60 -i - -c:v libx264 -pix_fmt yuv420p movie.mp4
  ```
- GIF:
  ```
  ffmpeg -y -framerate 30 -i frame-%06d.png \
    -vf "split[a][b];[a]palettegen[p];[b][p]paletteuse" movie.gif
  ```

`Movie.to_mp4` checks for `ffmpeg` with `Bos.OS.Cmd.find_tool`, writes frames to
a temporary directory, runs the encoder, and reports a clear error if the tool
is missing. A `Movie.write_frames` API lets the user encode with their own
pipeline. `ocaml-ffmpeg` is not a v1 dependency: the CLI is simpler, avoids a
heavy C dependency, and is exactly the kind of offline batch job the movie
pipeline is.

### 6.4 Interactive preview (later)

If interactive preview is needed, the minimal route is SDL2 (`tsdl`): render a
frame with `Hugin_vg_raster.render`, upload the RGBA tensor to a texture, blit
it, and handle space/arrow keys for play/pause/scrub. A camera orbit can be
driven by mouse drag. `Bogue` would add sliders and a timeline but is a larger
dependency. Raylib is a separate real-time 3D backend, not a Hugin renderer;
it is not part of this plan unless real-time 3D becomes a requirement.

## 7. Object mapping and visual defaults

| GA object | Renderer primitive | Default |
|---|---|---|
| CGA point | `Point` | filled dot, dark ink |
| CGA point pair | two `Point`s + `Segment` | dots and a thin dashed segment |
| CGA line | `Segment` (clipped) or `Polyline` | 2 pt stroke, dark blue |
| CGA circle | `Circle` (sampled) | 2 pt stroke, teal; optional translucent fill |
| CGA sphere | `Sphere` (silhouette) | translucent blue fill, darker rim |
| CGA plane | `Plane` patch + grid | very translucent grey/blue, thin grid |
| PGA point/line/plane | same primitives | same defaults |
| Motor | `Frame` | origin dot; x/y/z arrows red/green/blue |
| Dilator/transversion | transformed objects, ghosts | show before/after with lower opacity |
| Trajectory | `Polyline` of sampled positions | fading trail, thin |

Labels (point names, axis labels, time) use `Hugin_font`/`Hugin_text` runs and
`Picture.glyphs`. They are a V4 polish item: the first movies can be
label-free. The camera's screen origin and text placement are in page points,
so labels sit naturally on the page.

## 8. Milestones

**V0 — plan (this document).** Confirm Hugin_vg, the PGA camera and the
movie pipeline. A throwaway probe has already validated the rendering path.

**V1 — static frames.** `Primitive`, `Camera` (PGA + pinhole fast path),
`Project` for points/segments/polygons, `Style`, `Frame` (PNG/SVG/PDF), a
static scene example. Tests: projection identities, near clipping, output
dimensions, SVG/PDF non-empty. If `Pga` is not ready, the pinhole path unblocks
V1; the PGA meet is the reference added as soon as `Pga` lands.

**V2 — GA objects.** CGA/PGA adapters for points, spheres, planes, lines,
circles, point pairs; sphere silhouettes; plane patches/grids; motor frames;
depth sorting; a static "object zoo" example (a point, sphere, plane, line and
circle in one scene).

**V3 — animation and movies.** `Anim` keyframes/ScLERP; `Movie` PNG sequence
and MP4; first movies: a rotating/translating sphere, a moving motor frame with
a trail, a screw interpolation. A 6-DOF arm example comes once the core motor
chain is stable.

**V4 — polish.** Labels and arrows, trails, orthographic camera, camera orbit,
themes, sphere shading via image/clip, GIF encoding, notebook RGBA previews,
publication SVG/PDF.

**V5 — optional.** Interactive SDL preview; parallel frame rendering; a
z-buffer or GA ray tracer for correct occlusion; a Cairo2 backend if gradients
or raster speed demand it.

## 9. Test plan

- **Projection**: a point on the optical axis projects to the page centre; an
  off-axis point matches the pinhole formula; a point behind the camera is
  rejected; orthographic mode scales without division; the PGA meet and the
  closed-form projection agree to 1e-9.
- **Clipping**: a segment crossing the near plane is cut at the right point; a
  polygon with vertices behind the camera is clipped; a circle crossing the
  near plane does not produce a wrap-around segment.
- **Depth**: two spheres, one in front of the other, produce the front colour
  at their overlap; a plane behind a sphere is hidden.
- **Objects**: a sphere silhouette is a closed polygon enclosing the projected
  centre; a projected circle is closed; a motor frame's axes are orthonormal
  in world space and their projections meet at the origin.
- **Animation**: keyframes at `t=0` and `t=duration` give the exact endpoint
  motors; a constant-screw interpolation has constant twist; a full rotation
  returns to the start.
- **Determinism**: the same frame rendered twice gives the same PNG bytes
  (Hugin guarantees this); small golden images are stable within a renderer
  version.
- **Movie**: a 10-frame 320×180 sequence encodes to an MP4 whose duration and
  frame count `ffprobe` reports; the test skips if `ffmpeg` is absent.
- **Performance**: time per 1280×720 frame for a scene with a plane, a sphere,
  a line, a circle and 100 points; target at least a few frames per second
  offline; measure before optimizing.

## 10. Open questions

1. **Package split**: separate `geom-viz` package (recommended) or a `geom.viz`
   sublibrary in the `geom` package? The latter makes `geom` depend on Hugin.
2. **Camera math**: keep the PGA meet as the reference and the pinhole formula
   as the fast path, or only one? Recommendation: both, with an equivalence
   test.
3. **Primitive layer vs. raw GA objects**: neutral primitives (recommended)
   keep the renderer model-agnostic; raw CGA/PGA objects would avoid adapters
   but couple the renderer to the algebra.
4. **Sphere shading**: flat fill first; radial-gradient image/clip later?
5. **Movie encoding**: PNG sequence + `ffmpeg` CLI (recommended) vs. raw RGBA
   pipe vs. `ocaml-ffmpeg`. Start with PNG; add the pipe if encoding is the
   bottleneck.
6. **Interactive preview**: none in v1; Tsdl later, Bogue only if a real GUI is
   needed.
7. **Labels**: use `Hugin_font`/`Hugin_text` directly, or add a small text
   helper? Defer to V4.
8. **Hidden surfaces**: painter's algorithm is enough? If intersecting planes
   look wrong, add a z-buffer or a ray-traced backend later.
9. **Coordinate conventions**: confirm `+y` up in the camera and `y`-down on
   the page; document the flip in one place.
10. **Core API dependencies**: `Motor.of_rotation` (or `of_quaternion`), and
    the CGA/PGA extractors (`sphere_center`, `sphere_radius`, `plane_normal`,
    `plane_offset`, `line_direction`, `circle_center`, `circle_radius`,
    `circle_normal`, `Pga.point_coords`, …) must exist in the core API. Add
    them to the `.mli` pass now, even if the viz implementation comes later.
11. **3D scene bounds / auto-framing**: add a `Camera.fit : Primitive.t list ->
    Camera.t` helper? Nice for examples; not essential.
12. **Output determinism across machines**: Hugin is pure OCaml and
    deterministic, but font and raster versions can change; golden tests should
    be tolerant or version-pinned.

## 11. Dependencies

- `geom` — CGA, PGA, motors, extractors.
- `hugin.vg`, `hugin.vg.raster`, `hugin.vg.svg`, `hugin.vg.pdf`,
  `hugin.gg`, `hugin.font` — the rendering stack. `hugin.vg.raster` brings
  `nx.io` for PNG.
- `nx` — image tensors, RGBA output, notebook display.
- `bos` or `unix` — spawning `ffmpeg` in `geom-viz.movie`.
- optional `tsdl` — interactive preview.
- optional `hugin` (figure API) — insets/legends, if we want them.

No Cairo, SDL, OpenGL or FFmpeg OCaml bindings are required for the core
visualizer.

## Appendix A — validated Hugin_vg snippet

The following is the essence of the probe that rendered successfully in this
workspace (640×360 PNG and SVG):

```ocaml
open Hugin_gg
open Hugin_vg

let () =
  let page_box = Box2.v 0. 0. 640. 360. in
  let bg = Picture.fill (Color.v 0.97 0.97 0.98) (Path.rect page_box) in
  let sphere =
    Picture.fill (Color.v ~alpha:0.85 0.20 0.52 0.86)
      (Path.circle (P2.v 300. 180.) 70.)
  in
  let rim =
    Picture.stroke (Stroke.v 2.0) (Color.v 0.10 0.28 0.55)
      (Path.circle (P2.v 300. 180.) 70.)
  in
  let page =
    Renderable.v 640. 360.
      (Picture.group [ bg; sphere; rim ])
  in
  let png = Hugin_vg_raster.png ~density:1. page in
  let svg = Hugin_vg_svg.render page in
  ignore (png, svg)
```

The actual probe also drew a grid, a stroked segment and a dot. The point is
that a scene built from `Hugin_vg.Picture` values maps directly onto the
primitives the GA renderer produces, and both raster and vector output work
without C dependencies.
