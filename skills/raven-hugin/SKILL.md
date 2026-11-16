---
name: raven-hugin
description: "Draw scientific figures from Nx tensors with Hugin in the Hennequin lab monorepo (murder). Use for scientific plotting in OCaml: line, dot, area, rect, rule, text, image or contour marks; channels, rows and scales; axes, legends and colour schemes; facets, grids, themes and paper-sized PNG, SVG or PDF output. Triggers: Hugin., open Hugin, save, num, cat, dim, index, Scale.linear, Scale.log, Scale.band, Scheme.viridis, Theme.v, Size.mm, legend, axis, heatmap, histogram, error bars, publication figure."
---

# Hugin

Hugin draws figures from Nx tensors. A figure is an immutable value: marks
read tensors through channels, scales fit and place their values, and axes
and legends follow from the scales. A figure renders to the same bytes on
every run.

Read this file first. The references carry the API detail:

- `references/marks.md` — channels, rows, every mark, stacks, facets.
- `references/scales-and-guides.md` — scales, colour schemes, axes,
  legends, ticks, time, histograms.
- `references/layout-and-output.md` — layers, grids, sizes, themes, the
  three rendering stages, views, custom marks.
- `references/scientific-recipes.md` — training curves, error bars,
  histograms, heatmaps, contours, image grids, paper panels.

For the tensors themselves see `raven-nx`; for device selection and
placement see `raven-rig`.

## House rules

- A figure is a value. Build it once with `|>` and end with `save`. There is
  no mutable plot object.
- `open Hugin` in a figure module and qualify Nx (`Nx.linspace`, `Nx.sin`).
  In a module that is mostly numerical, keep `open Nx` and qualify the Hugin
  calls (`Hugin.dim`, `Hugin.num`), or open Hugin after Nx and write
  `Nx.dim` for the axis size. Both define `dim`; the last open wins.
- Every channel over a tensor takes a host tensor. Move a device result back
  with `Nx.place Nx.Host.on` before plotting, outside the jitted function. A
  set minted by `Nx.devices [ Rig.host ]` is not the `Nx.host` brand that
  channels require.
- Plot at reporting boundaries. A figure keeps the tensor until it is drawn,
  but `Nx.item` and `Nx.to_array` force a wait; do not read a loss on every
  step to plot it.
- Data goes into channels: `~x:(num x)`, not `~x:x`. A mark already is a
  figure.
- A channel's `~title` titles the axis or legend. Put units there
  (`~title:"time (s)"`), and set it once per scale.
- Let the scale fit the data. Set `~domain` only when a limit is part of the
  comparison, and then use the same domain on every panel compared.
- Let marks imply their geometry: a bar starts at zero, a bar's band is
  padded, an image hides its axes and reverses y. Override with a reason.
- Paper figures: `Theme.v ~size:8. ()` and `Size.mm`. Slides:
  `Theme.talk` with a larger page. Screen: the defaults.
- Warnings are data problems, not structural errors. `save` prints them;
  collect them with `~warn` in tests and examples.
- Describe what the figure shows. Do not rank Hugin against another
  plotting library in comments or docs.

## The pipeline

```text
figure ──resolve──▶ Resolved.t ──layout──▶ Layout.t ──draw──▶ Drawing.t
                  scales fitted       panels, ticks       picture
```

`save` runs the three stages and chooses the format from the file extension:
PNG, SVG or PDF. `resolve` reads no size or theme, so one figure can be
resolved once and laid out at several sizes. Each stage takes `~prev` and
returns it unchanged when its inputs are equal. Link `hugin` in dune; it
re-exports the modules a figure uses.

Defaults: 360 by 240 points, density 2 pixels per point, `Theme.default`
(10 pt Inter, viridis, tableau10).

## Quick start

Five curves, one tensor of shape `[5; 200]`, coloured by run:

```ocaml
open Hugin

let steps = Nx.linspace Nx.float64 0. 1. 200

let losses =
  let rates = Nx.create Nx.float64 [| 5; 1 |] [| 2.; 2.5; 3.; 3.5; 4. |] in
  let noise = Nx.Rng.normal ~key:(Nx.Rng.key 0) Nx.float64 [| 5; 200 |] in
  Nx.add (Nx.exp (Nx.neg (Nx.mul rates steps))) (Nx.mul_s noise 0.01)

let () =
  line
    ~x:(num ~title:"progress" steps)
    ~y:(num ~title:"loss" losses)
    ~stroke:(dim ~title:"run" 0)
    ()
  |> title "Five runs"
  |> save "runs.png"
```

`num` reads a tensor as quantities. `dim 0` reads the index along axis 0 of
the mark's shape as a category, so each run gets a colour and a legend
entry. The mark's shape is `[5; 200]`; each of its 1,000 elements is a row.

## Channels and rows

A channel says how a mark reads data.

| Channel | Reads | Example |
| --- | --- | --- |
| `num t` | the elements of a tensor, as quantities | `num losses` |
| `cat ~labels t` | integer codes, as labelled categories | `cat ~labels classes codes` |
| `strings a` | strings, as categories | `strings methods` |
| `floats a` | floats, as quantities | `floats [| 0. |]` |
| `dim k` | the index along axis `k` of the mark's shape, as a category | `dim 0` |
| `index k` | the index along axis `k`, as a quantity | `index (-1)` |
| `const v` | one value for every row | `const Color.red` |

The channels of a mark broadcast together like Nx operands; every element
of the result is one row. A `[5; 200]` tensor with a `[200]` x is 1,000
rows of one mark, arranged as 5 series of 200. `line` draws one curve per
index along every axis but the last.

`dim` and `index` read the *mark's* shape, not the tensor passed to another
channel. The shape is the broadcast of all channels; check it when a `dim`
seems to name the wrong axis.

A value is missing if it is NaN or infinite, if `~valid` is false there, or
if the scale cannot place it, such as a non-positive value on a log scale.
A dropped row leaves a gap in a line; it is not replaced.

## Marks

| Mark | Draws | Needs |
| --- | --- | --- |
| `line ~y ()` | a curve per series | `y`; `x` defaults to `index (-1)` |
| `dot ~x ~y ()` | a symbol per row | `x`, `y` |
| `area ~y ()` | a filled region to a baseline | `y`; baseline zero, or `y2` |
| `rect ()` | a bar, heatmap cell, histogram bin or span | whichever extents apply |
| `rule ()` | a reference line, stem or error bar | `x` and/or `y`, with an end |
| `abline ~slope ~intercept ()` | the line y = a x + b | constants or channels |
| `text ~x ~y ~text ()` | a label per row | `x`, `y`, `text` |
| `image px` | a raster image, or one per leading index | a rank-2+ tensor |
| `contour ~fill ()` | filled isolines of a sampled field | `fill` |
| `frame ()` | the outline of each panel | nothing |

Common compositions:

```ocaml
(* Bars: a band x, a length y. *)
rect ~x:(strings ~title:"method" methods) ~y:(num ~title:"accuracy" mean) ()

(* Heatmap: rows and columns are categories, values are fill. *)
rect ~x:(dim ~title:"column" 1) ~y:(dim ~title:"row" 0)
  ~fill:(num ~title:"z" z) ()

(* Error bars: a rule from the lower to the upper end under a dot. *)
layer
  [
    rule ~x ~y:(num (Nx.sub mean err)) ~y2:(num (Nx.add mean err)) ();
    dot ~x ~y:(num ~title:"accuracy" mean) ();
  ]

(* Dense points: dots reduce themselves to an image of their rows. *)
dot ~x:(num px) ~y:(num py) ()
```

`line` takes `~curve` (`Curve.linear`, `Curve.step_after`,
`Curve.natural`, ...), `~dash`, `~width`, `~fill` and `~opacity`. `rect`
and `area` take `~stack`. `rect` with only a band `x` and a quantitative
`y` is a bar; with both `x` and `y` on band scales and a `fill`, it is a
heatmap.

## Scales and guides

A channel reads a scale by name, or else the scale of its role: `x` and
`y` for positions, `color` for `fill` and `stroke`, `size`, `width`,
`symbol`, `dash`, `opacity`, `fx`, `fy`.

```ocaml
Scale.linear ()                   (* proportion *)
Scale.log ()                      (* positive values, by log *)
Scale.symlog ()                   (* either sign *)
Scale.pow ~exponent:0.5 ()        (* by a power *)
Scale.band ()                     (* categories *)
```

Every constructor takes `~name`, `~domain`, `~reverse`, `~ticks`, `~scheme`
and `~unknown`. Quantitative scales also take `~nice`, `~clamp`, `~stepped`,
`~notation` and `~areas`; `linear`, `symlog`, `pow` and `custom` add `~zero`;
`band` takes `~padding`, `~wrap`, `~symbols` and `~dashes` instead; `time`
takes `~tz_offset_s`.

A scale a channel configures is the scale of its role. To put a limit on
the y scale, give the `~y` channel a `Scale.linear ~domain:(0., 1.) ()`.
A name of your own denotes one scale wherever it appears:

```ocaml
let unit = Scale.linear ~domain:(0., 1.) ~notation:Number.Percent in
line
  ~x:(num ~scale:(unit ~name:"x" ()) ~title:"predicted" p)
  ~y:(num ~scale:(unit ~name:"y" ()) ~title:"observed" observed)
  ()
```

Axes and legends are generated from the scales. `axis` and `legend` replace
one, and draw nothing themselves:

```ocaml
layer
  [
    dot ~x ~y ~fill:(cat ~labels:[| "a"; "b"; "c" |] codes) ();
    axis ~grid:true "y";
    axis ~show:false "x";
    legend ~side:`Bottom ~ncol:3 "color";
  ]
```

`axis ~grid:true` draws grid lines at the ticks.
``legend ~side:(`Inside `Top_right) "color"`` puts a legend in a panel
corner; choose a corner the data leaves empty. A legend and an axis are
titled by the distinct titles of the channels that read their scale; set
`~title` once.

Colours are constants; schemes belong to scales:

```ocaml
let teal = Color.v 0.1 0.5 0.55
let faded = Color.with_alpha 0.4 teal
let blend = Color.mix 0.5 teal Color.white

rect ~x:(dim 1) ~y:(dim 0)
  ~fill:(num ~scale:(Scale.linear ~scheme:Scheme.rdbu ()) z)
  ()
```

A continuous scale takes `~scheme` (viridis by default); a band scale takes
a palette (tableau10 by default). Use a sequential scheme for magnitudes, a
diverging one only when a meaningful midpoint exists, and a qualitative
palette for unordered categories.

## Composition

```ocaml
layer [ background; series; annotation ]   (* first at the bottom *)
grid [ [ a; b ]; [ c; d ] ]                (* rows, from the top *)
grid ~heights:[ 2.; 1. ] [ [ span ~cols:2 a ]; [ b; c ] ]
grid [ [ a ]; [ b ] ] |> share [ ("x", `Shared) ]
```

The children of a `layer` share every scale. The cells of a grid keep their
own x and y scales and share the rest, so a grid has one colour legend
unless `share` says otherwise. `share [ ("x", `Shared) ]` gives two stacked
panels one x domain and labels the axis once.

Facet channels split one mark into panels and share every scale:
`~fx:(dim ~title:"run" 0)` and `~fy:(cat ~labels:classes codes)`.
`Scale.band ~wrap:3 ()` wraps the panels into rows.

`title "..."` sets the figure title. `title "(a)"` labels a panel when the
titled figure is a grid cell. `coord (Coord.cartesian ~aspect:1. ())` makes
one unit of y as long as one unit of x, for trajectories, calibration plots
and square heatmap cells.

## Presentation

```ocaml
let print = Theme.v ~size:8. ~palette:Scheme.okabe_ito () in
grid [ [ title "(a)" a; title "(b)" b ] ]
|> save ~theme:print
     ~size:(Size.figure (Size.mm 85.) (Size.mm 55.))
     "figure.pdf"
```

- `Size.figure w h` fixes the page; `Size.panels w h` fixes each data area
  and lets the page grow. `Size.mm` converts millimetres.
- `Theme.v ~ink ~paper ~accent ~size ~fonts ~palette ~scheme ()` sets every
  length from the base size. `Theme.dark`, `Theme.talk` and `Theme.poster`
  are presets, combined through their accessors.
- `save ~density:(Size.dpi 300.)` writes a 300 dpi PNG. PDF and SVG are
  vector; use them for papers.
- `save ~warn:(fun w -> ...)` collects warnings instead of printing them.
  `pp_warning` formats one.

Typical widths: 85 mm for a single column, 180 mm for a double column,
with a height that gives the panels the shape the data needs.

## Reading a figure

Mistakes in structure raise `Invalid_argument` when the figure is made or
resolved, and name the node at fault: channels that do not broadcast, two
marks reading different x scales in one panel, a nested figure under two
coordinate systems. Data problems are warnings: an out-of-range category
code, a character no face has, a field that is not monotone. Nothing is
substituted; the figure is still drawn.

`Hugin.pp` renders SVG in a Quill notebook and prints a one-line summary
elsewhere. Use `Drawing.warnings` or `save ~warn` to see warnings from all
three stages.

## Pitfalls

- A channel takes a host tensor. `Nx.place Nx.Host.on x` moves a device
  value back; `Nx.devices [ Rig.host ]` is a different brand.
- `~x` and `~y` are channels. `~x:x` is a type error; write `~x:(num x)`.
- `const` for a position is a normalised fraction, `0.` to `1.`. For one
  data position use `floats [| x |]`.
- `dim k` reads the mark's broadcast shape, and `index k` reads it as a
  quantity. Neither reads an arbitrary tensor's axis.
- A `[5; 200]` line is five series of 200 points, not one series of 1,000.
  `line` draws along the last axis.
- Without `~x`, a line uses `index (-1)`, the positions 0 to n-1, not the
  values of another channel.
- `rule ~y` is horizontal and `rule ~x` is vertical. An error bar needs
  both ends: `~x` with `~y` and `~y2`, or `~y` with `~x` and `~x2`.
- Two marks in one panel must read the same x and y scales. Putting a named
  x scale on one mark and leaving the other on `"x"` raises at resolve.
- A categorical `~stroke` and `~dash` from different band scales yields two
  legends. Give both channels one named band scale.
- A `~title` set on two channels of one scale is joined with commas. Set it
  once, on the channel that owns the meaning.
- `image` implies no axes and a reversed y. Do not layer data coordinates
  over it unless you mean pixel coordinates.
- `contour` takes `x` per column and `y` per row: `[m]` and `[n; 1]` for an
  `[n; m]` field. Its levels are the fill scale's domain ends and ticks.
- `~domain` fixes limits and stops fitting. `~nice:false` keeps the fitted
  domain at the data's extent.
- A figure compares tensors physically. Mutating a tensor in place between
  resolves is not seen; build a new figure or pass a fresh tensor.
- `Theme.talk` enlarges text and every derived length. A slide also needs a
  larger `Size.figure`, or the layout has less room than the text needs.

## References

- `references/marks.md` — channel constructors, broadcast rules, every
  mark's arguments and defaults, stacks, facets, missing values.
- `references/scales-and-guides.md` — scale properties, names and scopes,
  colour schemes, axes, legends, ticks, notation, time.
- `references/layout-and-output.md` — composition, sizes, themes, `resolve`
  / `layout` / `draw`, output without files, views, custom marks.
- `references/scientific-recipes.md` — the figures the lab draws, with
  working code and the style choices behind them.

The pinned sources and Hugin's own `doc/` and `examples/` are under
`_build/_private/default/.pkg/hugin.dev-*/source/packages/hugin/`.
