# Layout and output

A figure combines into bigger figures. This file covers layers, grids,
facets, sizes and themes, the three rendering stages, views, custom marks
and the ways a figure leaves the program.

## Composition

```ocaml
val layer : t list -> t
val grid :
  ?widths:float list -> ?heights:float list -> t list list -> t
val span : ?rows:int -> ?cols:int -> t -> t
val share : (string * [ `Shared | `Independent ]) list -> t -> t
val title : ?align:Text.Layout.halign -> string -> t -> t
val title' : ?align:Text.Layout.halign -> Text.t -> t -> t
val coord : Coord.t -> t -> t
val name : string -> t -> t
val bind : 'a View.key -> ('a -> t) -> t
```

### `layer`

Draws its children over one another, the first at the bottom. The children
share every scale, so a line, a band and a reference rule share axes and
legends:

```ocaml
layer
  [
    area ~x:(num t) ~y:(num (signal 1.)) ~opacity:(const 0.2) ();
    line ~x:(num t) ~y:(num (signal 1.)) ();
    rule ~y:(floats [| 0. |]) ~dash:(const Dash.dashed) ();
  ]
```

A mark that draws nothing is a valid layer child; `layer []` is an empty
cell.

### `grid`

Arranges figures in rows of cells, from the top and from the left. The data
areas of a column share their left and right edges, and those of a row their
top and bottom edges, whatever the width of the tick labels.

```ocaml
grid [ [ a; b ]; [ c; d ] ]
grid ~heights:[ 2.; 1. ] [ [ span ~cols:2 a ]; [ b; c ] ]
```

`~widths` and `~heights` weigh the columns and rows. A weight is not a
length: the tracks share what the page leaves, each taking at least what it
needs. `span ~cols:2 f` makes `f` a cell two columns wide.

The cells of a grid keep their own x and y scales and share every other
scale. A figure therefore has one colour legend however it is arranged.

### `share`

Changes the sharing of one scale name at one node:

```ocaml
grid [ [ a ]; [ b ] ] |> share [ ("x", `Shared) ]
```

With one x scale, the two cells line up on one domain and the axis is
labelled once, below the bottom cell. `share [ ("color", `Independent) ]`
gives each cell of a grid its own colour scale and legend.

### `title`

Sets a bold title above the figure, aligned with the left edge of its data
areas by default. Titles nest; a grid of titled cells labels each panel:

```ocaml
grid [ [ title "(a)" a; title "(b)" b ] ]
|> title ~align:`Center "Two panels"
```

`title'` takes a `Text.t` for rich text.

### `coord`

Sets the coordinate system of the panels:

```ocaml
coord (Coord.cartesian ~aspect:1. ())
```

`~aspect:1.` makes one unit of y as long as one unit of x: useful for
trajectories, calibration plots, correlation plots and heatmaps whose cells
should be square. Without `~aspect`, the panel takes the box the layout
gives it. A panel that cannot hold its aspect is drawn in the largest box of
that aspect, centred in its cell.

A mark may imply a coordinate system; `image` implies
`Coord.cartesian ~aspect:1. ()`. An explicit `coord` above it wins, and two
different implied systems in one panel raise at resolve.

### `name` and `bind`

`name "series"` replaces the node's index in its id with a stable string, so
warnings and ids do not shift when a sibling is added. `bind key f` rebuilds
`f` from a view value at resolve time; see [Views](#views).

## Facets

`~fx` and `~fy` split one mark into panels, one column per `fx` category
and one row per `fy` category. Facet scales are band scales; `~wrap:n`
wraps the columns into rows of `n`. Facet panels share every scale, and a
mark without facet channels, layered with a faceted one, is drawn in every
panel.

```ocaml
line ~x:(num t) ~y:(num curves) ~fx:(dim ~title:"seed" 0) ()
line ~x:(num t) ~y:(num curves)
  ~fx:(dim ~scale:(Scale.band ~wrap:2 ()) 0) ()
```

## Sizes

```ocaml
type t
val figure : float -> float -> t
val panels : float -> float -> t
val mm : float -> float
val dpi : float -> float
```

Lengths are in points; one point is 1/72 inch.

- `Size.figure w h` fixes the page. The panels share what axes, titles,
  legends and gaps leave. If the page is too small, `layout` raises and names
  a size at which the figure fits.
- `Size.panels w h` fixes each data area and lets the page grow. In a grid,
  a column of weight `k` gets a data area `k *. w` points wide.
- `Size.mm 85.` is 85 millimetres; `Size.dpi 300.` is a density of 300
  pixels per inch.

The default is `Size.figure 360. 240.` at density 2, so a PNG is 720 by 480
pixels. A paper column is about 85 mm wide, a double column about 180 mm.
Set the height from the panel shape the data needs, not from a fixed aspect:
a time series is wider than tall, a calibration plot is square.

## Themes

A theme holds what a figure does not state: ink, paper and accent colours,
the base size, the faces, the default schemes and the locale.

```ocaml
val v :
  ?ink:Color.t -> ?paper:Color.t -> ?accent:Color.t ->
  ?size:float -> ?fonts:Font.t list ->
  ?palette:Scheme.t -> ?scheme:Scheme.t -> ?locale:Locale.t ->
  unit -> t
```

| Argument | Default |
| --- | --- |
| `ink` | `Color.gray 0.1` |
| `paper` | `Color.white` |
| `accent` | the first colour of the palette |
| `size` | `10.` points |
| `fonts` | `[ Font.regular; Font.bold ]` |
| `palette` | `Scheme.tableau10` |
| `scheme` | `Scheme.viridis` |
| `locale` | `Locale.default` |

The faces are bundled with Hugin, so a figure needs no system font. Every
other length is a multiple of the base size: tick and legend labels at
0.9 em, channel titles at 1 em, figure titles at 1.2 em in bold, lines and
rules at 0.15 em wide, ticks 0.35 em long, legend swatches 0.8 em square,
and a margin of 0.5 em around the page.

Presets are ordinary themes, combined through their accessors:

```ocaml
Theme.dark     (* light ink on dark paper, a scheme that contrasts with it *)
Theme.talk     (* base size 16, for slides *)
Theme.poster   (* base size 20 *)

let slides =
  Theme.(v ~ink:(ink dark) ~paper:(paper dark) ~scheme:(scheme dark)
           ~size:(size talk) ())
```

A theme is read by `layout` and `draw`, not by `resolve`, so fitted scales
never depend on it. The same resolved figure can be laid out in several
themes:

```ocaml
let paper = Theme.v ~size:8. ~palette:Scheme.okabe_ito () in
save ~theme:paper ~size:(Size.figure (Size.mm 85.) (Size.mm 55.)) "a.pdf" fig;
save ~theme:Theme.talk ~size:(Size.figure 576. 384.) "a-slide.png" fig
```

`Theme.talk` raises every length but does not enlarge the page. Give it a
page proportionally larger, or the layout has less room than the text needs.

## The three stages

```text
figure ──resolve──▶ Resolved.t ──layout──▶ Layout.t ──draw──▶ Drawing.t
                  fitted scales      panels, ticks       picture
```

```ocaml
val resolve : ?prev:Resolved.t -> ?view:View.t -> t -> Resolved.t
val layout : ?prev:Layout.t -> ?theme:Theme.t -> Size.t -> Resolved.t -> Layout.t
val draw : ?prev:Drawing.t -> density:float -> Layout.t -> Drawing.t
```

- `resolve` applies the view and `bind`s, assigns ids, forms scopes, checks
  the composition, merges and fits the scales. It reads neither size, theme
  nor density.
- `layout` measures text in the theme's faces, chooses and freezes ticks,
  solves the grid, places the guides and builds each panel's projection.
- `draw` reduces marks where their data lives, reads each channel once as
  float64, calls each mark's draw function and paints the paper, panels and
  guides.

`save` runs all three. Open them when you need the pieces:

```ocaml
let resolved = resolve fig in
let y = Resolved.scale resolved (Scale.linear ~name:"y" ()) in
# Scale.domain y;;
- : float Scale.domain = Scale.Floats (0.1, 1.1)

let laid = layout (Size.figure 360. 240.) resolved in
let drawing = draw ~density:2. laid in
```

`Resolved.warnings`, `Layout.warnings` and `Drawing.warnings` list the
warnings of the stages, in the order of the figure.

Each stage takes `~prev` and returns it when its inputs are equal, which
makes redraw-on-change cheap:

```ocaml
let again = resolve ~prev:resolved fig in
# again == resolved;;
- : bool = true
```

Figures compare tensors physically. A tensor changed in place between two
resolves is not seen; build a new figure or pass a fresh tensor.

## Output

### Files

```ocaml
val save :
  ?warn:(warning -> unit) -> ?view:View.t -> ?theme:Theme.t ->
  ?size:Size.t -> ?density:float -> string -> t -> unit
```

`save file fig` writes the format the extension names, ignoring case:
PNG for `.png`, SVG for `.svg`, PDF for `.pdf`. PNG is drawn at `~density`
and records the density and the sRGB colour space. `size` defaults to
`Size.figure 360. 240.` and `density` to `2.`.

`~warn` receives each warning before the file is written; by default each
is formatted with `pp_warning` on standard error. `~view`, `~theme`,
`~size` and `~density` are the same arguments as the stages take.

A wrong extension raises `Invalid_argument`; an unwritable file raises
`Sys_error`.

### Without a file

`Drawing.renderable` is the picture on its page. The renderer libraries turn
it into pixels or documents:

```ocaml
(* (libraries hugin hugin.vg.raster hugin.vg.svg hugin.vg.pdf) *)
let drawing = draw ~density:2. (layout (Size.figure 360. 240.) (resolve fig)) in
let r = Drawing.renderable drawing in
let pixels = Hugin_vg_raster.render ~density:2. r in   (* [|h; w; 4|] uint8 *)
let png = Hugin_vg_raster.png ~density:2. r in         (* PNG bytes *)
let svg = Hugin_vg_svg.render r in                     (* SVG text *)
let pdf = Hugin_vg_pdf.render r in                     (* PDF bytes *)
```

`Hugin_vg_raster.render` returns RGBA pixels with straight alpha on the
host. Pixels the picture paints nothing on have alpha 0. The SVG and PDF
renderers return strings.

### Notebooks

`Hugin.pp` formats a figure for a Quill notebook: it renders the figure as
SVG and hands the document to the display protocol. Elsewhere the formatter
prints one line, `hugin figure`, with a warning count if there are warnings.

```ocaml
let fig = Hugin.(line ~y:(num curves) ())
```

## Views

A view holds values a viewer sets on a figure, such as a slider or a zoom. A
key names a value; `bind` builds the part of the figure that reads it.

```ocaml
let smoothing = View.number "smoothing" ~init:0.5 in
let live =
  bind smoothing (fun alpha -> line ~y:(num (Nx.ewma ~alpha noisy)) ())
in
save ~view:View.(set smoothing 0.1 empty) "smoothed.png" live
```

`View.number`, `View.choice` and `View.interval` make keys of a float, a
string and an interval. Every continuous scale has a zoom key without
declaring one:

```ocaml
let x_zoom = View.zoom (Scale.linear ~name:"x" ()) in
save ~view:View.(set x_zoom (Some (0., 20.)) empty) "zoomed.png" fig
```

Views live outside figures and keys are identified by name, so a figure
rebuilt by the same code reads the values set on the previous one.
`resolve` applies `bind` once per call.

## Custom marks

A mark is a list of channel bindings and a function that draws one panel's
rows as a picture. `Mark.v` makes one.

```ocaml
val Mark.v :
  name:string -> ?reduce:Mark.reducer -> ?coord:Coord.t ->
  ?shape:int array -> ?swatch:(Mark.rows -> Picture.t) ->
  Mark.binding list -> (Mark.rows -> Picture.t) -> t

val Mark.bind :
  ?imply:'d Scale.t -> ?guide:bool -> ?stack:Stack.offset ->
  ('d, 'r) Role.t -> ('d, 'r) channel -> Mark.binding
```

Positions from `Mark.points` are in points on the page, with y pointing
down. A dropped row, or a row outside the domain, is at NaN; a path through
NaN leaves a gap, so the draw function needs no check of its own.

```ocaml
let angle = Role.value ~name:"angle"

let needles ~x ~y ~angle:a () =
  Mark.v ~name:"needles"
    [ Mark.bind Role.x x; Mark.bind Role.y y; Mark.bind angle a ]
    (fun rows ->
      let xs, ys = Mark.points rows in
      let th = Option.get (Mark.get rows angle) in
      let ink = Theme.ink (Mark.theme rows) in
      let needle i =
        Picture.stroke (Stroke.v 1.) ink
          (Path.polyline
             [| xs.(i); xs.(i) +. (10. *. cos th.(i)) |]
             [| ys.(i); ys.(i) -. (10. *. sin th.(i)) |])
      in
      Picture.group (List.init (Mark.length rows) needle))

needles ~x:(num gx) ~y:(num gy) ~angle:(num theta) ()
```

`Mark.get`, `Mark.normalized`, `Mark.range`, `Mark.positions`, `Mark.extent`
and `Mark.series` read a panel's rows. `Mark.scale` and `Mark.ticks` read
the fitted scale and frozen ticks, `Mark.theme` reads the theme, and
`Mark.text` sets text in its faces. `Mark.warn` adds a data warning under
the mark's id.

A custom mark layers, facets and fits the x and y scales like a built-in
one. `~reduce` names a reducer for dense rows (`Mark.m4`, `Mark.cells`,
`Mark.raster`). `~swatch` draws its legend entry; unset, the draw function
is called with one row in a one-em box.

## Warnings and ids

A warning is `id * string`: the node whose data a problem concerns, and a
message for users. `pp_warning` formats it.

```ocaml
let warnings = ref [] in
save ~warn:(fun w -> warnings := w :: !warnings) "fig.png" fig;
List.iter (Format.printf "%a@." pp_warning) (List.rev !warnings)
```

Ids are paths from the root of the figure, so a warning names the mark,
layer or cell at fault. `name "series"` replaces an index in the path with a
stable string. Structural mistakes raise `Invalid_argument` at construction
or resolve and name the nodes; data problems are warnings, and the figure
is still drawn. No value is substituted for a missing one.

`Hugin.pp` counts drawing warnings in its summary line. In a test, give
`~warn` and assert on the list rather than reading standard error.
