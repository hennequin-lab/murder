# Marks and channels

A mark draws rows of data. A channel says how the mark reads its data, and a
role names the slot the channel fills. This file covers the channel
constructors, how channels broadcast into rows, and each built-in mark.

## The channel type

```ocaml
type ('d, 'r) channel
```

`'d` is the domain type of the values: `float` for quantities, `Time.t` for
instants, `string` for categories. `'r` is the range of the role they are
bound to: `float` for positions, `Color.t` for colours, `Symbol.t`,
`Dash.t`, `Text.t` or `string`. A channel bound with `let` serves the one
range its first use fixes.

Every channel over data takes a **host** tensor (`Nx.host Nx.t`). Hugin
reads it at draw time, not when the channel is made. A value on another
device set moves back with `Nx.place Nx.Host.on`:

```ocaml
let host_loss = Nx.place Nx.Host.on loss
```

## Constructors

```ocaml
val num :
  ?scale:float Scale.t ->
  ?valid:Nx.host Nx.bool_t ->
  ?title:string ->
  ('a, 'b, Nx.host) Nx.t -> (float, 'r) channel

val cat :
  ?scale:string Scale.t ->
  ?valid:Nx.host Nx.bool_t ->
  ?title:string ->
  ?labels:string array ->
  ('a, 'b, Nx.host) Nx.t -> (string, 'r) channel

val strings :
  ?scale:string Scale.t -> ?title:string ->
  string array -> (string, 'r) channel

val floats :
  ?scale:float Scale.t -> ?title:string ->
  float array -> (float, 'r) channel

val dim :
  ?scale:string Scale.t -> ?valid:Nx.host Nx.bool_t ->
  ?title:string -> ?labels:string array ->
  int -> (string, 'r) channel

val index :
  ?scale:float Scale.t -> ?title:string ->
  int -> (float, 'r) channel

val const : 'r -> ('d, 'r) channel

val map_range : ('r -> 'r) -> ('d, 'r) channel -> ('d, 'r) channel
```

### `num`

Reads the elements of a tensor as quantities. The dtype must be an integer
or a float; booleans and complex numbers raise. An integer beyond 2^53 is
rounded to a float. `~valid` is a boolean mask that broadcasts to the
tensor's shape without growing it; false elements are missing. `~title`
titles the scale's guide.

### `cat`

Reads integer codes as categories.

- With `~labels`, code `i` is the category `labels.(i)`. Every label joins
  the scale's domain, used or not, so colours stay put when data is
  filtered. A code outside `[0; length - 1]` is missing, with a warning.
- Without `~labels`, code `i` is an indexed category shown by another
  channel on the same scale (`dim`) or in decimal.

Codes must have an integer dtype, and labels must be distinct.

### `strings` and `floats`

Take an OCaml array rather than a tensor. `strings` has shape `[|n|]` and
adds its distinct strings to the scale's domain in order of first
appearance. `floats` has shape `[|n|]`; NaN and infinities are missing.

```ocaml
rule ~y:(floats [| 0. |]) ~dash:(const Dash.dashed) ()
```

### `dim` and `index`

Read an axis of the **mark's broadcast shape**, not of a tensor passed to
another channel.

- `dim k` is the index along axis `k`, as a category. A negative `k` counts
  from the last axis. `~labels` may repeat; the integer identifies the
  category. `dim` contributes every index of its axis to the domain, used
  or not.
- `index k` is the index as a quantity. A negative `k` counts from the last
  axis. It holds no tensor and does not take part in broadcasting.

```ocaml
(* [curves] is [5; 200]: five series along the last axis. *)
line ~y:(num curves) ~stroke:(dim 0) ()

(* Without ~x, a line reads index (-1): 0 .. 199. *)
line ~y:(num curves) ~x:(index ~title:"step" (-1)) ()
```

`dim` raises when the mark is made if its shape has no axis `k`, or if the
labels differ in length from that axis.

### `const`

One value for every row. A constant reads no scale, contributes to no
domain and yields no guide. Its meaning is fixed by the role: a position is
a normalised fraction from `0.` to `1.`, a colour is a `Color.t`, a size is
an area in square points.

```ocaml
~x:(const 0.5)          (* the middle of the panel, not x = 0.5 in data *)
~stroke:(const Color.red)
~size:(const 9.)
~dash:(const Dash.dashed)
~text:(const (Text.v "reference"))
```

### `map_range`

Applies a function after the role has mapped the value, and leaves missing
values alone. The channel still fits its scale, but a scale read only
through `map_range` yields no legend.

```ocaml
(* Text on a coloured cell, in the colour that reads on it. *)
text ~x:(dim 1) ~y:(dim 0) ~text:(num counts)
  ~fill:(map_range Color.contrast share) ()
```

### `kind` and `varies`

`kind c` reports the kind of scale the channel reads, or `None` for a
constant. `varies shape c axis` tells whether the channel can differ along
an axis of a mark of that shape. Both are for writing marks, not for
everyday figures.

## Rows, shapes and series

The channels of a mark broadcast together as Nx operands do. The result is
the mark's **shape**, and each element of the shape is one **row**.

```ocaml
let steps = Nx.linspace Nx.float64 0. 1. 200          (* [200] *)
let rates = Nx.create Nx.float64 [| 5; 1 |] [| 2.; 3.; 4.; 5.; 6. |]  (* [5; 1] *)
let curves = Nx.exp (Nx.neg (Nx.mul rates steps))     (* [5; 200] *)

line ~x:(num steps) ~y:(num curves) ~stroke:(dim 0) ()
```

The mark's shape is `[5; 200]`. `line` draws one curve per index of every
axis but the last, so this is five curves of 200 points. `dim 0` sees five
indices and gives each curve a category.

A **series** of a line or area is one index along every axis but the last,
split further by the categories of its colour channels. A categorical `x`
does not split a series.

`strings` and `floats` have shape `[|n|]`; a `[5; 200]` tensor meeting them
broadcasts along the last axis. A rank-1 channel next to a rank-2 tensor is
one value per column, not one per row.

### Missing values

A value is missing when:

- it is NaN or infinite;
- its channel's `~valid` mask is false there;
- its scale cannot place it, such as a non-positive value on a log scale or
  a string that names no category.

A row with a missing value is dropped. A line breaks at the gap; an area is
split; a rect with a missing extent draws nothing. Nothing is imputed. Give
the warning sink to `save` to see data problems:

```ocaml
let warnings = ref [] in
save ~warn:(fun w -> warnings := w :: !warnings) "f.png" fig
```

## Roles

| Role | Range | Default scale | A constant is |
| --- | --- | --- | --- |
| `x`, `y`, `x2`, `y2` | `float` | `"x"`, `"y"` | a fraction of the panel |
| `fill`, `stroke` | `Color.t` | `"color"` | a colour |
| `opacity` | `float` | `"opacity"` | an opacity from 0 to 1 |
| `size` | `float` | `"size"` | an area in square points |
| `width` | `float` | `"width"` | a width in points |
| `symbol` | `Symbol.t` | `"symbol"` | a symbol |
| `dash` | `Dash.t` | `"dash"` | a dash pattern |
| `text` | `Text.t` | none | a text |
| `fx`, `fy` | `string` | `"fx"`, `"fy"` | the panel to draw in |

`size` takes quantities; `symbol`, `dash`, `fx` and `fy` take categories.
A mark with no colour channel paints with the theme's accent; `rule` and
`text` paint with its ink.

## Marks

Every mark takes `~fx` and `~fy`, and every mark but `image` takes
`~opacity`. A mark raises when it is made if its channels do not broadcast,
and names the channels at fault.

### `dot`

```ocaml
val dot :
  ?fill:('f, Color.t) channel -> ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel -> ?size:(float, float) channel ->
  ?symbol:(string, Symbol.t) channel ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  x:('x, float) channel -> y:('y, float) channel -> unit -> t
```

A symbol at each row's point, in row order. `~size` is the symbol's area in
square points. Without `~fill` or `~stroke` it is filled with the accent;
with `~stroke` alone it is outlined only. `~symbol` defaults to a circle,
or to the categories of a band scale; filled dots use `Symbol.filled`,
outlined ones `Symbol.stroked`.

```ocaml
dot ~x:(num px) ~y:(num py)
  ~fill:(cat ~title:"group" ~labels:[| "control"; "treated" |] group)
  ~size:(num ~title:"weight" w)
  ()
```

Tens of thousands of dots in a panel are drawn as one image of their rows.

### `line`

```ocaml
val line :
  ?x:('x, float) channel -> ?stroke:('s, Color.t) channel ->
  ?fill:('f, Color.t) channel -> ?width:('w, float) channel ->
  ?dash:(string, Dash.t) channel -> ?opacity:('o, float) channel ->
  ?curve:Curve.t ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  y:('y, float) channel -> unit -> t
```

A curve through the points of each series, in order along the last axis.
`~x` defaults to `index (-1)`.

- `~curve` is `Curve.linear` by default; `Curve.step_after`,
  `Curve.step_before`, `Curve.step_mid`, `Curve.natural`, `Curve.monotone_x`,
  `Curve.monotone_y`, `Curve.catmull_rom` and `Curve.basis` are the others.
  A curve breaks at a dropped row.
- Without `~fill`, each series is stroked at `~width` (the theme's line
  width), broken by `~dash` (`Dash.solid` by default).
- With `~fill`, each series is closed and filled under the even-odd rule,
  and outlined only if `~stroke` is given.
- A series takes colour, width, dash and opacity from its first surviving
  row, with a warning if one varies along the series.

To tell series apart by colour and pattern from one legend, give both
channels one named band scale:

```ocaml
let run = Scale.band ~name:"run" () in
let by_run () = dim ~scale:run ~title:"run" ~labels:[| "a"; "b"; "c" |] 0 in
line ~x:(num t) ~y:(num curves) ~stroke:(by_run ()) ~dash:(by_run ()) ()
```

### `area`

```ocaml
val area :
  ?x:('x, float) channel -> ?y2:('y, float) channel ->
  ?fill:('f, Color.t) channel -> ?opacity:('o, float) channel ->
  ?curve:Curve.t -> ?stack:Stack.offset ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  y:('y, float) channel -> unit -> t
```

The region between the curve through `y` and a baseline. Without `~y2`,
`y` is a length and the baseline is zero, clamped into the domain. With
`~y2`, it is a band between two curves.

```ocaml
area ~x:(num t) ~y:(num lo) ~y2:(num hi) ~opacity:(const 0.3) ()
```

`~stack` is `` `Zero ``, `` `Expand `` or `` `Center ``; it cannot be
combined with `~y2`. A streamgraph is a centred stack.

### `rect`

```ocaml
val rect :
  ?x:('x, float) channel -> ?x2:('x, float) channel ->
  ?y:('y, float) channel -> ?y2:('y, float) channel ->
  ?fill:('f, Color.t) channel -> ?stroke:('s, Color.t) channel ->
  ?opacity:('o, float) channel -> ?stack:Stack.offset ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  unit -> t
```

The rectangle each row spans. What it spans depends on its channels:

- `x` and `x2`: the hull of the two, an end on a band scale standing for
  its band.
- `x` alone on a band scale: the band. If `y` is quantitative, this is a
  **bar**, and `rect` implies padding on the band scale so bars stand apart.
  If `y` is also a band scale, it is a **heatmap cell**, with no padding.
- `x` alone on a quantitative scale: a **length** from zero to `x`, zero
  clamped into the domain. On a log scale a length starts at the domain's
  lower end.
- `x` alone on a temporal scale: an instant; use `x` and `x2` for a span.
- No `x`: the whole panel.

The same rules apply along y. `~stack` stacks the one of `x` and `y` that
is a length; exactly one must be a length.

```ocaml
(* Bars. *)
rect ~x:(strings ~title:"model" models) ~y:(num ~title:"accuracy" acc) ()

(* Heatmap. *)
rect ~x:(dim ~title:"column" 1) ~y:(dim ~title:"row" 0)
  ~fill:(num ~title:"z" z) ()

(* Histogram from bin edges and counts. *)
let h = Stats.histogram ~bins:30 values in
rect ~x:(num h.x) ~x2:(num h.x2) ~y:(num ~title:"count" h.count) ()

(* A shaded x span. *)
rect ~x:(floats [| 2. |]) ~x2:(floats [| 4. |])
  ~fill:(const (Color.gray 0.9)) ()

(* Stacked bars: three series over four quarters. *)
rect ~x:(dim ~title:"quarter" 1) ~y:(num sales)
  ~fill:(dim ~title:"region" 0) ~stack:`Zero ()
```

`x2` without `x`, or `y2` without `y`, raises.

### `rule`

```ocaml
val rule :
  ?x:('x, float) channel -> ?x2:('x, float) channel ->
  ?y:('y, float) channel -> ?y2:('y, float) channel ->
  ?stroke:('s, Color.t) channel -> ?width:('w, float) channel ->
  ?dash:(string, Dash.t) channel -> ?opacity:('o, float) channel ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  unit -> t
```

A straight segment per row:

- `x` without `x2`: vertical, over the y extent `rect` would give `y` and
  `y2`. `rule ~x ()` crosses the panel; `rule ~x ~y ()` is a stem from
  zero.
- `y` without `y2`: horizontal, over the x extent.
- `x`, `x2`, `y`, `y2`: the segment between the two points.
- `y` and `y2` at each `x`: an error bar.
- `y` and `y2` at each band `x`: an error bar at the band centre.

```ocaml
(* Reference line. *)
rule ~y:(floats [| 0. |]) ~dash:(const Dash.dashed) ()

(* Vertical error bars. *)
rule ~x ~y:(num lo) ~y2:(num hi) ()

(* Horizontal error bars. *)
rule ~y ~x:(num lo) ~x2:(num hi) ()
```

`~dash` breaks the rule like a line. `~width` is in points.

### `abline`

```ocaml
val abline :
  ?stroke:('s, Color.t) channel -> ?width:('w, float) channel ->
  ?dash:(string, Dash.t) channel -> ?opacity:('o, float) channel ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  slope:(float, float) channel -> intercept:(float, float) channel ->
  unit -> t
```

The line `y = slope * x + intercept` across each panel's x domain. Slope
and intercept are values: they read no scale and do not widen the domain.
One line per row if the channels hold several values. On a nonlinear scale
the line is drawn as a curve.

```ocaml
abline ~slope:(const 1.) ~intercept:(const 0.) ~dash:(const Dash.dashed) ()
```

### `text`

```ocaml
val text :
  ?fill:('f, Color.t) channel -> ?opacity:('o, float) channel ->
  ?dx:float -> ?dy:float ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  x:('x, float) channel -> y:('y, float) channel ->
  text:('t, Text.t) channel -> unit -> t
```

Each row's text, centred on its point and moved `dx` points right and `dy`
points up. `~text` takes `num`, `strings`, `cat ~labels` or
`const (Text.v "...")`. Numbers of one mark share the decimals the most
precise value needs, in its source dtype.

```ocaml
layer
  [
    rect ~x ~y:(num scores) ();
    text ~x ~y:(num scores) ~text:(num scores) ~dy:6. ();
  ]
```

Rich text, with styles, colours and super- or subscripts, is a `Text.t`:

```ocaml
Text.(concat [ v "R"; sup (v "2") ])
```

### `image`

```ocaml
val image :
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  ('a, 'b, Nx.host) Nx.t -> t
```

A rank-2 tensor is one grey image of shape `[|h; w|]`. Otherwise the
trailing axes are `[|h; w; c|]`, with `c` equal to 1 (grey), 3 (RGB) or 4
(RGBA), and the leading axes hold several images, one per datum. `fx` and
`fy` put them in panels.

Values range over 0 to 255 for `uint8` and 0 to 1 for floats, clamped. A
pixel with a NaN component paints nothing. Each image covers `[0, w]` on x
and `[0, h]` on y, row 0 at the top. `image` implies no axes, `nice:false`
and `reverse` on y, and `Coord.cartesian ~aspect:1. ()`. Marks layered over
it draw in pixel coordinates.

```ocaml
image ~fx:(dim ~scale:(Scale.band ~wrap:3 ()) 0) batch
```

A rank above 2 with a last axis other than 1, 3 or 4 raises.

### `contour`

```ocaml
val contour :
  ?x:('x, float) channel -> ?y:('y, float) channel ->
  ?opacity:('o, float) channel ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  fill:(float, Color.t) channel -> unit -> t
```

Filled contours of fields sampled on grids. The last two axes of the mark's
shape are the grid's rows and columns. `x` gives each column's position and
`y` each row's, so for an `[n; m]` field, `x` has shape `[m]` and `y` has
shape `[n; 1]`. `contour` implies `nice:false` so the field fills the panel.

The levels are 0 and 1, the normalised ends of the fill scale's domain,
plus the ticks the layout froze for it. `~scale:(Scale.linear ~ticks:levels
())` fixes them; `contour` implies `stepped` on the fill scale. Missing
samples cut holes.

```ocaml
let xs = Nx.linspace Nx.float64 (-2.) 2. 60 in
let ys = Nx.reshape [| 50; 1 |] (Nx.linspace Nx.float64 (-1.5) 1.5 50) in
let field = Nx.add (Nx.square xs) (Nx.mul_s (Nx.square ys) 2.) in
contour ~x:(num ~title:"x" xs) ~y:(num ~title:"y" ys)
  ~fill:(num ~title:"f" field) ()
```

`fill` cannot be a constant. `x` must not vary along rows and `y` must not
vary along columns; a field with non-monotone coordinates draws nothing,
with a warning.

### `frame`

```ocaml
val frame :
  ?stroke:('s, Color.t) channel -> ?opacity:('o, float) channel ->
  ?fx:(string, string) channel -> ?fy:(string, string) channel ->
  unit -> t
```

Outlines each panel's data area with the ink, under the axes:
`layer [ f; frame () ]` boxes the panels.

## Stacks

`~stack` lays the lengths of one position end to end. It applies to `rect`
and `area`.

| Offset | Meaning |
| --- | --- |
| `` `Zero `` | piles start at zero, positive lengths upward, negative downward |
| `` `Expand `` | each column is scaled to fill 0 to 1 |
| `` `Center `` | each column is centred on zero, for streamgraphs |

Rows are laid in row order. A missing length stacks nothing. A start the
scale cannot place, such as zero on a log scale, is clamped into the domain.

## Facets

`~fx` and `~fy` split one mark into panels: each row goes in the column of
its `fx` category and the row of its `fy` category. Facet scales are band
scales; `Scale.band ~wrap:n ()` wraps the columns into rows of `n`.

```ocaml
line ~x:(num t) ~y:(num signals)
  ~fx:(dim ~title:"damping" ~labels:[| "0.1"; "0.2"; "0.4"; "0.8" |] 0)
  ()
```

- Facet panels share every scale.
- A mark without facet channels, layered with a faceted one, is drawn in
  every panel; a reference line appears in each.
- A facet constant draws its mark in that panel only, with a warning if no
  such panel exists.

## Reducers

`draw` reduces marks that have more rows than the output has pixels, on the
device where their data lives, and reads the reduced values. The built-in
marks use `Mark.m4` for dense lines, `Mark.cells` for rects and
`Mark.raster` for dots. A custom mark chooses a reducer with
`Mark.v ~reduce`; `Mark.m4`, `Mark.cells` and `Mark.raster` are available
for reuse.

## Porting the first Hugin

| First hugin | Current Hugin |
| --- | --- |
| `line ~x ~y ()` | `line ~x:(num x) ~y:(num y) ()` |
| `point ~x ~y ()` | `dot ~x:(num x) ~y:(num y) ()` |
| `~label:"sin"` | a categorical colour channel, e.g. `~stroke:(strings [| "sin" |])` |
| `line ~color c` | `~stroke:(const c)` |
| `bar ~x ~height ()` | `rect ~x:(strings names) ~y:(num height) ()` |
| `hist ~x ~bins` | `Stats.histogram ~bins n x`, then `rect` from `h.x` to `h.x2` |
| `heatmap ~data ()` | `rect ~x:(dim 1) ~y:(dim 0) ~fill:(num data) ()` |
| `imshow ~stretch` | the fill scale's constructor: `Scale.log`, `Scale.pow`, ... |
| `fill_between ~y1 ~y2` | `area ~y:(num y1) ~y2:(num y2) ()` |
| `hline ~y`, `vline ~x` | `rule ~y:(floats [| y |]) ()`, `rule ~x:(floats [| x |]) ()` |
| `errorbar ~yerr` | `layer [ rule ~x ~y:lo ~y2:hi (); dot ~x ~y () ]` |
| `xlabel "x"` | `~title:"x"` on the x channel, or `axis ~title:"x" "x"` |
| `xlim lo hi` | `~scale:(Scale.linear ~domain:(lo, hi) ())` |
| ``xscale `Log`` | `~scale:(Scale.log ())` |
| `xticks vs` | `~scale:(Scale.linear ~ticks:vs ())` |
| `grid_lines true` | `layer [ f; axis ~grid:true "x"; axis ~grid:true "y" ]` |
| `legend ~loc:Upper_right` | ``layer [ f; legend ~side:(`Inside `Top_right) "color" ]`` |
| `render_png file` | `save file` |
| `with_theme th` | `save ~theme:th` |
