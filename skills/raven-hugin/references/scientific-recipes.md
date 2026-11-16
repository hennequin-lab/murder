# Scientific recipes

Working patterns for the figures the lab draws. The code uses the public
interface of this pin; the shape of each recipe matters more than its toy
data.

## Before drawing

- Put the data on the host. Every channel takes `Nx.host`. A result that
  came back from a device moves with `Nx.place Nx.Host.on`; do it outside
  the jitted function, at the reporting boundary.
- Build the figure from data that is already reduced or sampled for the
  page. A line with 100,000 points is reduced by `Mark.m4`; a scatter with
  tens of thousands of dots is drawn as one image. You do not need to bin
  by hand for rendering, but you do for a statistic.
- Keep the figure in the same program as the computation when it must be
  reproducible. A figure is a pure value; `save` writes the same bytes on
  every run.
- Give every channel a title with units. The title is the axis or legend
  label; there is no separate label object.
- Collect warnings with `~warn` when a figure is part of a test or an
  example. A warning is a data problem, and the figure still draws.

## Training curves

Loss spans orders of magnitude, so plot it on a log scale and show the raw
run under a smoothed line. A vertical rule at each epoch boundary gives the
axis a second reading:

```ocaml
open Hugin

let steps = Nx.linspace Nx.float64 0. 2_999. 3_000

let loss =
  let noise = Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| 3_000 |] in
  Nx.mul
    (Nx.add_s (Nx.exp (Nx.mul_s steps (-0.0015))) 0.05)
    (Nx.exp (Nx.mul_s noise 0.25))

let () =
  let epochs = Nx.linspace Nx.float64 0. 3_000. 7 in
  layer
    [
      rule ~x:(num epochs) ~opacity:(const 0.15) ();
      line ~x:(num ~title:"step" steps)
        ~y:(num ~scale:(Scale.log ()) ~title:"loss" loss)
        ~opacity:(const 0.3) ();
      line ~x:(num steps) ~y:(num (Nx.ewma ~alpha:0.02 loss)) ();
    ]
  |> title "Training loss"
  |> save ~size:(Size.figure 420. 260.) "loss.png"
```

Loss on a log scale drops non-positive values: a diverged run leaves a gap
instead of a misleading spike. Clip or filter before plotting if the zero
is an artefact.

A training dashboard stacks two panels that share x, so the reader compares
loss and accuracy at the same step:

```ocaml
grid [ [ losses ]; [ accuracy ] ]
|> share [ ("x", `Shared) ]
|> save ~size:(Size.figure 420. 320.) "dashboard.png"
```

Plot raw and smoothed runs of several seeds as `[k; n]` tensors. One mark
draws all of them; a `dim 0` colour channel gives each seed a colour and a
legend. If the seeds are replicates of one condition, draw their band and
the mean instead of one line per seed:

```ocaml
let lo = Nx.min ~axes:[ 0 ] runs and hi = Nx.max ~axes:[ 0 ] runs in
layer
  [
    area ~x:(num t) ~y:(num lo) ~y2:(num hi) ~opacity:(const 0.3) ();
    line ~x:(num t) ~y:(num (Nx.mean ~axes:[ 0 ] runs)) ();
  ]
```

## Uncertainty

Error bars are a composition: a rule from the lower to the upper end under
the point estimate. For horizontal bars, swap the roles:

```ocaml
let methods = [| "baseline"; "dropout"; "mixup"; "ours" |] in

let () =
  let mean = Nx.create Nx.float64 [| 4 |] [| 0.71; 0.74; 0.76; 0.80 |] in
  let err = Nx.create Nx.float64 [| 4 |] [| 0.02; 0.03; 0.025; 0.015 |] in
  let x = strings ~title:"method" methods in
  layer
    [
      rule ~x ~y:(num (Nx.sub mean err)) ~y2:(num (Nx.add mean err)) ();
      dot ~x ~y:(num ~title:"accuracy" mean) ();
    ]
  |> save "errorbars.png"
```

A band is better than bars when the uncertainty is continuous over x:
`area ~y:lo ~y2:hi` for the interval, `line ~y:mean` on top. State in the
caption what the band is (s.d., s.e.m., a percentile interval); the figure
cannot.

## Distributions

`Stats.histogram` gives bin edges and counts. Draw each bin from edge to
edge. For several distributions over shared bins, stack the tensors along a
new leading axis and let `dim 0` colour them:

```ocaml
let h = Stats.histogram ~bins:40 values in
rect ~x:(num ~title:"weight" h.x) ~x2:(num h.x2)
  ~y:(num ~title:"density" h.density)
  ~fill:(dim ~labels:[| "initialisation"; "trained" |] 0)
  ~opacity:(const 0.6) ()
```

Use `h.count` for counts and `h.density` for densities; the density times
the bin widths sums to one, so histograms of different sample sizes compare.
Overlapping bars are easier to read with `~opacity:(const 0.5)` and an
outline, or as two facets. Do not put two y axes in one panel; use two
panels or densities.

## Matrices

A matrix is a `rect` whose rows and columns read `dim`, with the values in
`fill`. Fix the colour domain when comparing matrices, and use a diverging
scheme only when zero is the midpoint:

```ocaml
let z = Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| 12; 16 |] in
rect ~x:(dim ~title:"column" 1) ~y:(dim ~title:"row" 0)
  ~fill:
    (num ~title:"z"
       ~scale:(Scale.linear ~domain:(-3., 3.) ~scheme:Scheme.rdbu ())
       z)
  ()
|> coord (Coord.cartesian ~aspect:1. ())
```

For a confusion matrix, colour by recall and write the count in the colour
that reads on the cell. `map_range Color.contrast` picks black or white per
cell:

```ocaml
let classes = [| "plane"; "car"; "bird"; "cat" |] in
let f = Nx.cast Nx.float64 counts in
let recall =
  num ~title:"recall" Nx.(div f (sum ~axes:[ 1 ] ~keepdims:true f))
in
let predicted = dim ~title:"predicted" ~labels:classes 1
and truth = dim ~title:"true" ~labels:classes 0 in
layer
  [
    rect ~x:predicted ~y:truth ~fill:recall ();
    text ~x:predicted ~y:truth ~text:(num counts)
      ~fill:(map_range Color.contrast recall) ();
  ]
|> coord (Coord.cartesian ~aspect:1. ())
```

`~labels` may repeat in `dim`, which is what a matrix with repeated class
names needs. Every index joins the domain even if no row uses it, so rows
and columns stay aligned when the matrix is filtered.

## Fields

A field is a rank-2 tensor with x per column and y per row. `contour` fills
the bands between levels; the levels are the fill scale's ticks. Use a log
fill scale for a positive field spanning orders of magnitude, and fix the
ticks when the levels must be comparable between figures:

```ocaml
let alphas = Nx.linspace Nx.float64 (-2.) 2. 81 in
let betas = Nx.reshape [| 61; 1 |] (Nx.linspace Nx.float64 (-1.5) 1.5 61) in
let loss =
  Nx.add_s
    (Nx.add
       (Nx.square (Nx.sub_s alphas 1.))
       (Nx.mul_s (Nx.square (Nx.sub betas (Nx.square alphas))) 10.))
    0.05
in
contour ~x:(num ~title:"alpha" alphas) ~y:(num ~title:"beta" betas)
  ~fill:(num ~title:"loss" ~scale:(Scale.log ()) loss)
  ()
```

Layer an optimiser's path over the field. The path is a `[steps; 2]` tensor;
`Nx.(slice [ A; I 0 ])` takes x and `Nx.(slice [ A; I 1 ])` y:

```ocaml
let px = num Nx.(slice [ A; I 0 ] path)
and py = num Nx.(slice [ A; I 1 ] path) in
layer
  [
    contour ~x:(num ~title:"alpha" alphas) ~y:(num ~title:"beta" betas)
      ~fill:(num ~title:"loss" ~scale:(Scale.log ()) loss)
      ();
    line ~x:px ~y:py ~stroke:(const Color.white) ();
    dot ~x:px ~y:py ~fill:(const Color.white) ();
  ]
```

`contour` implies `nice:false`, so the field fills the panel. Missing
samples cut holes; they are not interpolated.

## Calibration and square axes

A calibration plot compares predicted probabilities with observed
frequencies. Fix both scales to `[0, 1]`, write them as percentages, and
make the axes the same length so the diagonal is a true 45 degrees:

```ocaml
let unit = Scale.linear ~domain:(0., 1.) ~notation:Number.Percent in
let x = num ~scale:(unit ~name:"x" ()) ~title:"predicted" p in
let y = num ~scale:(unit ~name:"y" ()) ~title:"observed" observed in
layer
  [
    abline ~slope:(const 1.) ~intercept:(const 0.)
      ~dash:(const Dash.dashed) ();
    line ~x ~y ~stroke:(dim ~title:"model" ~labels:[| "A"; "B" |] 0) ();
    dot ~x ~y ~fill:(dim ~title:"model" ~labels:[| "A"; "B" |] 0) ();
  ]
|> coord (Coord.cartesian ~aspect:1. ())
```

`abline` does not widen either domain and is cut at the panel edges, so it
stays diagonal under a fixed domain. The line and the dots both read the
default `"color"` scale, so the figure has one legend for the models. To
add a dash channel to the same categories, give every channel one named
band scale.

## Images

A batch of images is a tensor `[n; h; w; c]`. The leading axis indexes the
images; `fx` puts them in panels. Wrapped panels with captions are an image
grid:

```ocaml
let captions = Array.map (fun (t, p) -> t ^ " -> " ^ p) pairs in
layer
  [
    image
      ~fx:(dim ~scale:(Scale.band ~wrap:4 ()) ~labels:captions 0)
      batch;
    rect
      ~fx:(dim ~valid:(Nx.not_equal truth pred) 0)
      ~stroke:(const Color.red) ();
  ]
```

`image` covers `[0, w]` by `[0, h]`, row 0 at the top, and implies no axes
and a reversed y. Pixels are 0 to 255 for `uint8` and 0 to 1 for floats.
`rect` layered over it draws in pixel coordinates, which is what the frame
of a mistaken image uses.

## A paper figure

Set the theme to 8 points, size the page in millimetres, label the panels
with `title`, and write PDF for the compositor. Two columns are about
180 mm wide; a single column is about 85 mm.

```ocaml
let a = line ~x:(num ~title:"time (s)" t) ~y:(num ~title:"amplitude" waves) () in
let b = dot ~x:(num ~title:"step" steps) ~y:(num ~title:"loss" loss) () in

let fig = grid [ [ title "(a)" a; title "(b)" b ] ] in
let theme = Theme.v ~size:8. ~palette:Scheme.okabe_ito () in
let size = Size.figure (Size.mm 85.) (Size.mm 55.) in
save ~theme ~size "figure.pdf" fig;
save ~theme ~size ~density:(Size.dpi 300.) "figure.png" fig
```

For a double-column figure, use `Size.figure (Size.mm 180.) ...` and a grid
with two columns. Keep the panel labels outside the data area, in the
figure's own titles, so the compositor can place the figure at any scale.

If the page is too small for its text and guides, `layout` raises and names
a size that fits. Do not silently shrink the theme: 8 points is already the
lower bound for a printed column.

## Style choices

- Encode the comparison in position when the values are meant to be read
  precisely. Colour and size are for grouping and magnitude, not for a
  precise ratio.
- One y axis per panel. For two quantities of different scale, use two
  panels sharing x; `share [ ("x", `Shared) ]` labels the axis once.
- Bars and areas start at zero. Dots and lines need not, unless the zero is
  part of the claim. If two panels compare the same quantity, use the same
  domain on both.
- Log scales for positive quantities spanning orders of magnitude: loss,
  learning rate, time to a threshold. Say so in the caption. `symlog` for a
  quantity that crosses zero over orders of magnitude.
- Sequential schemes for ordered magnitudes, diverging only with a
  meaningful midpoint, qualitative palettes for unordered categories. One
  palette per figure. Facet before adding a tenth colour.
- Small multiples share their scales. `grid` cells keep x and y unless
  `share` says otherwise; facets share every scale. Use a grid when panels
  need different data, a facet when one mark splits by category.
- `Coord.cartesian ~aspect:1.` when shape matters: trajectories, phase
  portraits, calibration, correlation, square heatmap cells. Time series
  are wider than tall.
- Fewer, larger marks. A legend inside a panel only in a corner the data
  leaves empty; otherwise put it below or beside the panels.
- Units in the axis title, a sentence-case title without a period, panel
  labels (a), (b), (c). Use `Text.sup` and `Text.sub` for powers and indices.
- Vector output for papers, 300 dpi PNG for a raster, and a larger theme for
  slides. `Theme.talk` needs a larger page as well.
- Look at the output after every change. `save` writes any warnings; a
  dropped row, a missing category or an axis label clipped by the page is a
  bug in the figure or in the data.

## Hugin or gnuplot + TikZ

Use Hugin when the figure is produced from Nx data in OCaml, must rebuild
from the same code, or benefits from a scale derived from the data. Use the
lab's gnuplot + TikZ workflow (the `figures` skill) when the figure needs
freeform TikZ annotation over data files, or when the surrounding document
already owns the typography. Both write PDF for print; do not mix two
plotting styles in one manuscript.

## Testing a figure

A figure is a value, so a test can check its structure without rendering
pixels: resolve it, inspect `Resolved.scale` domains, and collect warnings.

```ocaml
let warnings = ref [] in
save ~warn:(fun w -> warnings := w :: !warnings) path fig;
(match !warnings with
 | [] -> ()
 | w :: _ -> failwith (Format.asprintf "%a" pp_warning w))
```

Rendering is deterministic, so a golden is possible: write to a temporary
file and compare its bytes, or compare the renderable. Regenerate a golden
with the pinned Hugin when the figure changes; do not edit it by hand. Keep
the figure's data small and synthetic, and put the file in a temporary
directory, not in the source tree.

## Plotting device data

A jitted step returns values on the device. Plot from a host copy, outside
the step:

```ocaml
let rec train ~i (params, state, history) =
  if i = 0 then (params, state, history)
  else
    let params, state, loss = step key (params, state) x y in
    let history = if i mod 100 = 0 then Nx.item [] loss :: history else history in
    train ~i:(i - 1) (params, state, history)

let () =
  let _, _, history = train ~i:steps (params, state, []) in
  line ~y:(num (Nx.create Nx.float64 [| List.length history |]
                  (Array.of_list (List.rev history)))) ()
  |> save "loss.png"
```

`Nx.item` forces a wait on the device. Read at the reporting interval, not
every step. If the history is kept as a device tensor, move it once with
`Nx.place Nx.Host.on` before building the figure; a set minted by
`Nx.devices [ Rig.host ]` is not the `Nx.host` brand that channels require.
