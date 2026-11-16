# Scales, colour and guides

A scale maps data values onto `[0, 1]`, and a role maps that interval onto a
position, a colour, an area or another mark property. Scales are fitted
once per figure, by name, and every mark that reads a name shares its
scale.

## Kinds

```ocaml
type _ kind =
  | Quantitative : float kind
  | Temporal : Time.t kind
  | Categorical : string kind
```

A scale's kind is fixed by its domain type. A user name read by `x` in one
panel and by `fill` in another has one scale and two guides: an axis and a
legend, with one set of ticks.

## Constructors

```ocaml
Scale.linear ()                     (* in proportion *)
Scale.log ~base:10. ()              (* positive values, by logarithm *)
Scale.symlog ~constant:1. ()        (* either sign, log away from zero *)
Scale.pow ~exponent:0.5 ()          (* by a power *)
Scale.custom ~transform:"asinh" ~forward:Float.asinh ~inverse:Float.sinh ()
Scale.time ~tz_offset_s:0 ()        (* instants, calendar ticks *)
Scale.band ()                       (* one band per category *)
```

| Constructor | Default domain | Notes |
| --- | --- | --- |
| `linear` | `[0, 1]` | |
| `log` | `[1, base]` | positive domain; `base` above 1, default 10 |
| `symlog` | `[0, 1]` | linear near zero, logarithmic away |
| `pow` | `[0, 1]` | `exponent` positive |
| `custom` | `[0, 1]` | `forward` and `inverse` strictly monotone |
| `time` | epoch to one day later | ticks fall on calendar boundaries |
| `band` | no categories | one band per category |

A `custom` transform is an OCaml function. It must not read mutable state,
and its `forward` must be strictly monotone where it is finite.

## Properties

Every constructor takes `~name`, `~domain`, `~reverse`, `~ticks`, `~scheme`
and `~unknown`. Quantitative constructors also take `~nice`, `~clamp`,
`~stepped`, `~notation` and `~areas`; `linear`, `symlog`, `pow` and
`custom` add `~zero`; `band` takes `~padding`, `~wrap`, `~symbols` and
`~dashes` instead; `time` takes `~tz_offset_s`.

| Property | Meaning | Default |
| --- | --- | --- |
| `name` | the scale's name | the role's name |
| `domain` | the fitted domain or explicit limits | data extent |
| `nice` | round a fitted domain outward | `true` |
| `zero` | include zero in a fitted domain | `false` |
| `clamp` | clamp values outside the domain | `false` |
| `reverse` | run the scale from 1 to 0 | `false` |
| `stepped` | constant range between ticks | `false` |
| `ticks` | fix the guide values | chosen |
| `notation` | tick label notation | by scale |
| `scheme` | colours for colour roles | theme |
| `areas` | size role at normalised 0 and 1 | theme |
| `unknown` | colour for missing values | no paint |
| `padding` | fraction of each band left empty | `0.` |
| `wrap` | panels per row for a facet | no wrap |
| `symbols` | symbols per category | theme |
| `dashes` | dash patterns per category | `Dash.all` |
| `tz_offset_s` | calendar offset for time ticks | `0` |

`~domain` is final: fitting neither widens nor rounds it. A continuous
domain must have finite ends with `lo <= hi`; a log domain must be positive.

`~nice:false` keeps a fitted domain at the data's extent. Use it when the
exact extent is the point, as for a field that should fill its panel; use
the default when round tick values help read the figure.

`~zero:true` widens the fitted domain to include zero. Bars and lengths
imply it; a point or line does not, unless the comparison is with zero.

`~clamp:true` keeps a value outside the domain at the nearest edge instead
of dropping it. `~reverse:true` runs the scale backwards. A band scale read
by `y` implies `reverse`, so its first category is at the top.

`~ticks` fixes the axis ticks, legend entries and colour bar values. On a
band scale it limits the axis labels and legend entries to the categories
named, while facet headers still name every panel.

`~notation` writes quantitative labels as `Number.Plain`, `Number.Exponent`,
`Number.Si` or `Number.Percent`.

`~scheme` colours a continuous colour role, and sets the palette of a band
scale. `~areas` sets the areas in square points that the size role maps 0
and 1 to.

### Scales as values

```ocaml
val kind : 'd t -> 'd kind
val domain : 'd t -> 'd domain          (* Floats, Instants or Categories *)
val name : 'd t -> string option
val normalize : 'd t -> 'd -> float      (* [0,1] or nan when missing *)
val invert : 'd t -> float -> 'd option
val fit : 'd domain option -> 'd t -> 'd t
val with_domain : 'd domain -> 'd t -> 'd t
val merge : 'd t -> 'd t -> ('d t, property) result
val imply : 'd t -> 'd t -> 'd t
val missing : float t -> ('a, 'b, 'd) Nx.t -> 'd Nx.bool_t
```

A domain is `Floats (lo, hi)`, `Instants (lo, hi)` or `Categories c`, with
categories `Labels of string array` or `Indices of (int * string) array`.

## Names and scopes

A channel reads the scale its specification names, or the default scale of
its role:

| Role | Scale |
| --- | --- |
| `x`, `x2` | `"x"` |
| `y`, `y2` | `"y"` |
| `fill`, `stroke` | `"color"` |
| `opacity` | `"opacity"` |
| `size` | `"size"` |
| `width` | `"width"` |
| `symbol` | `"symbol"` |
| `dash` | `"dash"` |
| `fx`, `fy` | `"fx"`, `"fy"` |

A specification without a name configures the default scale of the role it
is given to:

```ocaml
(* Every reader of "y" in this scope is logarithmic. *)
line ~x:(num t) ~y:(num ~scale:(Scale.log ()) ~title:"loss" loss) ()
```

A name of your own denotes one scale wherever it appears in the figure:

```ocaml
let rate = Scale.linear ~name:"rate" () in
line ~y:(num curves) ~stroke:(num ~scale:rate rates) ()
```

`x`, `y`, `fx` and `fy` are scoped by the innermost grid cell; every other
name by the whole figure. So:

- the children of a `layer` share every scale;
- the cells of a grid keep their own x and y scales and share every other
  scale, which gives a figure one colour legend however it is arranged;
- `share [ ("x", `Shared) ]` gives grid cells one x scale, and
  `share [ ("color", `Independent) ]` gives each cell its own colour scale.

### Merging

A property is **explicit** when a channel sets it, **implied** when a mark
implies it, and **defaulted** otherwise. Explicit beats implied beats
defaulted.

Marks imply properties that their geometry needs:

- `rect` and `rule` imply `zero` on a length.
- `rect` implies `padding` on the band scale of a bar.
- A band `y` implies `reverse`.
- An `image` implies `nice:false`, no axes and `reverse` on y.
- A `contour` implies `nice:false` on its x and y and `stepped` on its fill.

`resolve` raises when two channels of one scale set a property to different
values at the same level, when two marks in a panel read different x or y
scales, or when a user name is read by two kinds. Name the scale, then put
the same specification on every channel that reads it.

### Fitting

`resolve` fits categorical scales first, then continuous scales:

- A continuous domain is the hull of the values of every channel that
  reads the scale, over the rows that are not dropped. It is widened by
  `zero` and rounded outward by `nice`.
- A labelled categorical domain is the union of every label of each
  `cat ~labels` channel, used or not, and the distinct strings of each
  `strings` channel, in the order the figure is written.
- An indexed categorical domain is the increasing union of the indices of
  `dim` channels and the codes of `cat` without labels.

An explicit domain replaces the fitted one; a view's zoom replaces both.

## Colour

`Color.t` is an sRGB colour with straight alpha.

```ocaml
let teal = Color.v 0.1 0.5 0.55
let faded = Color.with_alpha 0.4 teal
let dark = Color.gray 0.15
let ink = Result.get_ok (Color.of_hex "#e4572e")
let blend = Color.mix 0.5 teal Color.white
let on_teal = Color.contrast teal
```

`Color.mix` interpolates in Oklab, so equal steps look equally different.
`Color.contrast c` is black or white, whichever reads better on `c`.
`Color.red`, `green`, `blue`, `black`, `white` and `transparent` are
constants.

A `Scheme.t` holds a ramp or a palette:

```ocaml
val ramp : Color.t array -> t          (* continuous gradient *)
val palette : Color.t array -> t       (* one colour per category *)
val reverse : t -> t
val color : ?unknown:Color.t -> t -> float -> Color.t
val table : t -> Color.t array
val colors : int -> t -> Color.t array
```

A continuous colour role reads the scheme's gradient; a band scale gives
each category the next colour of the palette. Without a scheme, a continuous
scale uses the theme's scheme (viridis) and a band scale its palette
(tableau10).

| Use | Schemes |
| --- | --- |
| sequential | `viridis`, `magma`, `inferno`, `plasma`, `cividis`, `turbo`, `blues`, `greens`, `greys`, `oranges`, `purples`, `reds`, `bugn`, `bupu`, `gnbu`, `orrd`, `pubu`, `pubugn`, `purd`, `rdpu`, `ylgn`, `ylgnbu`, `ylorbr`, `ylorrd` |
| diverging | `rdbu`, `brbg`, `piyg`, `prgn`, `puor`, `rdgy`, `rdylbu`, `rdylgn`, `spectral` |
| qualitative | `okabe_ito`, `tableau10`, `accent`, `dark2`, `paired`, `pastel1`, `pastel2`, `set1`, `set2`, `set3` |

Use a diverging scheme only when the midpoint means something, normally
zero. Use a qualitative palette for unordered categories, and facet before
the palette runs out of distinguishable colours.

`map_range` applies a function to the role's value after the scale. It is
the way to write text over a coloured cell in a readable colour:

```ocaml
rect ~x:(dim 1) ~y:(dim 0) ~fill:share ();
text ~x:(dim 1) ~y:(dim 0) ~text:(num counts)
  ~fill:(map_range Color.contrast share) ()
```

## Guides

Every position scale gets an axis on its panel. Every other scale gets a
legend: a colour bar for a continuous colour scale, one entry per category
or tick otherwise. A guide is titled by the distinct titles of the channels
that read its scale, joined with commas; set `~title` on one channel.

A categorical scale yields no legend when each channel that reads it has
the data of a position channel whose axis is shown in the same panels. Bars
coloured by their own x category therefore carry no redundant legend.

`axis` and `legend` replace a generated guide. They draw nothing
themselves and configure the guide of the named scale in the scope of the
figures they are layered with:

```ocaml
val axis :
  ?side:[ `Left | `Right | `Top | `Bottom ] ->
  ?grid:bool -> ?show:bool -> ?title:string -> string -> t

val axis' :
  ?side:[ `Left | `Right | `Top | `Bottom ] ->
  ?grid:bool -> ?show:bool -> ?title:Text.t -> string -> t

type corner = [ `Top_left | `Top_right | `Bottom_left | `Bottom_right ]
type inside = [ corner | `Left | `Right | `Top | `Bottom | `Center ]

val legend :
  ?side:[< `Left | `Right | `Top | `Bottom | `Inside of inside ] ->
  ?ncol:int -> ?show:bool -> ?title:string -> string -> t
```

- `axis ~grid:true "y"` draws grid lines across the panel at the ticks.
- `axis ~show:false "x"` hides the axis. Layering both hides both axes.
- ``axis ~side:`Top "x"`` moves an x axis above the panel.
- ``legend ~side:`Bottom "color"`` moves a legend below the panels and wraps
  its entries into rows.
- `legend ~ncol:3 "color"` fixes three entries per row.
- ``legend ~side:(`Inside `Top_right) "color"`` puts a legend in a panel
  corner. The legend is drawn under the marks and no place is chosen for
  being empty, so pick a corner the data leaves empty.
- `axis'` takes a `Text.t` for styles, colours and super- or subscripts:
  `axis' ~title:Text.(concat [ v "x"; sup (v "2") ]) "x"`.

`axis` and `legend` raise at resolve if the name has no scale of that kind,
if a panel holds two different axes or legends for one scale, or if the
side does not match the role.

## Ticks, notation and time

`layout` chooses round tick values whose labels do not overlap, at the
theme's lengths. On a log scale it uses the decimal steps of the base; on a
time scale it uses calendar boundaries. `~ticks` replaces the choice, and
`~notation` the labels.

```ocaml
let lr = Scale.log ~ticks:[| 1e-4; 1e-3; 1e-2 |] ~notation:Number.Exponent () in
let acc = Scale.linear ~notation:Number.Percent () in
line
  ~x:(num ~scale:lr ~title:"learning rate" lr_values)
  ~y:(num ~scale:acc ~title:"accuracy" acc_values)
  ()
```

A tick label names its value. A label that is not the value, such as a
category name, belongs to a band scale.

Temporal data uses `time`:

```ocaml
val time :
  ?scale:Time.t Scale.t -> ?valid:Nx.host Nx.bool_t ->
  ?title:string -> Time.resolution ->
  ('a, 'b, Nx.host) Nx.t -> (Time.t, 'r) channel
```

The tensor holds integer counts since the Unix epoch in `Time.S`, `Time.Ms`,
`Time.Us` or `Time.Ns`. The axis ticks fall on calendar boundaries and
labels name the calendar, with a second line where the coarser unit changes.
`~tz_offset_s` sets the calendar's offset from UTC.

```ocaml
let start = 1_767_225_600L (* 2026-01-01 UTC *) in
let days = Nx.add_s (Nx.mul_s (Nx.arange Nx.int64 0 90 1) 86_400L) start in
line ~x:(time Time.S days) ~y:(num ~title:"level" level) ()
```

`Time.of_date (2026, 1, 1)` gives an instant at UTC;
`Time.of_date ~tz_offset_s:3600 (2026, 1, 1)` gives it at UTC+01:00.
`Time.to_int64 Time.S` converts an instant back to seconds.

## Histograms

```ocaml
type 'd bins = {
  x : 'd Nx.float64_t;       (* left edges, [|n|] *)
  x2 : 'd Nx.float64_t;      (* right edges, [|n|] *)
  count : 'd Nx.float64_t;   (* counts, last axis n *)
  density : 'd Nx.float64_t; (* count / total / bin width *)
}

val histogram : ?bins:int -> ('a, 'b, 'd) Nx.t -> 'd bins
```

`Stats.histogram` computes on the data's device set with Nx operations and
reads nothing back. The bins span all finite values and have equal width;
the last bin includes its right edge. NaN and infinities are in no bin.
`bins` defaults to Sturges' rule. For an `[k; n]` tensor it returns one
histogram per row, over shared bins.

```ocaml
let h = Stats.histogram ~bins:40 values in
rect ~x:(num ~title:"weight" h.x) ~x2:(num h.x2)
  ~y:(num ~title:"density" h.density)
  ~fill:(dim ~labels:[| "init"; "trained" |] 0)
  ~opacity:(const 0.6) ()
```

The result lies on the data's device set, so a host figure needs the data
on the host. Move device data back before the histogram, or place the
result with `Nx.place Nx.Host.on`.
