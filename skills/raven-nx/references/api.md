# Nx API catalogue

Signatures are abbreviated: `t` is `('v,'s,'d) Nx.t`, `host t` is
`('v,'s,Nx.host) Nx.t`, `p` is a placement, `dt` a dtype. Read `nx.mli` for
the contract, error cases and numerical bounds.

## Types and dtypes

```ocaml
type ('v, 's, 'd) t
type ('v, 's) dtype = ('v, 's) Dtype.t
type 'd float32_t = (float, Dtype.float32_elt, 'd) t
type 'd float64_t | 'd float16_t | 'd bfloat16_t
type 'd int64_t | 'd int32_t | 'd int16_t | 'd int8_t
type 'd uint64_t | 'd uint32_t | 'd uint16_t | 'd uint8_t
type 'd bool_t | 'd bit_t
type 'd complex128_t | 'd complex64_t
type 'd packed = P : ('v,'s,'d) t -> 'd packed
```

Dtype values: `float64 float32 float16 bfloat16 int64 int32 int16 int8
uint64 uint32 uint16 uint8 bool bit complex128 complex64 float8_e4m3fn
float8_e5m2 float4_e2m1fn int4 uint4`.

```ocaml
Nx.Dtype.name : ('v,'s) dtype -> string
Nx.Dtype.is : 'k kind -> ('v,'s) dtype -> bool       (* Nx.Dtype.Float, Int, ... *)
Nx.Dtype.of_float : ('v,'s) dtype -> float -> 'v
Nx.unpack : ('v,'s) dtype -> 'd packed -> ('v,'s,'d) t
```

## Metadata

```ocaml
val shape : t -> int array
val dtype : t -> ('v,'s) dtype
val ndim  : t -> int
val dim   : int -> t -> int
val numel : t -> int
val placement : t -> 'd Placement.t
val beside    : t -> 'd Placement.t
```

## Device sets

```ocaml
type +'d devices
type host
val rigs    : 'd devices -> Rig.t list
val devices : ?kernels:(module Nx_kernel.S) -> Rig.t list -> (module Devices)

module Host : Devices with type d = host
module Placement : sig
  val on : 'd devices -> 'd t
  val device : 'd devices -> Rig.t -> 'd t
  val split : axis:int -> 'd devices -> 'd t
  val mesh : 'd Mesh.t -> (int * string list) list -> 'd t
  val uncut : 'd t -> 'd t
  val devices : 'd t -> 'd devices
  val equal : 'd t -> 'e t -> bool
  val same : 'd t -> 'e t -> ('d,'e) Type.eq option
  val pp : Format.formatter -> 'd t -> unit
end
module Mesh : sig
  val v : 'd devices -> (string * int) list -> 'd t
end

val place : 'e Placement.t -> t -> ('v,'s,'e) t
val sync  : t -> unit
```

## Creation

Host forms (`... host t`), each with `_at p` and `_like` counterparts:

```ocaml
zeros   dt shape
ones    dt shape
full    dt shape v
scalar  dt v
arange  dt start stop step          (* ints *)
arange_f dt start stop step         (* floats *)
linspace dt ?endpoint start stop n
logspace dt ?endpoint ?base start stop n
geomspace dt ?endpoint start stop n
eye ?m ?k dt n
create  dt shape values             (* 'v array, C order *)
init    dt shape (int array -> 'v)
map_item ('v -> 'v) t               (* host only, not traceable *)
of_bigarray genarray
copy    t
donate  t
```

`_like` forms: `zeros_like`, `ones_like`, `full_like`, `scalar_like`.

Random creation has `_at` forms that take `?key`:

```ocaml
normal ?key dt shape        normal_at p ?key dt shape
uniform ...                 uniform_at ...
exponential, gumbel, randint, bernoulli, gamma, beta, von_mises,
poisson, binomial, bits
```

## Reads and display

```ocaml
val to_array    : t -> 'v array
val item        : int list -> t -> 'v
val to_bigarray : ('v,'s) Bigarray.kind -> t -> genarray
val to_string   : t -> string
val print       : t -> unit
val pp          : Format.formatter -> t -> unit
```

Reads wait on the set and copy to the host. `item` of a traced value raises.

## Movement

```ocaml
val reshape      : int array -> t -> t
val broadcast_to : int array -> t -> t
val squeeze      : ?axes:int list -> t -> t
val unsqueeze    : axes:int list -> t -> t
val transpose    : ?axes:int list -> t -> t
val flip         : ?axes:int list -> t -> t
val split        : axis:int -> int -> t -> t list
val split_at     : axis:int -> int list -> t -> t list
val concatenate  : axis:int -> t list -> t
val stack        : ?axis:int -> t list -> t
val pad          : (int * int) array -> 'v -> t -> t
val tile         : int array -> t -> t
val repeat       : ?axis:int -> int -> t -> t
```

## Indexing

```ocaml
type 'd index =
  | I of int
  | L of int list
  | T of 'd int64_t
  | R of int * int
  | Rs of int * int * int
  | A
  | N
  | D of 'd int64_t * int
  | M of 'd bool_t

val slice : 'd index list -> t -> t
val get   : int list -> t -> t
val take  : ?axis:int -> 'd int64_t -> t -> t
val take_along_axis : axis:int -> 'd int64_t -> t -> t
val set     : 'd index list -> t -> t -> t          (* set idx v x *)
val scatter : ?combine:combine -> axis:int -> 'd int64_t -> t -> t -> t
type combine = Set | Add | Max | Min
val unravel_index : 'd int64_t -> int array -> 'd int64_t list
```

Infix shorthand: `x.%{p}` is `get p x`, `x.${idx}` is `slice idx x`.

## Elementwise arithmetic

```ocaml
add sub mul div pow neg
add_s sub_s mul_s div_s pow_s
maximum minimum maximum_s minimum_s
abs sign sqrt rsqrt exp exp2 expm1 log log2 log1p
sin cos tan asin acos atan atan2
sinh cosh tanh asinh acosh atanh
floor ceil round trunc mod_ mod_s
sigmoid softplus
erf erfc erfinv ndtr ndtri lgamma digamma
equal not_equal less less_equal greater greater_equal
equal_s ... greater_equal_s
logical_and logical_or logical_not
where : 'd bool_t -> t -> t -> t
clamp : ?min:'v -> ?max:'v -> t -> t
cast : ('w,'r) dtype -> t -> ('w,'r,'d) t
bitcast : ('w,'r) dtype -> t -> ('w,'r,'d) t
```

`Nx.relu` is not in `Nx` at this pin; activations live in Kaun. The lab's
one-line ReLU is `maximum x (zeros_like x)`.

## Reductions

```ocaml
sum  ?axes ?keepdims t
prod ?axes ?keepdims t
max  ?axes ?keepdims t
min  ?axes ?keepdims t
mean ?axes ?keepdims t
var  ?axes ?keepdims ?ddof t
std  ?axes ?keepdims ?ddof t
logsumexp ?axes ?keepdims t
cumsum ?axis t
cumprod ?axis t
argmax ?axis ?keepdims t     (* returns 'd int64_t *)
argmin ?axis ?keepdims t
softmax ?axes ?scale t
log_softmax ?axes ?scale t
all ?axes ?keepdims t | any ?axes ?keepdims t   (* 'd bool_t *)
associative_scan ?axis s f x
```

A reduction's `~axes` is a list. `argmax` takes `~axis`.

## Linear algebra

```ocaml
matmul a b                    (* Infix *@ *)
dot a b
contract ?sizes ?acc ?init dt pattern a b
einsum ?sizes ?acc pattern tensors
rearrange ?sizes pattern x
Pattern.v : string -> Pattern.t
Pattern.inverse : Pattern.t -> Pattern.t

matrix_transpose a
matrix_power a n
tril ?k a | triu ?k a
diagonal ?offset ?axis1 ?axis2 a
trace ?offset a
diag ?k a
eye ?m ?k dt n

cholesky ?triangle a
qr ?factors a                 (* (q, r) *)
lu a                          (* (perm, l, u) *)
svd ?factors dt a             (* (u, s, vh) *)
svdvals dt a
eigh ?triangle dt a           (* (w, v) *)
eigvalsh ?triangle dt a
eig dt a | eigvals dt a
det a
solve a b
solve_triangular ?triangle ?transpose ?unit_diag a b
inv a
lstsq ?rtol dt a b            (* (x, residuals, rank, s) *)
```

`triangle = Lower | Upper`, `factors = Reduced | Complete`,
`transpose = Transpose | Adjoint`.

## Infix

```ocaml
module Infix : sig
  val ( + ) ( - ) ( * ) ( / ) ( ** ) ( ~- ) : t -> t -> t
  val ( +$ ) ( -$ ) ( *$ ) ( /$ ) ( **$ ) : t -> 'v -> t
  val ( *@ ) : t -> t -> t
  val ( = ) ( <> ) ( < ) ( <= ) ( > ) ( >= ) : t -> t -> 'd bool_t
  val ( =$ ) ( <>$ ) ( <$ ) ( <=$ ) ( >$ ) ( >=$ ) : t -> 'v -> 'd bool_t
  val ( && ) ( || ) : 'd bool_t -> 'd bool_t -> 'd bool_t
  val ( .%{} ) : t -> int list -> t
  val ( .${} ) : t -> 'd index list -> t
  val ( land ) ( lor ) ( lxor ) : t -> t -> t
end
```

`&&` and `||` compute both operands. Use `where` or `Rune.cond` for a real
branch.

## Checks

```ocaml
val check :
  ('a,'d) Ptree.t -> 'd bool_t -> 'a -> (int array -> 'a -> exn) -> unit
```

Raises `fail i d` at the first failing element in C order. On the host it
raises at the call; on another set at the next wait on that set.

## Host interop

```ocaml
Nx_array       (* the array without the placement layer: layout, buffer *)
Nx.Repr        (* crossing between Nx values and Nx_array, borrow/copy *)
Nx.Prim        (* the operation language: interpretation, custom ops *)
Nx_kernel.S    (* the contract a kernel library implements *)
Nx_cpu         (* kernels for host memory, library nx.cpu *)
```

`Nx_array` is for code that needs strides, offsets or the `Rig.Buffer` under
a value. Application code should not need it.
