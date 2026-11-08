# tolk: `args_fit` rejects a host member call whose argument is a looser view

A reproduction, and the diagnosis, of a failure that `dilqr`'s compiled
adaptive solve hits. This directory holds both so that the report can be
pointed at, and the case run, from a fixed commit.

- **Environment:** raven `72044867` (this repo's pin), OCaml 5.5.2, x86-64 with
  AVX-512, clang 23.
- **Not a dilqr bug:** `dilqr` only supplies the program. The check that fails
  is in `packages/tolk/lib/uop/spec.ml`, and the alignment it disagrees with is
  made in `packages/tolk/lib/codegen.ml`'s `launcher`.
- **Not a miscompile here:** with verification off the same program builds and
  matches the eager solve to `1e-9`. What fails is an invariant tolk states and
  relies on, which happens to be harmless on this host.

## Reproducing

```sh
# from the root of murder-wt/next
rm -rf ~/.cache/tolk      # see below: a warm tolk cache can hide the failure
dune exec packages/dilqr/scratch/tolk-member-align/repro.exe -- 1   # passes
dune exec packages/dilqr/scratch/tolk-member-align/repro.exe -- 2   # the failure
SPEC=0 dune exec packages/dilqr/scratch/tolk-member-align/repro.exe -- 2   # passes
```

Clear tolk's disk cache first, or set `CACHELEVEL=0`. Verification runs when a
program is compiled, and a program compiled by an earlier run — including the
`SPEC=0` control below, since `SPEC` is not part of what tolk shapes or keys a
program on — is served from the cache without one. With a warm cache the
horizon-2 run prints a cost instead of raising.

Horizon 1 passes and horizon 2 fails, so the loop (the rollout and backward
scan) is what brings it in. A plain `Rune.scan` over a matrix, including views
at element offsets 1 and 2, passes: the dilqr program needs its whole adaptive
solve — `Rune.cond`, `Rune.iterate`, the batched Riccati pass — to reach the
pattern, and I could not reduce it further.

The failure:

```
Fatal error: exception Invalid_argument("UOp verification failed at 31 on Ops.CALL dtypes.void 7
  [(Ops.LINEAR, dtypes.void, 'member_1'),
   (Ops.PARAM, dtypes.f64, ParamArg(0, dtypes.f64)),
   (Ops.PARAM, dtypes.f64, ParamArg(1, dtypes.f64)),
   (Ops.PARAM, dtypes.f64, ParamArg(2, dtypes.f64, align=8)),
   (Ops.PARAM, dtypes.f64, ParamArg(3, dtypes.f64)),
   (Ops.PARAM, dtypes.i32, ParamArg(5, dtypes.i32, vmin_vmax=(0, 1), addrspace=AddrSpace.ALU)),
   (Ops.RANGE, dtypes.i32, (AxisType.LOOP, 2))]
  CallInfo(None, 'member', True, False)")
```

## The check that fails

`packages/tolk/lib/uop/spec.ml`, inherited from tinygrad:

```ocaml
(* Each argument of storage starts where its parameter in the body takes it to:
   what is known of the argument's start implies what the parameter assumes,
   which its vector accesses were merged for. *)
let args_fit c =
  let args = Array.of_list (src_without_body c) in
  List.for_all
    (fun u ->
      match (op u, arg u) with
      | Op.Param, Param p
        when p.addrspace = Dtype.Global && p.slot >= 0 && p.slot < Array.length args ->
          fits_start ~param:(p.align, p.phase) (storage_phase args.(p.slot))
      | _ -> true)
    (toposort ~calls:Skip (body c))
```

with `fits_start ~param:(align, phase) (a, p) = a >= align && p mod align = phase`
(`packages/tolk/lib/uop/shape.ml:3251`). `repr_param_arg` prints `align` only
when it is not 16, so the default a parameter node is created with is **16**.

`DBGTV=1` dumps the launcher that trips it:

```
%687 = param 2 : dtype=f64 slot=2                       (* align 8, phase 0 *)
%688 = param () : dtype=i32 slot=5 bounds=[0,1] addrspace=ALU
%696 = range %695 : LOOP 2
%697 = call %686, %0, %37, %687, %39, %688, %696 : name='member' precompile=true
```

`%686` is the member's body, and it uses **its own** slot-2 parameter node,
created with the default `align=16`. The call passes `%687`, a 2-element float64
view at an 8-byte offset, `align=8`. `8 >= 16` is false, so the check rejects
the call and with it the program.

## Why the alignment disagrees

`lower_member` (`packages/tolk/lib/codegen.ml`, around line 2050) takes the
block body and its argument count only:

```ocaml
let lower_member ctx b ~nargs =
  let low = full_rewrite_to_sink ~optimize b ctx.ren in
  ...
```

so the member's `PARAM`s keep the alignment they were given when the loop's call
was built (`packages/rune/lib/loop.ml`'s `repeat`, `Ops.call
~precompile:true`), and never see the alignment of the value the call site
actually passes. `launcher` also caches members by `(body, nargs)` alone, so two
calls of one body with differently aligned arguments share one member body.

In tinygrad the equivalent body is built by `kernelize` from the call's
arguments' shape trackers, which is why upstream does not hit this. Here, any
argument that is a view at a phase the body does not expect — a row or a
half-vector of a larger float64 buffer, which is the common case — trips the
check.

## Impact

- A hard failure: no program is built, so the compiled path is unusable for the
  affected calls. In this repo `dilqr`'s "compiled adaptive pendulum matches
  eager" fails at these pins and passes with `SPEC=0`.
- The check is not wrong in principle: a body whose accesses were merged for 16
  bytes reading an 8-byte-aligned buffer is exactly what it exists to catch. On
  x86-64 the code it rejects happens to be correct, because unaligned vector
  loads are fine there; other targets need not be so forgiving. So it wants
  fixing rather than loosening.

## Suggested fix

Give the member body the call's alignments. Either:

1. pass the call's arguments to `lower_member` and build the body's `PARAM`
   nodes from their storage phases, as tinygrad's `kernelize` does; or
2. key the member cache on the arguments' `(align, phase)` as well as
   `(body, nargs)`, so a looser view gets a body built for it.

Either makes `storage_phase args.(p.slot)` imply the body's assumption, which is
the property `args_fit` states.

## Secondary: the diagnostic crashes

`DEBUG=3` and `DBGTV=1` are the ways to see the program that failed
verification, and both raise instead of printing it:

```
Not_found
  Raised at Stdlib__Hashtbl.MakeSeeded.find
  Called from Tolk__Render.render_uir.src_str  (packages/tolk/lib/uop/render.ml:149)
  Called from Tolk__Spec.type_verify           (packages/tolk/lib/uop/spec.ml:60)
```

`src_str` looks a call body up by name in a table that does not hold it. Worth
fixing with the above, since that dump is what a user needs to localize a spec
failure at all.
