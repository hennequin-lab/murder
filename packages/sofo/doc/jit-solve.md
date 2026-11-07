# Why the subspace solve runs on the host

`Sofo.Optim.coordinates` solves the damped sketched system

```
z = U (S + γI)⁻¹ Uᵀ C          G̃ = U S Uᵀ
```

with `Nx.eigh`, and `Optim.update` and `Optim.update_jit` call it on the host:
the sketch's `c`, `ggn` and `gram` are copied off the device, solved, and the
`k` coordinates are copied back for the compiled shift. This note records why
that is still the right choice, and what would have to change for the solve to
join the compiled half.

It is **not** that the eigendecomposition cannot be compiled any more. It can.

## Raven compiles it

Rune lowers nx's matrix routines to tolk graphs of Jacobi rotations
(`packages/rune/lib/lower_linalg.ml`, `Eigh`, `Eigh_values`, `Svd`, `Svd_values`,
`Qr`, `Lu`, `Cholesky`, `Solve_triangular`), so `Nx.eigh` traced under
`Rune.jit` is a program, not an eager call. It is also accurate: a compiled
`Nx.eigh` of a float64 `k×k` symmetric matrix agrees with the eager one to
`1e-15` relative on the eigenvalues, which is the rounding of the arithmetic,
not of the algorithm.

Measured on the pinned Raven (clang 23, `-O2 -march=native`, tolk's disk cache
cleared before each row, `k×k` symmetric matrices):

| k  | compiled `eigh` compile+first call | replay     | eager `Nx.eigh` |
| -- | ---------------------------------- | ---------- | --------------- |
| 8  | 78 ms                              | 0.029 ms   | 0.062 ms        |
| 16 | 99 ms                              | 0.259 ms   | 0.069 ms        |
| 20 | 151 ms                             | 0.401 ms   | 0.083 ms        |
| 24 | **8 450 ms**                       | 0.703 ms   | 0.092 ms        |

Replay is fine: about eight times the host kernel's time at `k = 24`, on a step
whose whole point is that `k ≪ P`. Compilation is not: 151 ms at `k = 20` and
8.5 s at `k = 24` is a cliff, not a slope, and the width of the vector that
falls off it is the reduce axis's extent — the whole axis, upcast in one piece —
so `k = 32` did not finish compiling in the several minutes it was given. SOFO's
own example sketches along `k = 128`.

## What takes the 8.5 seconds

One kernel. At `k = 24` the compiled `eigh` renders 15 kernels totalling about
60 KB of C; the other fourteen compile in at most 70 ms each, and one,
`r_24_24`, takes the whole 8.5 s. Its source is 10 KB — 213 lines — so this is
not a size problem but a compile-time blowup in one optimization pass:

```
$ clang -O2 -ftime-report -c r_24_24.c ...
  8.3636 (100.0%)  Total
  8.3561 ( 99.9%)  Optimizer
  8.3511 ( 99.9%)  LoopIdiomRecognizePass
  0.0047 (  0.1%)  Machine code generation
```

`-O0` compiles the same file in 25 ms; `-O1` and `-O2` both take 8.6 s;
`-fno-vectorize -fno-slp-vectorize` still takes 8.4 s. There is no compiler flag
that turns the pass off (`-mllvm -disable-loopidiom` is not an LLVM option; it
fails with `Unknown command line argument`).

The construct it chokes on is how tolk vectorizes a *mask over a reduce axis*.
The kernel holds

```c
typedef int int24 __attribute__((aligned(4),ext_vector_type(24)));
typedef double double24 __attribute__((aligned(8),ext_vector_type(24)));
...
double24 alu2 = __builtin_convertvector((Lidx0<(int24){0,1,...,23}), long24)
              ? (double24){val24,val6,val17,val22,val23,val1,...}
              : (double24){val31[0],val31[1],...};
```

a 24-lane select between two lane-wise layouts of the same 24 elements, whose
condition is a *broadcast of the loop index compared against an arange*. LLVM's
loop-idiom recognizer then runs SCEV over that pattern inside every enclosing
loop, and its cost explodes with the number of lanes — 20 lanes are fine,
24 lanes cost 8.5 s.

The masks come from the lowering, not from the caller: `lower_linalg.ml` builds
them from `row_index`/`column_index` aranges — `lower` and `twice` in `eigh`,
`diagonal_within` and `finite`, and every reduction through `over_matrix`, which
reduces the last two axes — so each of those lanes is a mask the code generator
must express as a vector of comparisons. A plain compiled `Nx.sum` of a `24×24`
is 52 ms, and a compiled `eigh` at `k = 20` is 151 ms: the blowup is specific to
the masked reduces of the Jacobi lowering, not to wide reductions as such.

The kernel search is not involved: `BEAM` is 0 by default, so `Rune.jit` runs no
search.

## What would unblock it

Any one of these:

1. **Tolk**: emit the mask of a reduce axis without a broadcast loop index
   compared against a materialized arange — the lane index is a compile-time
   property of the upcast — or cap the upcast of a masked reduce at 8–16 lanes.
   This is the cheap, in-house fix, and it would also shrink the generated
   sources.
2. **LLVM**: the `LoopIdiomRecognizePass` blowup on wide
   `select`-of-broadcast-index chains is a compiler bug of its own; a flag to
   disable the pass would do as a workaround.
3. **A hand-written kernel** for the spectrum (or for the whole solve), which
   is a Raven-side feature: nx has no custom-kernel escape hatch.

Until then the solve stays where it is: `k×k` on the host, O(k³), 0.09 ms at
`k = 24`, three small copies per iteration against a forward pass over `P`
parameters. That round trip — three small tensors down, the `k` coordinates
up — is the only eager work left in a step.

## Reproducing

The measurement above is `Rune.jit` around one `Nx.eigh`, on a `k×k` symmetric
matrix `ggn`:

```ocaml
let compiled_eigh =
  Rune.jit
    Nx.Ptree.(tensor @-> returns (pair tensor tensor))
    (fun a -> Nx.eigh Nx.float64 a)

(* m: a k×k standard-normal draw; the first call compiles, the rest replay *)
let w, v = compiled_eigh Infix.(m *@ transpose m)
```

Clear tolk's disk cache first — `~/.cache/tolk` — or the compile is not paid
again. Two settings make the diagnosis:

- `DEBUG=4` prints every kernel's C source as it is rendered, so the offending
  kernel can be extracted and timed on its own;
- `CC` may be pointed at a wrapper around clang that logs each invocation's
  time, which attributes the compile to a source without guessing.

## If it is ever fixed

`Optim.coordinates` needs no change of interface, but two of implementation,
neither made here because nothing can compile it usefully yet: `item []` of the
spectrum's largest and smallest singular value, which a `jit` refuses, becomes
tensor arithmetic (`s_max` and `s_min` are then 0-d tensors, and the three
damping modes a 0-d `γ` broadcast over `s`); and the two guards (a
rank-deficient Gram, `Relative_from_bottom` with a zero smallest singular value)
become `Nx.check`s, which behave as they do now — on the host they raise at the
call, on a device at the next wait. With those, one function is both the host
solve and a compiled step, which is the point.

What would change is `Optim.update_jit`, which would build one program over
`(params, dirs, ggn, c, gram, lr)` instead of a compiled shift around a host
solve; `damping` and `preconditioner` are closure constants of the program, so
they would be keyed rather than passed per call. The numbers `coordinates`
returns are the ones `update` already uses, so the tests that compare a compiled
step with an eager one cover the change.
