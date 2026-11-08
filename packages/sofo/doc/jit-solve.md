# Why the subspace solve runs on the host

`Sofo.Optim.coordinates` solves the damped sketched system

```
z = U (S + γI)⁻¹ Uᵀ C          G̃ = U S Uᵀ
```

with `Nx.eigh`, and `Optim.update` and `Optim.update_jit` call it on the host:
the sketch's `c`, `ggn` and `gram` are copied off the device, solved, and the
`k` coordinates are copied back for the compiled shift.

This note records why. The reason has changed once: it used to be that an
eigendecomposition could not be compiled at all, then that compiling one took
seconds, and now it is that compiling one is slower to run than the host's.

## It compiles now

Rune lowers nx's matrix routines to tolk graphs of Jacobi rotations
(`packages/rune/lib/lower_linalg.ml`: `Eigh`, `Svd`, `Qr`, `Lu`, `Cholesky`,
`Solve_triangular`), so `Nx.eigh` traced under `Rune.jit` is a program, not an
eager call. It is accurate — `1e-15` relative on the eigenvalues against the
host kernel in float64, which is the rounding of the arithmetic.

It also compiles quickly now. Raven 72044867 compiles host kernels with
`-mllvm -disable-loop-idiom-all`, which closes a blowup in LLVM 22 and later:
the loop-idiom recogniser runs a CRC recogniser on every countable loop, and it
walks the operands of a loop's running selection without marking those it has
seen. `Lower_linalg.exponent` folds a matrix's largest magnitude as a chain of
float maxima, each reading the last twice — 2³¹ paths at 32 rows — so a 24-row
`Nx.eigh` took 8.5 s to compile and a 32-row one never finished. The same
release stops a compiled `eigh`'s sweeps once the matrix settles, at most the
fixed count (on a device whose work is this process's code; a GPU keeps the
fixed count, since reading the flag each sweep would make the call synchronous).

Measured on this machine (clang 23, host, tolk's disk cache cleared per row):

| k   | compiled `eigh`: compile+first | replay     | eager `Nx.eigh` |
| --- | ------------------------------ | ---------- | --------------- |
| 8   | 114 ms                         | 0.009 ms   | 0.007 ms        |
| 24  | 126 ms (was 8 450 ms)          | 0.177 ms   | 0.026 ms        |
| 64  | 166 ms                         | 6.16 ms    | 0.169 ms        |
| 128 | 174 ms                         | 29.5 ms    | 3.84 ms         |

## Why it stays on the host anyway

Because the host's kernel is the faster one, by more than the round trip costs.
The same solve as a value — whitening the Gram, the damped inversion, and the
whitening back (`Sofo.Optim.coordinates`'s arithmetic, with `s_max` and `s_min`
as tensors) — against the eager call on the same inputs:

| k   | compiled: compile+first | compiled: replay | eager `coordinates` |
| --- | ----------------------- | ---------------- | ------------------- |
| 32  | 305 ms                  | 1.66 ms          | 0.106 ms            |
| 64  | 203 ms                  | 14.0 ms          | 0.530 ms            |
| 128 | 257 ms                  | 64.6 ms          | 8.66 ms             |

Replay agrees with the eager solve to `5e-10` relative at `k = 128` (the
whitening divides by the Gram's smallest singular values, so the rounding is
worse than the eigendecomposition's own).

On a device the comparison is starker. A SOFO step with 2 048 parameters, its
parameters, directions and sketched matrices on an AMD Strix Halo APU, the two
ways: today's host solve around the compiled shift, and one compiled program
that solves and shifts where the parameters live.

| k   | host solve + compiled shift | compiled solve + shift on the device |
| --- | --------------------------- | ------------------------------------ |
| 32  | 0.27 ms                     | 7.45 ms (compiles in 160 ms)         |
| 64  | 0.57 ms                     | 36.4 ms (145 ms)                     |
| 128 | 9.48 ms                     | 95.6 ms (149 ms)                     |

Read back every step, as a logging loop does, the same pairs are 0.24/0.47/9.67
and 6.84/33.2/87.9 ms. The round trip is the cheaper half of a step by a wide
margin: three `k×k` float64 tensors down and `k` numbers up is 393 KB at
`k = 128`, and the host solve it buys is a blocked tridiagonalization, not 38
Jacobi sweeps of tiny kernels on a device whose float64 runs at a fraction of
its float32 rate. Three things keep the device path slow: the sweeps cannot
stop early there, each sweep is a separate launch, and float64 is dear.

So the split stands: the sketch and the shift compile, the solve does not, and
what crosses is `k`-sized. That is also what keeps a step's arithmetic
identically the eager one's: `coordinates` is the function tests compare a
compiled step against.

## If it ever changed

Two changes of implementation, neither made because nothing compiles it
usefully yet: `item []` of the spectrum's largest and smallest singular value,
which a `jit` refuses, becomes tensor arithmetic (`s_max` and `s_min` are then
0-d tensors, and the three damping modes a 0-d `γ` broadcast over `s`); and the
two guards (a rank-deficient Gram, `Relative_from_bottom` with a zero smallest
singular value) become `Nx.check`s, which behave as they do now — on the host
they raise at the call, on a device at the next wait. With those, one function
is both the host solve and a compiled step.

What would change is `Optim.update_jit`, which would build one program over
`(params, dirs, ggn, c, gram, lr)` instead of a compiled shift around a host
solve; `damping` and `preconditioner` are closure constants of the program, so
they would be keyed rather than passed per call. It is worth doing if a caller
runs where the crossing is dear — another machine's host, or a device whose
float64 does not cost what this APU's does — but not before that caller
measures it.

## Reproducing

The eigendecomposition row is `Rune.jit` around one `Nx.eigh`, on a `k×k`
symmetric matrix `ggn`:

```ocaml
let compiled_eigh =
  Rune.jit
    Nx.Ptree.(tensor @-> returns (pair tensor tensor))
    (fun a -> Nx.eigh Nx.float64 a)

(* m: a k×k standard-normal draw; the first call compiles, the rest replay *)
let w, v = compiled_eigh Infix.(m *@ transpose m)
```

Clear tolk's disk cache first — `~/.cache/tolk` — or the compile is not paid
again. `DEBUG=4` prints every kernel's C source as it is rendered, and `CC` may
be pointed at a wrapper around clang that logs each invocation's time; the
two together are how the loop-idiom blowup was found (a single 24-lane `select`
of a broadcast loop index, 8.35 s of 8.36 s in `LoopIdiomRecognizePass`).

The device rows are one `Rune.jit` over `(params, dirs, ggn, c, gram, lr)` whose
`coordinates` body is the arithmetic above, against `Optim.update_jit` on the
same values, with the parameters, directions and matrices placed by
`Nx.Ptree.place` and `Nx.place`.
