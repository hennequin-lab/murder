# Why the subspace solve is eager, not compiled

`Sofo.Optim.coordinates` solves the damped sketched system

```
z = U (S + γI)⁻¹ Uᵀ C          G̃ = U S Uᵀ
```

with `Nx.eigh`, and `Optim.update`, `Optim.update_jit`, `update_sketch` and
`step` call it eagerly: the sketch and the shift are compiled, the solve in
between is not. The note records why, and where "eager" runs.

The reason has changed three times. It began that an eigendecomposition could
not be compiled at all; then that compiling one cost seconds and took a fixed
sweep count; then that the compiled program was the slow one to replay. Now the
eager side is not only the host's kernel: every eager backend Raven ships
(`nx.cpu`, `nx.cuda`, `nx.amd`) carries a spectral routine of its own, and it is
the one worth running.

## What each side computes

Rune lowers `Nx.eigh` to a tolk graph of Jacobi rotations
(`packages/rune/lib/lower_linalg.ml`), whatever the order and whatever the
device, at a fixed sweep count on a GPU. Jacobi is a fine algorithm at order 32
and a poor one past it: it rotates every pair of columns, so it does `O(k³)`
work per sweep and needs tens of sweeps of kernels too small to fill a GPU.

An eager backend does what a LAPACK routine does. The host's `nx.cpu` runs
ports of LAPACK's (with ReLAPACK's recursive splittings). `nx.cuda` and
`nx.amd` rotate in LDS by cyclic Jacobi at order 32 or less, and past that
reduce to a bidiagonal matrix, find its singular values by bisection or by
implicit QR sweeps, and apply the reflectors —
`packages/nx/lib/cuda/spectral.mli` and `nx_amd.mli` state the bounds and the
iteration limits. Same contract, the arithmetic the routine's doc bounds.

Measured on this machine (Intel i9-9900K host, Radeon AI PRO R9700 / gfx1201
through `nx.amd`, OCaml 5.5.2), one solve of a well-conditioned `k×k` `G̃` and
a `k`-vector `c`, in float64, as `Sofo.Optim.coordinates` computes it
(`Relative_from_top 1e-6`, `Inverse`):

| k   | eager, host | eager, GPU | compiled, GPU: compile+first | compiled, GPU: replay |
| --- | ----------- | ---------- | ---------------------------- | --------------------- |
| 32  | 0.10 ms     | 1.5 ms     | 18 ms                        | 8 ms                  |
| 64  | 0.35 ms     | 5.9 ms     | 38 ms                        | 19 ms                 |
| 128 | 1.55 ms     | 7.0 ms     | 82 ms                        | 43 ms                 |

So the eager GPU solve is three to six times the compiled program's replay at
every order, and the compiled one pays 18–82 ms before its first. The
compiled program is accurate — it agrees with the eager solve to `1e-13`
relative at `k = 32`, `1.6e-12` at 64 and `4.8e-12` at 128, on the host — which
was never the problem. Speed is, and it is what the backends exist to supply.

## Where the eager solve runs

It runs where the sketch ran, when that set computes eagerly:
`Optim.update_jit` reads `Nx.Devices.has_kernels` and calls `coordinates` at the
step's placement. A set minted with `~kernels:(module Nx_cuda)` or
`(module Nx_amd)` then solves on the GPU and moves nothing but the `k`
coordinates, which the compiled shift reads where they already lie.

A set minted without kernels — `Nx.devices [ gpu ]` — computes eagerly with
nothing, only under `Rune.jit`. There the solve has nowhere to go but the host,
and
`update_jit` places the three `k×k` tensors there and the `k` coordinates back:
the round trip the library used to make unconditionally.

That round trip is now the thing to avoid, not the thing to prefer. Copying
three `k×k` float64 tensors down and `k` numbers up is 393 KB at `k = 128`,
which is nothing; the *wait* is what it costs. The first `Nx.place` onto the
host waits for the sketch's queued work, the host kernel runs `O(k³)` while the
GPU sits idle, and the shift is submitted after it, so every iteration stalls
once. Measured on the same machine, the same three tensors down and one up with
the wait included: 4 ms at `k = 32`, and 4–13 ms at 64 and 128 over repeated
runs, the spread being the card's clock rather than the bytes. The eager GPU
solve is 1.5–7 ms over the same orders, with no wait in it.

## One function, both sides

`coordinates` used to read `s_max` and `s_min` back with `item` and refuse a
spectrum with a host comparison, which is why an earlier version of this note
described the two edits a compiled solve would need. Those edits are made: the
extrema are 0-d tensors, the damping is a 0-d tensor that broadcasts over the
spectrum, and the two refusals are `Nx.check`s, which raise at the call on the
host and at the next wait on a device. So the same `coordinates` is the eager
solve and, traced under `Rune.jit`, a compiled program — the compiled rows above
are that program, not a copy of its arithmetic. Nothing here compiles it: the
eager kernel is faster.

## Reproducing

Host and device rows are one `Sofo.Optim.coordinates ~damping:(`Relative_from_top
1e-6)` call on `ggn = m mᵀ`, `m` a `k×k` standard-normal draw, and a normal
`c`, the GPU values placed with `Nx.place`; time `n` calls and read the last
result, so the queue is waited for once. The compiled rows are the same call
inside

```ocaml
Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) (fun ggn c ->
  Sofo.Optim.coordinates ~damping:(`Relative_from_top 1e-6) ggn c)
```

on the placed values. Clear tolk's disk cache (`~/.cache/tolk`) first, or the
compile is not paid again; `DEBUG=4` prints every kernel's C source as it is
rendered. The round-trip row is three `Nx.place Nx.Host.on` of the `k×k`
tensors, the solve, and `Nx.place` of the result back at the step's placement.
