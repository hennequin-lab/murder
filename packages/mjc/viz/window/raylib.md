# The raylib window (V1)

`window.ml` and `main.ml` are written; the dune stanza that builds them is
commented out because the workspace resolves libraries from `dune.lock` and
raylib is not in it. Any `(libraries raylib ...)` stanza fails in this
workspace until it is.

The one step that adds it, in the order that keeps the churn reviewable:

1. Declare it. In `dune-project`, the `mjc` package's `depends` gains
   `raylib`. (A separate `mjc-viz` package would do as well; either way the
   lock only resolves what some package depends on.)
2. `dune pkg lock`. Note that it rewrites more than the new entry: run it in a
   clean tree and read `git diff --stat dune.lock` before committing. When this
   was tried with raylib absent from any `depends`, it modified `base`,
   `hugin`, `jera`, `lz`, `nx` and `lock.dune` without adding raylib, and the
   changes were reverted.
3. Uncomment the stanza in `dune` here, then
   `dune exec packages/mjc/viz/window/main.exe -- --model packages/mjc/test/fixtures/blocks.mjcb`.

Keys: `space` pause, `.` single step, `r` reset, `g` grid, `f` frames,
`c` contacts, `x` forces, `esc` quit; right-drag orbits, left-drag pans, wheel
zooms. `--check` runs three frames headless and exits 0, printing `skipped`
when no display is available.

The drawing is all from `Mjc_viz.Scene`, so the window and the stills backend
show the same picture; only the primitive calls differ (`draw_sphere`,
`draw_capsule`, `draw_line_3d` for box edges).
