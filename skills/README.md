# Raven skills for pi

Three project skills for the Raven packages this monorepo uses. Each is a
directory with a `SKILL.md` and `references/`.

| Skill | Loads when |
| --- | --- |
| [`raven-nx`](raven-nx/SKILL.md) | a task touches Nx arrays: shapes, dtypes, slicing, broadcasting, reductions, linear algebra, `Nx.Ptree` records, `Nx.Rng` keys, device sets and placement, donation, checks |
| [`raven-rune`](raven-rune/SKILL.md) | a task differentiates, vectorizes, loops or compiles Nx code: `grad`, `value_and_grad`, `jvp`, `vjp`, `vmap`, `scan`, `iterate`, `switch`, `jit`, custom rules, `root`, `remat`, `Total`, `check_grads` |
| [`raven-rig`](raven-rig/SKILL.md) | a task selects or opens a device, moves work to a GPU, adds a `-dev` flag, touches `Rig.Buffer`, `Rig.Submission` or `Rig.Image`, or explains a lost device or an eager-operation error on a GPU set |

The content is drawn from the pinned Raven sources and from the lab's own use
in `packages/` and `getting-started/`: the `open Nx` and `Infix` style, the
`[@@deriving ptree]` parameter records, `Nx.devices` and `D.on`, the
`-dev` device selection in `sofo`, `Rune.scan` over a horizon in `dilqr`, and
the `custom_jvp` + `custom_transpose` pair in `dilqr/lib/implicit_grad.ml`.

## Loading them in pi

Pi discovers project skills from the `skills` setting in the project's
`.pi/settings.json`. Paths in project settings resolve from the `.pi`
directory, so this repository points one level up:

```json
{
  "skills": ["../skills"]
}
```

Run pi from the repository root (the directory that holds `.pi/`), grant
project trust, and the three skills appear in `available_skills`. Run
`/reload` after editing a skill, or force one with `/skill:raven-nx`,
`/skill:raven-rune`, `/skill:raven-rig`.

## Scope

The skills cover Nx, Rune and Rig only. Vega, Hugin, Munin, Jera, Kaun,
`ppx_ptree` and the rest of the stack are out of scope; `Nx.Ptree` appears
where Rune and Nx need it.

When the Raven pin in `dune-project` moves, re-check the API sections against
the new `nx.mli`, `rune.mli` and `rig.mli` in `_build/_private/default/.pkg/`.
