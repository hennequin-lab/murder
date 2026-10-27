# Setup

## Install Dune

Use Dune 3.24.2 or newer. The standalone installer is the simplest:

```sh
curl -fsSL https://get.dune.build/install | sh
```

Put `~/.local/bin` on your `PATH` if the installer did not. Check:

```sh
dune --version
```

Dune package management downloads the OCaml compiler and the project's
dependencies on the first build. You do not need an opam switch.

## Clone and build

```sh
git clone git@github.com:hennequin-lab/murder.git
cd murder
./dev setup
dune exec getting-started/01-nx/ex01_arrays.exe
```

The first build downloads Raven and its dependencies and can take a few
minutes. Later builds are incremental. `./dev setup` also installs the dev
tools (`ocamlformat`, `ocamlmerlin`, `ocamllsp`, `odoc`) into `_build`, so the
formatter and editor tooling are available without touching an opam switch.

## Editor

Install the OCaml platform extension for VS Code, or `ocaml-eglot`/`ocaml-lsp`
for Emacs/Vim. `./dev setup` already installed the language server and merlin;
put them on your `PATH` with:

```sh
eval $(dune tools env)
```

## Daily commands

```sh
./dev fmt     # format
./dev check   # format check, lint, tests
./dev test    # tests
```

## Troubleshooting

- **`dune: command not found`:** add `~/.local/bin` to `PATH`.
- **A package is missing from the lock:** run `./dev lock`; see
  `CONTRIBUTING.md` for the current Raven pin.
- **`Program ocamlformat not found in the tree or in PATH`, or an ocamlformat
  version mismatch:** run `dune tools install ocamlformat`. Dune installs the
  version named in `.ocamlformat` and relocks when that file changes.
- **A weird incremental build:** `rm -rf _build` and rebuild.
