#!/bin/sh
# Create, list, and remove checkouts for parallel work.
#
# Dune reads every dune-project it finds under the workspace root, so a
# worktree inside the repository breaks both checkouts: a hidden directory
# builds the outer checkout, a visible one defines every package twice. Keep
# worktrees in a sibling directory; MURDER_WT_ROOT moves them elsewhere.

set -eu

die() {
  printf 'worktree: %s\n' "$1" >&2
  exit 1
}

common_dir=$(git rev-parse --path-format=absolute --git-common-dir) ||
  die "not inside a git checkout"
main_checkout=$(dirname "$common_dir")
root=${MURDER_WT_ROOT:-$(dirname "$main_checkout")/$(basename "$main_checkout")-wt}

usage() {
  cat <<EOF
Usage: ./dev wt <command> [args]

  new <name> [branch] [base]  create $root/<name> and prepare it
  list                        list worktrees
  rm <name>                   remove a clean worktree
  prune                       drop records of deleted worktree directories

The default branch is wt/<name> and the default base is main. Set
MURDER_WT_ROOT to put worktrees somewhere else.
EOF
}

case ${1:-help} in
  new)
    name=${2:-}
    [ -n "$name" ] || die "usage: ./dev wt new <name> [branch] [base]"
    case $name in
      *[!A-Za-z0-9._-]* | .* | *..*)
        die "name must be a single path segment: $name"
        ;;
    esac
    branch=${3:-wt/$name}
    base=${4:-main}
    path=$root/$name
    if [ -e "$path" ]; then
      die "worktree already exists: $path"
    fi
    mkdir -p "$root"
    if git show-ref --verify --quiet "refs/heads/$branch"; then
      git worktree add "$path" "$branch"
    else
      git worktree add "$path" -b "$branch" "$base"
    fi
    # Prepare the checkout the same way as a fresh clone: hooks, the
    # generated dev opam file, and the .cmt files. With a warm Dune cache
    # this is a few seconds; the first worktree on a machine pays for the
    # compiler and dependency build.
    (cd "$path" && ./dev setup)
    printf 'Worktree: %s\n' "$path"
    printf 'Branch:   %s\n' "$branch"
    printf 'Remove:   ./dev wt rm %s\n' "$name"
    ;;
  list)
    git worktree list
    ;;
  rm)
    name=${2:-}
    [ -n "$name" ] || die "usage: ./dev wt rm <name>"
    path=$root/$name
    if [ ! -d "$path" ]; then
      die "no worktree at $path"
    fi
    # Refuses when the checkout is dirty; remove the work by hand or commit
    # it first. The branch is left in place so a merge can still use it.
    git worktree remove "$path"
    git worktree prune
    printf 'Removed %s\n' "$path"
    ;;
  prune)
    git worktree prune -v
    ;;
  help | *)
    usage
    ;;
esac
