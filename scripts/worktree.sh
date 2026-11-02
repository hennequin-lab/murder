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
  rm <name> [--force]         remove a clean worktree
  prune                       drop records of deleted worktree directories

The default branch is wt/<name> and the default base is main. A branch that
exists only on origin is checked out from origin/<branch> and tracks it. Set
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
    local_ref=refs/heads/$branch
    tracking_ref=refs/remotes/origin/$branch
    if [ -e "$path" ]; then
      die "worktree already exists: $path (./dev wt rm --force $name drops it)"
    fi
    # A branch is checked out in one worktree at a time, so name the checkout
    # that holds it instead of letting git answer "is already used by worktree".
    holder=$(git worktree list --porcelain | awk -v want="$local_ref" '
      /^worktree / { path = $2 }
      $1 == "branch" && $2 == want { print path }')
    if [ -n "$holder" ]; then
      die "branch $branch is checked out at $holder (use another branch, or work there)"
    fi
    # A colleague's branch may exist only on origin, having never been checked
    # out here. Fetch that one branch so naming it takes their work; a branch
    # that neither side has is a new branch, and needs no fetch. The explicit
    # refspec writes the tracking ref even in a single-branch clone, where a
    # plain `git fetch origin <branch>` writes only FETCH_HEAD.
    if [ "$branch" != "wt/$name" ] &&
      ! git show-ref --verify --quiet "$local_ref" &&
      ! git show-ref --verify --quiet "$tracking_ref"; then
      git fetch --quiet origin "+$local_ref:$tracking_ref" 2>/dev/null || true
    fi
    mkdir -p "$root"
    if git show-ref --verify --quiet "$local_ref"; then
      git worktree add "$path" "$branch"
    elif git show-ref --verify --quiet "$tracking_ref"; then
      # Start the local branch at origin/<branch> and follow it. git refuses
      # --track when remote.origin.fetch does not map the branch, as in a
      # single-branch clone, and leaves the new branch behind when it refuses:
      # drop that, start again, and say what tracking needs.
      if ! git worktree add "$path" -b "$branch" --track "origin/$branch"; then
        git branch -D "$branch" >/dev/null 2>&1 || true
        git worktree add "$path" -b "$branch" "origin/$branch"
        printf 'worktree: %s starts at origin/%s and tracks nothing\n' \
          "$branch" "$branch" >&2
        printf 'worktree: remote.origin.fetch does not map it in this clone\n' >&2
        printf 'worktree: git config --add remote.origin.fetch %s\n' \
          "+$local_ref:$tracking_ref" >&2
      fi
    else
      git rev-parse --verify --quiet "$base^{commit}" >/dev/null ||
        die "no such base: $base (./dev wt new wants a branch or commit that exists)"
      git worktree add "$path" -b "$branch" "$base"
    fi
    # Prepare the checkout the same way as a fresh clone: hooks, dev tools and
    # the .cmt files. With a warm Dune cache this is quick; the first worktree
    # on a machine pays for the compiler, dependency and tool builds.
    if ! (cd "$path" && ./dev setup); then
      printf 'worktree: ./dev setup failed in %s\n' "$path" >&2
      printf 'worktree: branch %s is checked out there\n' "$branch" >&2
      printf 'worktree: fix the cause and rerun ./dev setup in that directory,\n' >&2
      printf 'worktree: or drop the checkout with ./dev wt rm --force %s\n' "$name" >&2
      exit 1
    fi
    printf 'Worktree: %s\n' "$path"
    printf 'Branch:   %s\n' "$branch"
    printf 'Remove:   ./dev wt rm %s\n' "$name"
    ;;
  list)
    git worktree list
    ;;
  rm)
    name=${2:-}
    [ -n "$name" ] || die "usage: ./dev wt rm <name> [--force]"
    path=$root/$name
    if [ ! -d "$path" ]; then
      die "no worktree at $path"
    fi
    # Refuses when the checkout is dirty; remove the work by hand or commit
    # it first. --force is for what a failed ./dev setup leaves behind, such as
    # a _build with no commits in it. The branch is left in place either way, so
    # a merge can still use it.
    if [ "${3:-}" = --force ]; then
      git worktree remove --force "$path"
    else
      git worktree remove "$path"
    fi
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
