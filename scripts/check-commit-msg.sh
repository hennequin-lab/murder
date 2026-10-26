#!/bin/sh
# Check a Conventional Commit subject; used by the commit-msg hook and CI.
set -eu

usage() {
  echo "usage: $0 COMMIT_MSG_FILE | $0 --subject SUBJECT" >&2
  exit 2
}

subject=
case "${1:-}" in
  --subject)
    [ "$#" -eq 2 ] || usage
    subject=$2
    ;;
  *)
    [ "$#" -eq 1 ] || usage
    subject=$(sed -n '1p' "$1")
    ;;
esac

types='feat|fix|perf|refactor|test|docs|bench|chore'
pattern="^($types)(\([a-z0-9][a-z0-9_-]*\))?!?: [A-Z].*$"

if [ -z "$subject" ] || ! printf '%s\n' "$subject" | grep -Eq "$pattern"; then
  cat >&2 <<EOF
Bad commit subject: $subject

Expected:
  type(scope): Imperative summary

Types: feat, fix, perf, refactor, test, docs, bench, chore
Scope: a package name, or repo, ci, docs, deps, tutorials,
       getting-started, dev

Example:
  fix(parser): Reject empty input
EOF
  exit 1
fi

case "$subject" in
  *.)
    echo "Bad commit subject: do not end the subject with a period." >&2
    exit 1
    ;;
esac

len=$(printf '%s' "$subject" | wc -c | tr -d ' ')
if [ "$len" -gt 72 ]; then
  echo "Bad commit subject: $len characters; keep it to 72 or fewer." >&2
  exit 1
fi

exit 0
