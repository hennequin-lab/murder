#!/bin/sh
set -eu

name=${1:?usage: new-package.sh NAME}
dest="packages/$name"

if [ -e "$dest" ]; then
  echo "$dest already exists" >&2
  exit 1
fi

cp -R packages/_template "$dest"
find "$dest" -type f -print0 |
  xargs -0 perl -pi -e "s/\@NAME\@/$name/g"

echo "Created $dest"
echo "Add its (package ...) stanza to dune-project, then run ./dev fmt."
