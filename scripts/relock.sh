#!/bin/sh
# Relock, then apply the current Raven-main workaround.
set -eu
dune pkg lock "$@"
python3 scripts/patch-lock.py
