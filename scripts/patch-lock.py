#!/usr/bin/env python3
"""Patch a freshly generated dune.lock for the current Raven main.

Two upstream issues make a straight `dune pkg lock && dune build` fail:

1. Dune builds a lock package with `dune build -p <pkg> @install`. Raven's
   private test libraries under packages/hugin/test are not attached to a
   package, so `@install` builds them too, and one of them copies
   packages/hugin/lib/marks.{ml,mli} from outside its package. Building
   `<pkg>.install` instead installs exactly the package and skips them.

2. GitHub's archive endpoint returns 504 for some Jane Street tarballs. The
   same bytes are served by codeload; the checksum in the lock still verifies.

Remove this script (and use plain `dune pkg lock`) once both are fixed
upstream. Never hand-edit dune.lock: run `./dev lock` so this patch is applied.
"""

import re
from pathlib import Path

ARCHIVE = re.compile(
    r"https://github\.com/([^/\s]+)/([^/\s]+)/archive/refs/tags/([^\s)]+)\.tar\.gz"
)
RAVEN = "git+https://github.com/raven-ml/raven.git"

for path in Path("dune.lock").glob("*.pkg"):
    text = path.read_text()
    if RAVEN in text:
        text = text.replace(
            "(build\n (all_platforms ((dune))))",
            "(build\n (all_platforms ((action (run dune build -p %{pkg-self:name} -j %{jobs} %{pkg-self:name}.install)))))",
        )
    text = ARCHIVE.sub(
        r"https://codeload.github.com/\1/\2/tar.gz/refs/tags/\3", text
    )
    path.write_text(text)
