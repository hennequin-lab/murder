"""Fail if a public value in lib/*.mli has no doc comment.

Every [val] in a package interface is part of what a user reads, so each one
carries a doc comment attached to it: the comment must be the last thing before
the [val], or sit on the same line. A group heading ({1 ...}) does not document
the values under it -- it organises them.

Usage: check_mli_docs.py <.mli files>...
       check_mli_docs.py            # every lib/*.mli of this package
"""

import re
import sys
import glob
import os

VAL = re.compile(r"^\s*val\s+(\w+)")
EOL_DOC = re.compile(r"\(\*\*.*\*\)\s*$")


def undocumented(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    missing = []
    for i, line in enumerate(lines):
        m = VAL.match(line)
        if not m:
            continue
        if EOL_DOC.search(line):
            continue
        j = i - 1
        while j >= 0 and lines[j].strip() == "":
            j -= 1
        if j >= 0 and lines[j].strip().endswith("*)"):
            continue
        missing.append((i + 1, m.group(1)))
    return missing


def main(argv):
    # Files are passed by dune as dependencies; a bare invocation scans this
    # package. Only .mli files are read, so a dependency list that also holds
    # the script itself is harmless.
    # Dune hands over every dependency of the rule, which includes the
    # preprocessed interfaces (binary ASTs, not source). Only sources count.
    paths = [
        p for p in argv if p.endswith(".mli") and ".pp." not in os.path.basename(p)
    ]
    if not paths:
        paths = sorted(glob.glob(os.path.join("lib", "*.mli")))
    bad = 0
    for path in paths:
        try:
            missing = undocumented(path)
        except UnicodeDecodeError as e:
            print(f"{path}: not readable as UTF-8 ({e})")
            bad += 1
            continue
        for line, name in missing:
            print(f"{path}:{line}: val {name} has no doc comment")
            bad += 1
    if bad:
        print(f"{bad} undocumented value(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
