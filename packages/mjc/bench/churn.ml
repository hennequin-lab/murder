(*---------------------------------------------------------------------------
  Copyright (c) 2026 The mjc authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Can the working set be predicted?

   `bench/rows.ml` says the solver factors 21-34 rows where 3-4 of them do work,
   and the cubic term is 86% of a step, so compacting the factorisation to the
   free block is the largest arithmetic win left: 2.0-5.4x at the model's worst
   case and 145x at its median. Both need the *set* in advance, and the only
   thing in advance is the warm start.

   The numbers this prints come from `tools/measure_churn.py`, which steps
   MuJoCo along each recorded trajectory and compares the free set of one step
   with the next -- the set the warm start would offer against the set the solve
   ends at. If they agree, compaction on the warm start is exact. They do not.

   hopper    steps 613  rows med  8 max 21 | unchanged 29.7% | +0.95 / -1.16 rows
   walker    steps 626  rows med  9 max 21 | unchanged 13.4% | +1.54 / -1.74 rows
   humanoid  steps 296  rows med  9 max 33 | unchanged 11.5% | +1.41 / -1.86 rows

   So the set changes on 70-88% of steps and by up to eleven rows at once, which
   is why this is a negative result written down rather than a knob: a compacted
   block sized from the warm start would be wrong on most steps, and one sized to
   absorb the churn is a block as large as the row buffer.

   Run: python3 tools/measure_churn.py  (needs mujoco)
   dune exec packages/mjc/bench/churn.exe  (repeats the table) *)

open Base
module Printf = Stdlib.Printf

let () =
  Printf.printf
    "The working set churns every step, so a static compaction cannot be sized\n\
     from the warm start. Measured by tools/measure_churn.py:\n\n";
  Printf.printf
    "| model | steps | rows median | rows max | set unchanged | rows in | rows out |\n";
  Printf.printf "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n%!";
  List.iter
    [ "hopper", 613, 8, 21, 29.7, 0.95, 1.16
    ; "walker", 626, 9, 21, 13.4, 1.54, 1.74
    ; "humanoid", 296, 9, 33, 11.5, 1.41, 1.86
    ]
    ~f:(fun (name, steps, med, mx, same, inn, out) ->
      Printf.printf
        "| %s | %d | %d | %d | %.1f%% | %.2f | %.2f |\n%!"
        name
        steps
        med
        mx
        same
        inn
        out)
