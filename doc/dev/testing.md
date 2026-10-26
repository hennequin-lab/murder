# Testing

Every package has a windtrap suite under `test/`. A suite is an ordinary
executable that `dune runtest` builds and runs.

```lisp
(test
 (name test_greeting)
 (package greeting)
 (libraries greeting windtrap))
```

```ocaml
open Windtrap

let () =
  exit
    (run "greeting"
       [ test "greets by name" (fun () ->
             equal string "hello, ada" (Greeting.greet "ada")) ])
```

## Rules

- State the `.mli` contract through the public interface.
- Use `prop` for laws, `stateful` for state across calls, `cases` for spec
  values, `expect` for long text.
- Expected values come from the spec; a value learned from the code is a
  baseline written as `expect`.
- Draw edge cases: empty and one-element shapes, bounds, dtypes, strided and
  broadcast views, NaN, infinities, integer extremes.
- Assert with `equal`, `less ~than`, `raises`; floats take a stated tolerance.
- Bug fixes start with a failing test.
- No network, wall clock, or hardware in tests. Keep data synthetic or tiny.
- A package's `runtest` should finish within about two minutes warm. Fix a slow
  test at its cause.
- Set `WINDTRAP_TIMEOUT` so a hang fails instead of blocking. CI uses a higher
  property count.
