Small utility for dealing with command line arguments in OCaml.

Some examples:

```ocaml
let dir = Cmdargs.(get_string "-d" |> force ~usage:"-d dir")
let a = Cmdargs.(get_float "-a" |> default 3.14)
let c = Cmdargs.check "-c"
```


