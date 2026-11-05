open Base

(* Compiled dynamics: the structure of a model, read once, and the step that
   uses it while reading the model's *parameters* as tensors. See the interface
   for why that split is the whole point. *)
type t = { config : Step.config }

let compile m = { config = Step.config m }
let run t m d = Step.step_with t.config m d
let dt t = Step.config_dt t.config

let horizon t m d n =
  let d = ref d in
  for _ = 1 to n do
    d := run t m !d
  done;
  !d

(* The mass matrix at [d]. [Mass.forward] reads no parameter with [Nx.item], so
   the same argument applies to it as to [run]. *)
let mass t m d =
  ignore t;
  (Mass.forward m d).mass_matrix

let inner t m d a b = Nx.dot a (Nx.matmul (mass t m d) b)
