(*---------------------------------------------------------------------------
  Copyright (c) 2026 The SYMO authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Illustration of loss lanscape and associated symmetries in a simple MLP *)

open Base
open Nx
open Symo

let in_dir = Cmdargs.in_dir "-d"
let hidden = 3
let input_dim = 1
let output_dim = 1

module SimpleMLP = struct
  module P = struct
    type 'a t =
      { w1 : 'a
      ; w2 : 'a
      }
    [@@deriving ptree, sexp]

    let dims = { w1 = [ input_dim; hidden ]; w2 = [ hidden; output_dim ] }

    (* The hidden units are permuted (group 0); the input and output axes are
       free. *)
    let symmetries : Symmetry.spec list t =
      { w1 = [ Symmetry.Id; Symmetry.Perm 0 ]; w2 = [ Symmetry.Perm 0; Symmetry.Id ] }
  end

  let init () =
    let w1 = randn float32 [| input_dim; hidden |] in
    let w2 = randn float32 [| hidden; output_dim |] in
    P.{ w1 = div w1 (norm w1); w2 = div w2 (norm w2) }

  let forward (p : float32_t P.t) x =
    (* input = horizon x bs x input_dim *)
    Infix.(relu (x *@ p.w1) *@ p.w2)
end

let teacher = SimpleMLP.init ()
let student = SimpleMLP.init ()
let input = randn float32 [| 20; input_dim |]
let target = SimpleMLP.forward teacher input

let loss (p : Nx.float32_t SimpleMLP.P.t) =
  let pred = SimpleMLP.forward p input in
  mean (square Infix.(pred - target))

module S = Symo.Make (SimpleMLP.P)

let s2 =
  S.Second_order.(
    dense_of_factors ~symmetric:true (factors_of_pair ~symmetric:true teacher teacher))

let pcs =
  let u, _, _ = svd s2 in
  slice [ A; R (0, 2) ] u

(* save the loss landscape in the two 2 PCs of the orbit *)
let landscape =
  let to_vec (p : _ SimpleMLP.P.t) =
    concatenate ~axis:0 [ reshape [| -1; 1 |] p.w1; reshape [| -1; 1 |] p.w2 ]
  in
  let of_vec v =
    let d1 = input_dim * hidden
    and d2 = output_dim * hidden in
    let w1 = slice [ R (0, d1) ] v in
    let w2 = slice [ R (d1, d1 + d2) ] v in
    SimpleMLP.P.
      { w1 = reshape [| input_dim; hidden |] w1
      ; w2 = reshape [| hidden; output_dim |] w2
      }
  in
  let xs, ys =
    let z = linspace float32 (-3.) 3. 100 in
    meshgrid z z
  in
  let xs = reshape [| 1; -1 |] (contiguous xs) in
  let ys = reshape [| 1; -1 |] (contiguous ys) in
  let grid =
    concatenate ~axis:0 [ xs; ys ]
    (* 2 x mesh_dim *)
  in
  let deltas = Infix.(pcs *@ grid) in
  let n = dim 1 deltas in
  let teacher_v = to_vec teacher in
  Array.init n ~f:(fun i ->
    let delta = slice [ A; I i ] deltas in
    let w = of_vec Infix.(teacher_v + reshape [| -1; 1 |] delta) in
    loss w |> item [])
  |> create float32 [| n |]
  |> reshape [| 100; 100 |]

let _ = Nx_io.save_txt (in_dir "landscape") landscape

let _ =
  (* The landscape is a field of losses; the heatmap and the filled contours
     read the same log-spaced fill scale, stepped at the contour levels. *)
  let field = log landscape in
  let levels = Array.map ~f:Float.log [| 1e-3; 1e-2; 1e-1; 1e-0 |] in
  let fill =
    Hugin.num
      ~scale:
        (Hugin.Scale.linear
           ~domain:(Float.log 1e-3, 0.)
           ~ticks:levels
           ~scheme:Hugin.Scheme.inferno
           ())
      field
  in
  let heatmap = Hugin.rect ~x:(Hugin.dim 1) ~y:(Hugin.dim 0) ~fill () in
  let contour = Hugin.contour ~x:(Hugin.dim 1) ~y:(Hugin.dim 0) ~fill () in
  Hugin.layer [ heatmap; contour; Hugin.frame () ] |> Hugin.save (in_dir "hi.png")
