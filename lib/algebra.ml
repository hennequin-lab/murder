module type SCALAR = sig
  (** The buffer element kind of {!Nx}, such as {!Nx.float32_elt}. *)
  type kind

  (** The data type descriptor used to build constants. *)
  val dtype : (float, kind) Nx.dtype
end

module type SIGNATURE = sig
  (** The metric signature of the algebra. *)
  val signature : Signature.t
end

(** {1:algebras Algebras} *)

module type ALGEBRA = sig
  (** The scalar kind of the algebra. *)
  type kind

  (** The type for multivectors. The last axis is the blade axis. *)
  type t = (float, kind) Nx.t

  (** The data type descriptor of the scalar kind. *)
  val dtype : (float, kind) Nx.dtype

  (** The metric signature. *)
  val signature : Signature.t

  (** [n] is the ambient dimension, {!Signature.dim} of the signature. *)
  val n : int

  (** [size] is the number of blades, [2^n]. *)
  val size : int

  (** {1:construction Construction} *)

  (** [scalar x] is the unbatched scalar multivector with coefficient [x] on
      the identity blade. *)
  val scalar : float -> t

  (** [one] is [scalar 1.]. *)
  val one : t

  (** [zero] is [scalar 0.]. *)
  val zero : t

  (** [basis b] is the unbatched multivector with coefficient [1.] on blade
      [b] and zero elsewhere.

      Raises [Invalid_argument] if [Blade.to_int b >= size]. *)
  val basis : Blade.t -> t

  (** [e i] is the basis vector [e_i], that is, [basis (Blade.of_basis i)].

      Raises [Invalid_argument] if [i] is outside [[1];[n]]. *)
  val e : int -> t

  (** [pseudoscalar] is [e_1 ... e_n]. *)
  val pseudoscalar : t

  (** [of_array a] is the unbatched multivector with coefficients [a], in
      increasing blade-index order.

      Raises [Invalid_argument] if [Array.length a <> size]. *)
  val of_array : float array -> t

  (** [to_array a] is the coefficient array of an unbatched multivector, in
      increasing blade-index order.

      Raises [Invalid_argument] if [a] has batch axes. *)
  val to_array : t -> float array

  (** [of_nx x] views [x] as a multivector, checking that its last axis has
      length [size].

      Raises [Invalid_argument] if the last axis is not [size]. *)
  val of_nx : (float, kind) Nx.t -> t

  (** [to_nx a] is [a] as an ordinary tensor. It is the identity; the function
      exists to make the representation explicit at API boundaries. *)
  val to_nx : t -> (float, kind) Nx.t

  (** {1:shape Shape and coefficients} *)

  (** [shape a] is the shape of [a], including the trailing blade axis. *)
  val shape : t -> int array

  (** [batch_shape a] is the shape of [a] without the trailing blade axis. *)
  val batch_shape : t -> int array

  (** [coeff a b] is the coefficient of blade [b] in [a]: a tensor of shape
      [batch_shape a], without the blade axis. *)
  val coeff : t -> Blade.t -> (float, kind) Nx.t

  (** [with_coeff a b c] is [a] with the coefficient of [b] replaced by [c].
      [c] must have shape [batch_shape a], or broadcast to it.

      Raises [Invalid_argument] if the shapes are incompatible. *)
  val with_coeff : t -> Blade.t -> (float, kind) Nx.t -> t

  (** [grade k a] keeps the grade-[k] part of [a] and zeros the rest. For [k]
      outside [[0];[n]] the result is zero. *)
  val grade : int -> t -> t

  (** [even a] keeps the even-grade part of [a]. *)
  val even : t -> t

  (** [odd a] keeps the odd-grade part of [a]. *)
  val odd : t -> t

  (** {1:products Products} *)

  (** [gp a b] is the geometric product [a b]. Batch axes broadcast according
      to {!Nx}. *)
  val gp : t -> t -> t

  (** [op a b] is the outer (exterior) product [a and b]. *)
  val op : t -> t -> t

  (** [inner a b] is the left contraction [a dot b]. For vectors it is the
      ordinary dot product. *)
  val inner : t -> t -> t

  (** [scalar_prod a b] is the scalar product [<tilde(a) b>_0], a tensor of
      shape [batch_shape] after broadcasting. *)
  val scalar_prod : t -> t -> (float, kind) Nx.t

  (** [commutator a b] is [(a b - b a) / 2], the Lie bracket of bivectors. *)
  val commutator : t -> t -> t

  (** {1:involutions Involutions} *)

  (** [reverse a] reverses the order of every product in [a]. On a grade-[k]
      blade it acts as [(-1)^(k (k - 1) / 2)]. *)
  val reverse : t -> t

  (** [grade_involution a] multiplies the grade-[k] part of [a] by [(-1)^k]. *)
  val grade_involution : t -> t

  (** [conjugate a] is the Clifford conjugate of [a], the composition of
      {!reverse} and {!grade_involution}. *)
  val conjugate : t -> t

  (** {1:duals Duals} *)

  (** [dual a] is the right Hodge dual [a I^(-1)], of grade [n - k] when [a]
      has grade [k].

      Raises [Invalid_argument] if the signature is degenerate, in which case
      [I] is not invertible. For a degenerate algebra use {!complement}. *)
  val dual : t -> t

  (** [undual a] is [a I], the inverse of {!dual}:
      [undual (dual a) = a].

      Raises [Invalid_argument] if the signature is degenerate. *)
  val undual : t -> t

  (** [complement a] is the signed Poincaré complement: on a basis blade
      [e_b], it is [complement_sign ~n b * e_(complement ~n b)]. It is defined
      for every signature and is the dual used in projective geometric
      algebra. *)
  val complement : t -> t

  (** [regressive a b] is the regressive (meet) product of [a] and [b],
      [a or b], defined through {!complement}:

      {[
       a or b = complement (complement a and complement b)
      ]} *)
  val regressive : t -> t -> t

  (** {1:norms Norms and inverses} *)

  (** [norm2 a] is the scalar [<tilde(a) a>_0], of shape [batch_shape a]. For a
      blade it is the squared magnitude up to sign. *)
  val norm2 : t -> (float, kind) Nx.t

  (** [norm a] is [sqrt (abs (norm2 a))], the magnitude of a blade. *)
  val norm : t -> (float, kind) Nx.t

  (** [normalize a] is [a] divided by {!norm}[ a]. For a null blade the result
      is unspecified (division by zero); check {!norm2} first. *)
  val normalize : t -> t

  (** [inverse a] is the inverse of a blade or versor,
      [tilde(a) / <tilde(a) a>_0]. For a general multivector the inverse may
      not exist; the function does not detect this. *)
  val inverse : t -> t

  (** {1:versors Versors and actions} *)

  (** [sandwich v x] is [v x v^(-1)], the action of the versor [v] on [x]. The
      inverse is computed from [tilde(v) v]; for a unit versor it is
      [tilde(v)] up to sign. *)
  val sandwich : t -> t -> t

  (** [adjoint v] is the matrix of the linear map [x -> v x v^(-1)] in the blade
      basis. Its shape is [batch_shape v @ [|size; size|]]; for a single versor
      it is [size] by [size]. *)
  val adjoint : t -> (float, kind) Nx.t

  (** {1:printing Printing} *)

  (** [pp ppf a] formats [a] for debugging. For a batched value it prints the
      shape and the coefficients of each batch element. *)
  val pp : Format.formatter -> t -> unit
end

(* --- implementation ------------------------------------------------------- *)

let product_shape s = Array.fold_left ( * ) 1 s
let invalid name msg = invalid_arg (name ^ ": " ^ msg)

module Make (S : SIGNATURE) (K : SCALAR) = struct
  type kind = K.kind
  type t = (float, kind) Nx.t

  let dtype = K.dtype
  let signature = S.signature
  let n = Signature.dim signature
  let size = Signature.size signature
  let full = size - 1
  let sigma = Signature.to_array signature
  let pseudoscalar_square = Signature.pseudoscalar_square signature
  let blade_grade i = Blade.grade (Blade.of_int i)
  let blade_mul_sign i j = Blade.mul_sign (Blade.of_int i) (Blade.of_int j)

  let metric_factor i j =
    let common = i land j in
    let rec loop bit acc =
      if bit >= n
      then acc
      else if common land (1 lsl bit) <> 0
      then loop (bit + 1) (acc * sigma.(bit))
      else loop (bit + 1) acc
    in
    loop 0 1

  let make_table f =
    let a = Array.make (size * size * size) 0. in
    for i = 0 to size - 1 do
      for j = 0 to size - 1 do
        let v = f i j in
        if v <> 0 then a.((((i lxor j * size) + i) * size) + j) <- float_of_int v
      done
    done;
    a

  let gp_table =
    make_table (fun i j ->
      let m = metric_factor i j in
      if m = 0 then 0 else blade_mul_sign i j * m)

  let op_table = make_table (fun i j -> if i land j = 0 then blade_mul_sign i j else 0)

  let inner_table =
    make_table (fun i j ->
      if i land j = i
      then (
        let m = metric_factor i j in
        if m = 0 then 0 else blade_mul_sign i j * m)
      else 0)

  let gp_tbl = Nx.create dtype [| size; size; size |] gp_table
  let op_tbl = Nx.create dtype [| size; size; size |] op_table
  let inner_tbl = Nx.create dtype [| size; size; size |] inner_table

  let rev_sign =
    Array.init size (fun i -> float_of_int (Blade.reverse_sign (Blade.of_int i)))

  let gi_sign =
    Array.init size (fun i -> float_of_int (Blade.grade_involution_sign (Blade.of_int i)))

  let conj_sign =
    Array.init size (fun i ->
      float_of_int (Blade.clifford_conjugation_sign (Blade.of_int i)))

  let rev_vec = Nx.create dtype [| size |] rev_sign
  let gi_vec = Nx.create dtype [| size |] gi_sign
  let conj_vec = Nx.create dtype [| size |] conj_sign

  let complement_perm =
    Array.init size (fun i -> Blade.to_int (Blade.complement ~n (Blade.of_int i)))

  let complement_sign =
    Array.init size (fun i -> float_of_int (Blade.complement_sign ~n (Blade.of_int i)))

  let complement_idx =
    Nx.create Nx.int64 [| size |] (Array.map Int64.of_int complement_perm)

  let complement_vec = Nx.create dtype [| size |] complement_sign
  let dual_perm = Array.init size (fun i -> full lxor i)

  let dual_sign =
    Array.init size (fun i ->
      if pseudoscalar_square = 0
      then 0.
      else (
        let s = blade_mul_sign i full in
        let m = metric_factor i full in
        float_of_int s *. float_of_int m /. float_of_int pseudoscalar_square))

  let undual_sign =
    Array.init size (fun i ->
      let s = blade_mul_sign i full in
      let m = metric_factor i full in
      float_of_int (s * m))

  let dual_idx = Nx.create Nx.int64 [| size |] (Array.map Int64.of_int dual_perm)
  let dual_vec = Nx.create dtype [| size |] dual_sign
  let undual_vec = Nx.create dtype [| size |] undual_sign

  let grade_mask =
    Array.init (n + 1) (fun k ->
      Nx.create
        dtype
        [| size |]
        (Array.init size (fun i -> if blade_grade i = k then 1. else 0.)))

  let even_mask =
    Nx.create
      dtype
      [| size |]
      (Array.init size (fun i -> if blade_grade i land 1 = 0 then 1. else 0.))

  let odd_mask =
    Nx.create
      dtype
      [| size |]
      (Array.init size (fun i -> if blade_grade i land 1 = 1 then 1. else 0.))

  let rank a = Array.length (Nx.shape a)

  let batch_shape a =
    let s = Nx.shape a in
    if Array.length s = 0 then invalid "batch_shape" "a multivector needs a blade axis";
    Array.sub s 0 (Array.length s - 1)

  let last_axis a =
    let r = rank a in
    if r = 0 then invalid "last_axis" "a multivector needs a blade axis";
    r - 1

  let take_last idx a = Nx.take ~axis:(last_axis a) ~indices:idx a

  let scalar x =
    let a = Array.make size 0. in
    a.(0) <- x;
    Nx.create dtype [| size |] a

  let one = scalar 1.
  let zero = scalar 0.

  let basis b =
    let i = Blade.to_int b in
    if i < 0 || i >= size then invalid "basis" "blade index out of range";
    let a = Array.make size 0. in
    a.(i) <- 1.;
    Nx.create dtype [| size |] a

  let e i =
    if i < 1 || i > n then invalid "e" "basis vector index out of range";
    basis (Blade.of_basis i)

  let pseudoscalar = basis (Blade.of_int full)

  let of_array a =
    if Array.length a <> size then invalid "of_array" "wrong number of coefficients";
    Nx.create dtype [| size |] a

  let to_array a =
    if batch_shape a <> [||] then invalid "to_array" "expected an unbatched multivector";
    Nx.to_array a

  let of_nx x =
    let s = Nx.shape x in
    if Array.length s = 0 || s.(Array.length s - 1) <> size
    then invalid "of_nx" "the last axis must be the blade axis";
    x

  let to_nx x = x
  let shape a = Nx.shape a
  let coeff a b = take_last (Nx.scalar Nx.int64 (Int64.of_int (Blade.to_int b))) a

  let with_coeff a b c =
    let batch = batch_shape a in
    let mask = Array.make size false in
    mask.(Blade.to_int b) <- true;
    let mask = Nx.create Nx.bool [| size |] mask in
    let c = Nx.broadcast_to batch c in
    let c = Nx.reshape (Array.append batch [| 1 |]) c in
    Nx.where mask c a

  let grade k a = if k < 0 || k > n then zero else Nx.mul a grade_mask.(k)
  let even a = Nx.mul a even_mask
  let odd a = Nx.mul a odd_mask
  let reverse a = Nx.mul a rev_vec
  let grade_involution a = Nx.mul a gi_vec
  let conjugate a = Nx.mul a conj_vec

  let contract tbl a b =
    let a, b =
      match Nx.broadcast_arrays [ a; b ] with
      | [ a; b ] -> a, b
      | _ -> assert false
    in
    let batch = batch_shape a in
    let bsz = product_shape batch in
    let a2 = Nx.reshape [| bsz; size |] a in
    let b2 = Nx.reshape [| bsz; size |] b in
    let out = Nx.einsum "bi,bj,kij->bk" [| a2; b2; tbl |] in
    Nx.reshape (Array.append batch [| size |]) out

  let gp a b = contract gp_tbl a b
  let op a b = contract op_tbl a b
  let inner a b = contract inner_tbl a b
  let commutator a b = Nx.div (Nx.sub (gp a b) (gp b a)) (Nx.scalar dtype 2.)

  let dual a =
    if pseudoscalar_square = 0 then invalid "dual" "the signature is degenerate";
    take_last dual_idx (Nx.mul a dual_vec)

  let undual a =
    if pseudoscalar_square = 0 then invalid "undual" "the signature is degenerate";
    take_last dual_idx (Nx.mul a undual_vec)

  let complement a = take_last complement_idx (Nx.mul a complement_vec)
  let regressive a b = complement (op (complement a) (complement b))
  let scalar_prod a b = coeff (gp (reverse a) b) (Blade.of_int 0)
  let norm2 a = coeff (gp (reverse a) a) (Blade.of_int 0)
  let norm a = Nx.sqrt (Nx.abs (norm2 a))
  let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x
  let normalize a = Nx.div a (expand (batch_shape a) (norm a))
  let inverse a = Nx.div (reverse a) (expand (batch_shape a) (norm2 a))
  let sandwich v x = gp (gp v x) (inverse v)

  let adjoint v =
    let cols = Array.init size (fun k -> sandwich v (basis (Blade.of_int k))) in
    Nx.stack ~axis:(-1) (Array.to_list cols)

  let pp ppf a =
    let sh = Nx.shape a in
    if Array.length sh = 1 && sh.(0) = size
    then (
      let arr = Nx.to_array a in
      Format.fprintf ppf "@[<hov 1>";
      let first = ref true in
      for i = 0 to size - 1 do
        if arr.(i) <> 0.
        then (
          if not !first then Format.fprintf ppf " +@ ";
          first := false;
          if i = 0
          then Format.fprintf ppf "%g" arr.(i)
          else Format.fprintf ppf "%g*%s" arr.(i) (Blade.to_string (Blade.of_int i)))
      done;
      if !first then Format.fprintf ppf "0";
      Format.fprintf ppf "@]")
    else
      Format.fprintf
        ppf
        "<multivector %s>"
        (String.concat "x" (Array.to_list (Array.map string_of_int sh)))
end
