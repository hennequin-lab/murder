module type VERSOR = sig
  (** The scalar kind of the underlying algebra. *)
  type kind

  (** The type for multivectors of the underlying algebra. *)
  type t

  (** [exp b] is the exponential of the bivector [b], the versor
      [sum_(k >= 0) b^k / k!]. For a simple bivector this is the closed form of
      the notes, §1.6; for the [SE(3)] subalgebra it is the motor exponential.

      Raises [Invalid_argument] if [b] is outside the subset supported by the
      current implementation. *)
  val exp : t -> t

  (** [log v] is a bivector whose exponential is the versor [v], with the
      branch chosen near the identity. It is the left inverse of {!exp} on the
      supported subset.

      Raises [Invalid_argument] if [v] is not a versor or is outside the
      supported subset. *)
  val log : t -> t

  (** [exp_simple b] is the exponential of a simple bivector, using the closed
      form

      {[
        exp b =
          cos theta + (sin theta / theta) b,   if b^2 = -theta^2 < 0,
          cosh theta + (sinh theta / theta) b, if b^2 = +theta^2 > 0,
          1 + b,                               if b^2 = 0,
      ]}

      with the limiting value at [theta = 0]. It is always defined for a
      simple bivector, including the null generators of translations,
      transversions and shears. *)
  val exp_simple : t -> t

  (** [slerp v0 v1 alpha] interpolates between the versors [v0] and [v1] along
      the group exponential:

      {[
       slerp v0 v1 alpha = exp (alpha * log (v0^(-1) v1)) v0.
      ]}

      [alpha] is a tensor of shape [batch_shape] with values in [[0];[1]].
      [alpha = 0] gives [v0] and [alpha = 1] gives [v1]. The interpolation is
      the screw linear interpolation (ScLERP) of rigid motions when the versors
      are motors. *)
  val slerp : t -> t -> (float, kind) Nx.t -> t
end

(* --- implementation ------------------------------------------------------- *)

module Make (A : Algebra.ALGEBRA) = struct
  type kind = A.kind
  type t = A.t

  let expand batch x = Nx.reshape (Array.append batch [| 1 |]) x

  let exp_simple b =
    let batch = A.batch_shape b in
    let b2 = A.coeff (A.gp b b) (Blade.of_int 0) in
    let theta = Nx.sqrt (Nx.abs b2) in
    let small = Nx.less_s theta 1e-6 in
    let theta_safe = Nx.where small (Nx.ones_like theta) theta in
    let neg = Nx.less_s b2 0. in
    let pos = Nx.greater_s b2 0. in
    let one = Nx.ones_like theta in
    let c_neg = Nx.cos theta
    and c_pos = Nx.cosh theta in
    let c0 = Nx.where neg c_neg (Nx.where pos c_pos one) in
    let c = Nx.where small one c0 in
    let s_neg = Nx.div (Nx.sin theta) theta_safe in
    let s_pos = Nx.div (Nx.sinh theta) theta_safe in
    let f0 = Nx.where neg s_neg (Nx.where pos s_pos one) in
    let f = Nx.where small one f0 in
    Nx.add (Nx.mul (expand batch c) A.one) (Nx.mul (expand batch f) b)

  let exp b =
    (* Scaling and squaring with a fixed 12-term Taylor series. This handles a
       general bivector, including the non-simple case. *)
    let factor = 1. /. 256. in
    let bs = Nx.mul_s b factor in
    let term = ref A.one
    and sum = ref A.one in
    for k = 1 to 12 do
      term := Nx.div_s (A.gp !term bs) (float_of_int k);
      sum := Nx.add !sum !term
    done;
    let acc = ref !sum in
    for _ = 1 to 8 do
      acc := A.gp !acc !acc
    done;
    !acc

  let log v =
    let c = A.coeff v (Blade.of_int 0) in
    let b = A.grade 2 v in
    let b2 = A.coeff (A.gp b b) (Blade.of_int 0) in
    let nb = Nx.sqrt (Nx.abs b2) in
    let small = Nx.less_s nb 1e-6 in
    let theta_rot = Nx.atan2 nb c in
    let theta_hyp = Nx.atanh (Nx.div nb c) in
    let theta =
      Nx.where
        (Nx.less_s b2 0.)
        theta_rot
        (Nx.where (Nx.greater_s b2 0.) theta_hyp (Nx.zeros_like b2))
    in
    let nb_safe = Nx.where small (Nx.ones_like nb) nb in
    let scale = Nx.where small (Nx.ones_like nb) (Nx.div theta nb_safe) in
    Nx.mul b (expand (A.batch_shape b) scale)

  let slerp v0 v1 alpha =
    let rel = A.gp (A.inverse v0) v1 in
    let tw = log rel in
    let abatch = Nx.shape alpha in
    A.gp (exp (Nx.mul tw (Nx.reshape (Array.append abatch [| 1 |]) alpha))) v0
end
