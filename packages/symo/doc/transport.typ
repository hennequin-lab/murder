// symo — the mathematics of the block coordinates, for humans.
// Compile from packages/symo/:  typst compile transport.typ
//
// =====================================================================
//  Blocks, test vectors and transport
//  The mathematics behind the second-order orbit average's block
//  coordinates, and why a function of the average costs a very small
//  eigendecomposition.
//
//  Companion to the symo package (murder/packages/symo). The paper
//  (Artemev et al., ICML 2026) states that an analytic function of an
//  orbit average lives in the same algebra and can be computed in a
//  reduced space; this note supplies the construction, the exactness
//  proof, and the size-independence statement ("transport").
// =====================================================================

#import "@preview/cetz:0.3.4": canvas, draw

// ------------------------------------------------------------------ style

#let accent = rgb("#a8432a")
#let accent-2 = rgb("#1d6070")
#let faint = rgb("#6f6f78")
#let tint = rgb("#f4f1ea")
#let mint = rgb("#e9f0f1")

#set page(
  paper: "a4",
  margin: (x: 2.3cm, y: 2.2cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 7.6pt, fill: faint, font: "Libertinus Serif")
      grid(
        columns: (1fr, auto),
        [blocks, test vectors and transport],
        [symo / murder],
      )
      v(-2pt)
      line(length: 100%, stroke: 0.4pt + faint.transparentize(60%))
    }
  },
)

#set text(font: "Libertinus Serif", size: 10.4pt, fill: rgb("#1a1a1e"))
#set math.equation(numbering: "(1)")
#set par(justify: true, leading: 0.68em, spacing: 1.08em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(1.05em)
  block(sticky: true, below: 0.1em, text(size: 14.6pt, fill: accent, weight: "bold", tracking: -0.2pt, it))
  v(-0.3em)
  line(length: 100%, stroke: 0.7pt + accent.transparentize(55%))
  v(0.45em)
}
#show heading.where(level: 2): it => {
  v(0.7em)
  block(sticky: true, below: 0.35em, text(size: 11.3pt, fill: accent-2, weight: "bold", it))
}
#show math.equation.where(block: true): set block(above: 0.85em, below: 0.85em)
#show figure: set block(above: 1.2em, below: 1.2em)
#show figure: set align(center)
#show figure.caption: it => block(
  width: 92%,
  text(size: 8.8pt, fill: faint, it),
)
#show raw.where(block: true): it => block(
  width: 100%,
  fill: tint,
  inset: 8pt,
  radius: 1.5pt,
  text(size: 8.6pt, font: "DejaVu Sans Mono", it),
)
#show footnote: set text(size: 8.4pt, fill: faint)

// ------------------------------------------------------------------ envs

#let env(kind, num, name: none, body) = block(
  width: 100%,
  above: 0.95em,
  below: 0.95em,
  inset: (left: 9pt, top: 4pt, bottom: 4pt, right: 3pt),
  stroke: (left: 1.5pt + accent.transparentize(50%)),
  {
    text(fill: accent, weight: "bold")[#kind #num]
    if name != none { text(fill: accent, weight: "bold")[ (#name)] }
    [. ]
    body
  },
)

#let defn(num, name: none, body) = env("Definition", num, name: name, body)
#let thm(num, name: none, body) = env("Theorem", num, name: name, body)
#let prop(num, name: none, body) = env("Proposition", num, name: name, body)
#let cor(num, name: none, body) = env("Corollary", num, name: name, body)
#let lem(num, name: none, body) = env("Lemma", num, name: name, body)

#let rem(num, body) = block(
  width: 100%,
  above: 0.8em,
  below: 0.9em,
  inset: (left: 9pt, top: 2pt, bottom: 2pt, right: 3pt),
  stroke: (left: 1.5pt + faint.transparentize(55%)),
  {
    text(fill: faint, weight: "bold")[Remark #num.]
    [ ]
    text(fill: rgb("#3a3a42"), body)
  },
)

#let proof(body) = block(
  width: 100%,
  above: 0.2em,
  below: 0.95em,
  {
    emph[Proof.] + [ ] + body + [ #h(1fr) $square$]
  },
)

#let small(body) = text(size: 9pt, body)

// ------------------------------------------------------------------ title

#align(center)[
  #v(0.4cm)
  #text(size: 19pt, weight: "bold", fill: rgb("#1a1a1e"))[
    Blocks, test vectors and transport
  ]
  #v(0.15cm)
  #text(size: 11.6pt, fill: accent-2)[
    the algebra that turns an orbit average into a few small matrices
  ]
  #v(0.55cm)
]

#block(
  width: 100%,
  inset: (x: 10pt, y: 8pt),
  fill: mint.transparentize(45%),
  radius: 2pt,
  text(size: 9.6pt)[
    *What this is.* A mathematical companion to the `symo` package
    (`murder/packages/symo`): a self-contained account of the objects the code
    calls *kinds*, *test vectors*, *blocks* and *transport*, and of the
    functional calculus that runs on them. The paper
    (Artemev et al., ICML 2026) states the
    conclusion — an analytic function of an orbit average lives in the same
    algebra and can be computed "in the reduced space" — but not the
    construction, not the proof that it is exact, and not what "the same
    operator at a smaller size" means. Those three things are what this note
    is for. Each section names its counterpart in the library.
  ],
)

#v(0.15cm)

= Setting

== One leaf, one pair

Fix a group id, and let $V = RR^n$ carry the permutation representation of
$cal(G) = S_n$: the group relabels $n$ units, and permuting the units acts on
$V$ by a permutation matrix, hence orthogonally. Real networks carry several
such groups (one per tied axis) and several leaves (one per parameter
tensor); the general statements are per group and per leaf pair, and §3.4 puts
them together. A *leaf pair* $(i, j)$ has

$ W_L = V^(⊗k_L) ⊗ F_L, quad W_R = V^(⊗k_R) ⊗ F_R, $

where $k_L, k_R$ are the numbers of axes of the two leaves that this group
permutes (the *permuted* axes) and $F_L, F_R$ are the products of the
remaining, *free* axes. The tensor factors can be reordered because the group
does not see $F$: with $sigma in S_n$ and $bold(P)_sigma$ its permutation
matrix,

$ rho_(L)(sigma) = bold(P)_sigma^(⊗k_L) ⊗ bold(1)_(F_L), quad quad
  rho_(R)(sigma) = bold(P)_sigma^(⊗k_R) ⊗ bold(1)_(F_R). $ <eq-rho>

Both are orthogonal, $rho_(L)(sigma)^top rho_(L)(sigma) = 1$. The operators of
interest — the second-order orbit average of a gradient moment — are exactly
the intertwiners of the pair,

$ cal(C) = { bold(S) : W_R -> W_L quad "such that" quad
  bold(S) rho_(L)(sigma) = rho_(R)(sigma) bold(S) quad forall sigma in cal(G) }. $ <eq-commutant>

$cal(C)$ is a linear subspace of the $D_L times D_R$ matrices, closed under
composition in the square case; it is the *commutant algebra* of the pair, and
the whole subject is the size of this space. Write $D = D_L = D_R = n^k f$
for a diagonal pair.

== The orbit averages

For vectors $bold(u) in W_L$, $bold(v) in W_R$ the second-order orbit average
is

$ cal(R)_2(bold(u), bold(v)) = EE_(sigma in cal(G))[
  (rho_(L)(sigma) bold(u)) (rho_(R)(sigma) bold(v))^top ], $ <eq-r2>

an average over the group of outer products; for a diagonal pair and
$bold(u) = bold(v) = bold(g)$ it is the average of $bold(g) bold(g)^top$ over
the orbit of $bold(g)$. (The first-order average is the same object with one
side trivial, $cal(R)_1(bold(v)) = EE_sigma[rho_(L)(sigma) bold(v)]$; nothing
below depends on it, and @eq-r2 is what the optimizer uses.)

Note the shape of @eq-r2: the *data* enters linearly and the *averaging* is a
linear operator on matrices. Everything in §2 follows from that.

== What the paper leaves out

The paper's estimator is built on three claims, all true, none proved there:

+ the orbit average is an invariant operator, and any analytic function of it
  is invariant too, so it has the same short description;
+ that short description is a list of small matrices ("blocks"), one per
  irreducible representation appearing in $W$;
+ functions act on the blocks, so the expensive part of the computation lives
  in a space whose dimension does not grow with $n$.

What is missing is the *dictionary* between the two descriptions: how a list of
scalars produced by an einsum becomes a set of matrices, why that map is an
isomorphism rather than a fit, and what happens to it when $n$ changes. That is
§4–§6. With it, §7 becomes three lines of algebra.

= The orbit average is a projection

#defn("2.1", name: "twirl")[
  The *twirl* of a matrix $bold(S) : W_R -> W_L$ is
  $ cal(T)(bold(S)) = EE_(sigma in cal(G))[ rho_(L)(sigma) bold(S) rho_(R)(sigma)^top ]. $
]

#prop("2.2")[
  $cal(T)$ is the orthogonal projection of $RR^(D_L times D_R)$, with the
  Frobenius inner product, onto the commutant $cal(C)$ of @eq-commutant.
]

#proof[
  Three facts. *(i) The image is invariant.* For $tau in cal(G)$,
  $cal(T)(bold(S)) rho_(R)(tau) = EE_sigma[rho_(L)(sigma) bold(S) rho_(R)(sigma^(-1) tau)]$,
  and reindexing $sigma' = sigma^(-1) tau$ (the average is right-invariant)
  gives $EE_(sigma')[rho_(L)(tau sigma') bold(S) rho_(R)(sigma')] = rho_(L)(tau) cal(T)(bold(S))$.
  So $cal(T)(bold(S)) in cal(C)$.
  *(ii) It fixes $cal(C)$.* If $bold(S) in cal(C)$ then
  $rho_(L)(sigma) bold(S) rho_(R)(sigma)^top = bold(S) rho_(R)(sigma) rho_(R)(sigma)^top = bold(S)$
  by orthogonality, so the average is $bold(S)$.
  *(iii) It is self-adjoint.* Using $⟨ bold(A), bold(B) ⟩ = tr(bold(A)^top bold(B))$
  and cyclicity,
  $⟨ bold(A), cal(T)(bold(B)) ⟩ = EE_sigma tr((rho_(L)(sigma)^top bold(A) rho_(R)(sigma))^top bold(B)) = ⟨ cal(T)(bold(A)), bold(B) ⟩$.
  A self-adjoint idempotent is an orthogonal projection, and (i)+(ii) identify
  its image.
]

#cor("2.3")[
  $cal(R)_2(bold(u), bold(v))$ is the orthogonal projection of the rank-one
  matrix $bold(u) bold(v)^top$ onto $cal(C)$. In particular
  $cal(R)_2(bold(u), bold(u))$ is positive semidefinite, and
  $cal(R)_2$ is idempotent: averaging twice changes nothing.
]

#proof[
  $cal(R)_2(bold(u), bold(v)) = cal(T)(bold(u) bold(v)^top)$ is the definition
  @eq-r2 read through Definition 2.1. Idempotence is Proposition 2.2. For positivity, the
  summands $(rho_(L)(sigma) bold(u))(rho_(L)(sigma) bold(u))^top$ are positive semidefinite and
  so is their average; a projection of a positive semidefinite matrix is
  positive semidefinite here because the projector is itself a twirl, which is
  not true of projections in general.
]

#rem("2.4")[
  Two consequences are used constantly below. First, $cal(R)_2$ is an
  *orthogonal projection* and not a regression: the best invariant
  approximation to $bold(g) bold(g)^top$ in Frobenius norm is its projection,
  and this is what makes the estimator of §4 exact rather than least-squares
  in the statistical sense. Second, $cal(C)$ is a linear space, so every
  average the optimizer takes — over output elements, over steps (the EMA),
  and the damping — stays inside it. The metric the solver diagonalizes in §7
  is therefore an element of $cal(C)$ exactly, up to float32 roundoff, which
  is the hypothesis the functional calculus needs.
]

#rem("2.5")[
  The invariance of $cal(C)$ under the twirl uses only that $rho$ is a
  representation by *orthogonal* matrices. Weight-space symmetry groups are
  subgroups of permutations (and, in the signed variant, signed permutations),
  so this is free. For a non-orthogonal group action the average of
  @eq-r2 would have to be replaced by a projection with a non-trivial weight,
  and the "damping is a free scalar" property of §7 would fail.
]

= Wedderburn coordinates

The point of the whole construction is that $cal(C)$ is *small and rigid*:
every element of it acts as the identity on most of the space. This section
states that precisely, and §4–§5 turn it into an algorithm.

== The isotypic decomposition

Let $W = V^(⊗k) ⊗ F$ be one side (so $k$ permuted axes and free space $F$).
Since $cal(G)$ acts trivially on $F$ and by $bold(P)_sigma^(⊗k)$ on
$V^(⊗k)$, the representation theory of $W$ is that of $V^(⊗k)$ tensored with a
trivial factor. The isotypic decomposition of $V^(⊗k)$ is

$ V^(⊗k) ≅ ⊕_lambda cal(S)_lambda ⊗ cal(M)_lambda, $ <eq-isotypic>

where $lambda$ runs over the *kinds* (partitions of at most $k$ parts, i.e.
irreducible representations of $S_n$ that appear in $k$-th tensor powers of the
permutation module), $cal(S)_lambda$ is the carrier of the kind ($dim =
d_lambda$) and $cal(M)_lambda$ is the multiplicity space ($dim = m_lambda$).
For the group $cal(G) = S_n$ acting on $V = RR^n$ and $k <= 2$, the list is
short:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    stroke: none,
    inset: (x: 8pt, y: 3.6pt),
    table.hline(stroke: 0.8pt),
    [*k*], [*kind $lambda$*], [*name*], [*$d_lambda$*], [*$m_lambda$*],
    table.hline(stroke: 0.4pt),
    [$0$], [$emptyset$], [trivial], [$1$], [$1$],
    table.hline(stroke: 0.3pt + faint.transparentize(60%)),
    [$1$], [$emptyset$], [trivial], [$1$], [$1$],
    [], [$[1]$], [standard $W$], [$n-1$], [$1$],
    table.hline(stroke: 0.3pt + faint.transparentize(60%)),
    [$2$], [$emptyset$], [trivial], [$1$], [$2$],
    [], [$[1]$], [standard $W$], [$n-1$], [$3$],
    [], [$[2]$], [symmetric square], [$n(n-3) slash 2$], [$1$],
    [], [$[1,1]$], [antisymmetric], [$(n-1)(n-2) slash 2$], [$1$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Kinds, carrier dimensions and multiplicities for $k <= 2$ in the stable
    range ($n >= 2k$, §3.3). The $k = 0$ row is a side on which this group does
    not act at all (a bias paired with a weight, say). Two identities hold in
    every row:
    $sum_lambda d_lambda m_lambda = n^k$ (the dimension of the space) and
    $sum_lambda m_lambda^2 = dim cal(C)$ (the dimension of the algebra). For
    $k = 2$: $2 + 3(n-1) + n(n-3) slash 2 + (n-1)(n-2) slash 2 = n^2$ and
    $4 + 9 + 1 + 1 = 15$.
  ],
) <fig-kinds>

The two $k = 1$ kinds are the familiar ones: $cal(S)_emptyset$ spans the all-ones
direction and $cal(S)_([1]) $ the zero-sum subspace, with projectors
$Pi = bold(J) slash n$ and $Pi_perp = bold(1) - Pi$.

== The operator is the identity on the carrier

#thm("3.1", name: "Wedderburn decomposition")[
  Let $cal(G)$ act as in @eq-rho and let $bold(S) : W_R -> W_L$ commute with
  it. Then for each kind $lambda$ there is a matrix
  $bold(B)_lambda : cal(M)_lambda^L ⊗ F_L -> cal(M)_lambda^R ⊗ F_R$ such that

  $ bold(S) = ⊕_lambda (bold(1)_(cal(S)_lambda) ⊗ bold(B)_lambda). $ <eq-wedderburn>

  Conversely every operator of this form commutes with $cal(G)$. The map
  $bold(B)_bullet |-> bold(S)$ is linear, injective, and its image is
  $cal(C)$; with $k_L = k_R$ and free spaces of equal size it is an
  isomorphism of algebras.
]

#proof[
  That the right-hand side intertwines is immediate: $rho_(L)(sigma)$ acts as
  the identity on $cal(M)^L ⊗ F_L$, so the two sides of the commutation
  relation differ only by $bold(P)_sigma^(⊗k)$ acting on the carrier, which
  is the same on both sides. For the converse, $bold(S)$ intertwines, so it
  maps the $mu$-isotypic component of $W_R$ into the $mu$-isotypic component of
  $W_L$ (Schur's lemma: the composite with the projection onto another
  isotypic component is an intertwiner between non-isomorphic irreducibles,
  hence zero), and on one isotypic component an intertwiner is
  $bold(1) ⊗ bold(B)_lambda$ again by Schur. Injectivity: if
  $bold(1) ⊗ bold(B) = 0$ then $bold(B) = 0$. Both spaces have dimension
  $sum_lambda m_lambda^L m_lambda^R f_L f_R$, which is finite and equal, so the
  map is onto.
]

The dimension count in the proof is worth stating on its own, because it is the
"how many numbers does this operator contain" answer:

$ dim cal(C) = sum_lambda m_lambda^L m_lambda^R f_L f_R $ <eq-dimC>

For a diagonal pair with $k = 1$ and free space $F$ this is $2 f^2$; for
$k = 2$ it is $15 f^2$; for the *symmetric* operators of a diagonal pair it is
$sum_lambda m_lambda (m_lambda + 1) slash 2 dot f^2$, i.e. $3 f^2$ and $11 f^2$
respectively. The code's count of basis components (§4) is @eq-dimC.

#rem("3.2")[
  Read @eq-wedderburn twice. On the big part — the carrier, of dimension
  $d_lambda$, which is the part that grows with $n$ — the operator is the
  *identity*: it carries no information and cannot. All of its content sits in
  $bold(B)_lambda$, whose size is the multiplicity (at most $3$ for $k <= 2$)
  times the free dimension. This is why the description is short, and it is
  the only reason a later function of $bold(S)$ can be computed cheaply: a
  function of $bold(1) ⊗ bold(B)$ is $bold(1) ⊗ f(bold(B))$, and the identity
  never needs to be multiplied out.
]

#rem("3.3")[
  For a *mixed* pair ($k_L != k_R$, e.g. a weight matrix paired with a bias)
  the block $bold(B)_lambda$ is rectangular, $m_lambda^L f_L$ by
  $m_lambda^R f_R$, and only the kinds that occur on *both* sides contribute:
  the common kinds. The library's `Blocks` module is written for this general
  case; the square case is the one with eigenvalues, and the one the optimizer
  uses.
]

== The stable range

#prop("3.4")[
  The multiplicities $m_lambda$ of @fig-kinds, and the decomposition
  @eq-wedderburn as an isomorphism, hold for $n >= 2k$, where $k$ is the
  number of axes the group permutes on either side. Below that threshold the
  list of kinds is different (kinds coincide or disappear), the block map is
  not invertible, and no transport exists.
]

#proof[
  The algebra of intertwiners of $V^(⊗k)$ is the partition algebra $P_k(n)$,
  whose basis is the set partitions of $2k$ points, so its dimension in the
  stable range is $B_(2k)$; by Schur–Wedderburn that dimension is also
  $sum_lambda m_lambda^2$. $P_k(n)$ is semisimple exactly for $n >= 2k$, and
  the failure below that is visible as linear relations among the diagram
  tensors — for $k = 2$, 14 remain independent at $n = 3$ and 8 at $n = 2$,
  against $15$ required. The code checks
  $n >= 2k$ per side (`Kind.check_stable`) and refuses the specification
  otherwise, because every statement in §5–§6 is a statement about a
  semisimple algebra.
]

#rem("3.5")[
  The threshold used by the library, $n >= 2k$ for *each* side, is slightly
  stronger than the plain semisimplicity condition $n >= k_L + k_R$ of the
  pair's partition algebra: it is the condition under which each side's own
  kind list and multiplicities are already the stable ones, so that the two
  canonical bases of §5 can be built from the same recipe. Being conservative
  here costs nothing in practice (a permuted axis with $n < 6$ has no
  curvature worth estimating) and buys a uniform construction.
]

== Several groups, several leaves

Two indexings complete the picture. With several groups, a *kind* is a tuple
of partitions, one per group; the carriers tensor and the multiplicities
multiply, so $cal(S)_lambda = ⊗_g cal(S)_(lambda^((g))) $ and
$m_lambda = product_g m_(lambda^((g)))$. With several leaves the parameter space
is a direct sum $W = ⊕_i W_i$ and an operator on it is a matrix of operators
$bold(S)_(i j) : W_j -> W_i$; the commutant is

$ cal(C)(⊕_i W_i) = ⊕_lambda op("End")( ⊕_i (cal(M)_lambda^((i)) ⊗ F_i) ), $ <eq-tree>

because a kind can only talk to itself. So for each global kind $lambda$ there
is *one* square block whose index set is pairs (leaf, copy, free index), of
size $sum_i m_lambda^((i)) f_i$ — the object `blocks_of_factors` assembles out
of the per-pair blocks, and the object §7 diagonalizes. The number of
eigendecompositions the optimizer runs is therefore the number of global
kinds, never the number of parameters.

= Factors: coordinates in a size-dependent basis

The first step of the pipeline produces scalars. This section says which
scalars, and in what sense they are coordinates.

== Terms and components

#defn("4.1", name: "term")[
  A *term* is a partial partition $P$ of the $k_L + k_R$ index occurrences of
  one group (the left and right axes the group permutes) into blocks of size
  at least two, together with the list of free indices. Its tensor is the
  product of Kronecker deltas over the blocks,

  $ bold(T)_(P)[i, j] = product_(B in P) product_((x,y) in B) delta_(i_x, i_y), $

  read as 1 when all occurrences in each block coincide and as a constant
  (all-ones) along the axes it does not mention, scaled by
  $1 slash sqrt(product_(x in.not P) dim x)$ so that the tensors are orthonormal
  for the unnormalized sum inner product. A *component* is a term, or a term
  summed with its transpose when the pair is diagonal and the estimator is
  required to be symmetric. The *factors* are the coefficients
  $c_1, dots, c_d$ of
  $ bold(S) = sum_k c_k bold(T)_k . $
]

The terms are the "equality patterns" of a matrix in the commutant: an entry
can depend only on which pairs of indices are equal, and a partial partition
records exactly that. The unmentioned axes are unconstrained (constant), which
is why the tensor is sparse and why applying it is an einsum.

#prop("4.2")[
  In the stable range the terms span $cal(C)$, and their number for one group
  is the Bell number $B_(k_L + k_R)$,

  $ d = sum_lambda m_lambda^L m_lambda^R = B_(k_L + k_R), $

  which for $k = 1$ gives $2$ and for $k = 2$ gives $15$. A diagonal pair with
  $f = 1$ that is compiled symmetrically has fewer components, $sum_lambda
  m_lambda (m_lambda + 1) slash 2$ — $3$ and $11$ — because a term and its
  transpose are bundled; with free axes that bundle can be incomplete (§8), and
  the compiler then keeps the full basis. Either way each component carries an
  $f_L times f_R$ tensor, so the number of *numbers* in a factor list is
  @eq-dimC.
]

#proof[
  A set partition of the $k_L + k_R$ occurrences is sent to the partial
  partition obtained by deleting its singleton blocks; this is a bijection
  onto the partial partitions with blocks of size at least two, so there are
  $B_(k_L+k_R)$ terms. They are linearly independent in the stable range (the
  design matrix below is positive definite, which the compiler verifies by
  attempting a Cholesky factorization at compile time), and their number times
  the $f_L f_R$ entries each component carries is the dimension of $cal(C)$,
  @eq-dimC.
]

The coefficient extraction is the least-squares problem

$ bold(G) bold(c) = bold(b), quad quad
  bold(G)_(k l) = ⟨ bold(T)_k, bold(T)_l ⟩, quad
  bold(b)_k = ⟨ bold(T)_k, bold(X) ⟩, $ <eq-normal>

with $X = bold(u) bold(v)^top$ (or a dense surrogate of it). Because
$cal(T)$ is an orthogonal projection onto $op("span") {bold(T)_k} = cal(C)$,

#prop("4.3", name: "the estimator is the coordinates of the projection")[
  The solution of @eq-normal is the coefficient vector of
  $cal(R)_2(bold(u), bold(v))$; if $X$ is already in $cal(C)$ the reconstruction
  is exact, and otherwise it is the unique best invariant approximation in
  Frobenius norm.
]

#proof[
  The projection $(bold(P)_(cal(C)) X)$ is characterized by
  $⟨ bold(T)_l, X - (bold(P)_(cal(C)) X) ⟩ = 0$ for every $l$, which
  is @eq-normal with $(bold(P)_(cal(C)) X) = sum_k c_k bold(T)_k$. Uniqueness is
  positive definiteness of $bold(G)$.
]

#rem("4.4")[
  The design matrix $bold(G)$ does not depend on the data — only on the
  symmetry specification and the dimensions — so it is factored once, at
  compile time (`Compiler.build_estimator`: Cholesky, with a jittered retry
  and an SVD pseudo-inverse if the basis is degenerate), and the per-step work
  is two triangular solves. $bold(G)$ is not the identity: the delta for a
  pair of indices and the all-ones pattern overlap, and more generally
  $bold(G)_(P Q) = product_(B in P bowtie.big Q) dim(B)$ is a weighted zeta matrix of
  the partition lattice. That is also why the basis is not orthogonal and why
  the factorization is worth doing properly once.
]

== The factors are coordinates, not matrices

The distinction that the rest of this note turns on is the following. A factor
list is a coordinate vector in a basis of $cal(C)$ that depends on $n$; a block
is the operator's matrix in a basis that does not. Writing one for the other is
the classic error, and the cleanest place to see the difference is $k = 1$.

Take a diagonal pair with one permuted axis and free space of dimension $f$,
so $bold(S)$ is an $n f times n f$ matrix. The two terms are

$ bold(T)_(emptyset) = Pi ⊗ dot, quad quad bold(T)_([1]) = bold(1)_n ⊗ dot,
  quad quad Pi = bold(J) slash n, $

and @eq-wedderburn says $bold(S) = Pi ⊗ bold(B)_(emptyset) + Pi_perp ⊗ bold(B)_([1])
$ with two $f times f$ blocks. Comparing the two expressions gives the
dictionary

$ bold(B)_(emptyset) = bold(F)_(emptyset) + bold(F)_([1]), quad quad
  bold(B)_([1]) = bold(F)_([1]), $ <eq-k1>

i.e. a *linear* map from two $f times f$ coordinates to two $f times f$
blocks which is not the identity, not block-diagonal, and not a function of the
coordinates taken one at a time. (Verified numerically in `test_blocks` for
$n = 5$, $f = 3$: the reconstruction from @eq-k1 agrees with the dense
operator to float32 roundoff.)

In the unnormalized $delta slash 1$ basis the same statement reads
$bold(B)_(emptyset) = bold(F)_delta + n bold(G)_1$: the coefficient of the all-ones
pattern is multiplied by $n$ on its way to the block. That is a genuine
size dependence and it is the historical source of the phrase "the factors
depend on the width". In the library's normalized basis the $1 slash n$ of the
normalization absorbs it, so for $k = 1$ the coordinates happen to be
size-free as well; for $k = 2$ they are not (§6.2). Either way, the blocks are
the object with a size-free meaning, and the coordinates are an encoding.

= Test vectors

We now have the two descriptions and know they are linearly related. The
remaining question is the practical one: given a factor list, *how* do we
compute the blocks, without forming the $D times D$ operator and without
touching $D$ basis vectors? The answer is a small table of inner products
built from a handful of canonical vectors, and it is the piece the paper does
not describe.

== The problem

$cal(R)_2$ hands us a coordinate vector $bold(c) in RR^d$ with $d$ small. We
want the matrices $bold(B)_lambda$. Applying the operator to the $D$ standard
basis vectors and reading the result entry by entry costs $D$ applications, a
$D^2$ computation, which is exactly what the compression was supposed to
avoid. Symmetry says far fewer vectors suffice — and says which ones: any
vector that lies in a single isotypic component.

#defn("5.1", name: "test vectors")[
  A *test vector* for the kind $lambda$ and copy $a$ is the image of a fixed
  *internal* vector of the carrier under the copy's canonical embedding. That
  is: fix, once and for all, an identification
  $cal(S)_lambda^L ≅ cal(S)_lambda^R ≅ cal(S)_lambda$ and a unit vector
  $bold(e)_lambda in cal(S)_lambda$, and set

  $ bold(z)_(lambda a)^L = bold(e)_lambda ⊗ bold(m)_a^L, quad quad
    bold(z)_(lambda b)^R = bold(e)_lambda ⊗ bold(m)_b^R, $ <eq-testvectors>

  where $bold(m)_a^L$, $bold(m)_b^R$ run over bases of the multiplicity
  spaces. Two conditions are part of the definition, not of the theorem:
  *the same* internal vector is used on the left and on the right, and the
  same one for every copy of a kind. With free axes the vectors are tensored
  with a basis of $F$.
]

The code's test vectors are exactly of this form, with the internal vector
spelled out for each kind — the all-ones vector for the trivial kind, the
Helmert vector $bold(h) = (1, -1, 0, dots, 0)^top slash sqrt(2)$ for the
standard kind, a four-cycle matrix for the symmetric square, and an
antisymmetrized elementary matrix for the antisymmetric kind. Appendix A lists
them; each is normalized to unit norm, and the library's copies turn out to be
mutually orthogonal as well, so the Gram matrices of §5.3 are identities — but
the construction does not assume it.

== The pairing theorem

#thm("5.2", name: "pairing")[
  Let $bold(z)^L_(lambda a)$, $bold(z)^R_(mu b)$ be test vectors as in Definition 5.1,
  and $bold(S) in cal(C)$. Then

  $ ⟨ bold(z)^L_(lambda a), bold(S) bold(z)^R_(mu b) ⟩
    = delta_(lambda mu) (bold(G)_lambda bold(B)_lambda)_(a b), $ <eq-pairing>

  where $bold(G)_lambda$ is the Gram matrix of the left copies,
  $(bold(G)_lambda)_(a a') = ⟨ bold(m)_a^L, bold(m)_(a')^L ⟩$, and
  $bold(B)_lambda$ is the block of @eq-wedderburn. In particular, with
  orthonormal copies the pairing *is* the block entry.
]

#proof[
  Distinct kinds are orthogonal: the isotypic components of a unitary
  representation are mutually orthogonal, so
  $⟨ bold(z)^L_(lambda a), bold(S) bold(z)^R_(mu b) ⟩ = 0$ for
  $lambda != mu$ because $bold(S)$ maps the $mu$-component of $W_R$ into the
  $mu$-component of $W_L$ (Schur). For $lambda = mu$, insert @eq-wedderburn
  and use that the carrier part acts as the identity in the middle:
  $ ⟨ bold(e) ⊗ bold(m)_a, (bold(1) ⊗ bold(B)_lambda)(bold(e) ⊗ bold(m)_b) ⟩
  = ⟨ bold(e), bold(e) ⟩ dot ⟨ bold(m)_a, bold(B)_lambda bold(m)_b ⟩ . $
  The first factor is 1 by the unit-norm condition, and the second is
  $sum_(a') ⟨ bold(m)_a, bold(m)_(a') ⟩ (bold(B)_lambda)_(a' b)$,
  which is the stated Gram product.
]

#rem("5.3")[
  The two conditions in Definition 5.1 are what make @eq-pairing a *recovery* rather
  than a scaled recovery. If the left and right internal vectors differed, the
  right-hand side would carry a factor $⟨ bold(e)^L, bold(e)^R ⟩$;
  if each copy used its own internal vector, the Gram matrix of the ambient
  vectors would not be the Gram matrix of the copies, and the correction of
  §5.3 would not be a change of basis in the multiplicity space. The code
  builds both sides from one function (`Kind.test_vectors`), which is the
  implementation of "the same internal vector".
]

== The dictionary

#cor("5.4", name: "the block table")[
  Define the *table*

  $ bold(P)_((lambda a b), k) = ⟨ bold(z)^L_(lambda a),
    bold(T)_k bold(z)^R_(lambda b) ⟩. $ <eq-table>

  Then for every factor list $bold(c)$ with
  $bold(S) = sum_k c_k bold(T)_k$, the blocks are

  $ (bold(B)_lambda)_(a b) = sum_(a') (bold(G)_lambda^(-1))_(a a')
    (bold(P) bold(c))_((lambda a' b)), $ <eq-blocks>

  and conversely $bold(c) = bold(P)^dagger bold(B)$, where $bold(P)^dagger$
  is the pseudo-inverse. Both maps are exact — not approximations — and are
  mutual inverses on $cal(C)$ and on the blocks respectively.
]

#proof[
  Linearity of the pairing in $bold(S)$ gives
  $⟨ bold(z)^L, bold(S) bold(z)^R ⟩ = sum_k c_k bold(P)_((lambda a b),k)$,
  and @eq-pairing identifies that with $(bold(G)_lambda bold(B)_lambda)_(a b)$.
  Inverting $bold(G)_lambda$ gives @eq-blocks. The table has one column per
  component ($d$ of them) and one row per pair of copies, $sum_lambda
  m_lambda^L m_lambda^R$ — so $d times d$ in the asymmetric case, and
  $sum_lambda m_lambda^L m_lambda^R$ by $d$ in the symmetric one ($15$ against
  $11$ for $k = 2$), where the pairs $(a, b)$ and $(b, a)$ carry the same entry.
  It has full rank in the stable range ($bold(G)$ is positive definite and the
  pairing is an isomorphism by Theorem 3.1), so $bold(P)^dagger$ is a two-sided
  inverse on the relevant subspaces.
]

#rem("5.5")[
  This is the whole content of the word "exact". $bold(P)$ is not a fit to
  the operator and there is no residual to report: it is the matrix of a
  change of basis, from the factor coordinates to the block coordinates, and
  it is inverted exactly. The code builds it once per (specification,
  dimensions) pair, in `Blocks.build`, and from then on the run-time step
  reads the table instead of probing anything. @eq-blocks appears in
  the implementation as `Blocks.to_blocks` (the $bold(P) bold(c)$ with the Gram
  correction already folded into the table) and `Blocks.of_blocks` (the
  pseudo-inverse); the Gram matrices are stored in the per-group records as
  `gram_l` and `gram_r`. In the symmetric $k = 2$ case the table is
  $15 times 11$: the entries $(a, b)$ and $(b, a)$ carry the same number, so
  the four extra rows are redundant and the pseudo-inverse is an exact left
  inverse on the range, not a least-squares fit.
]

#rem("5.6")[
  *Why "test vector" and not "probe vector".* Both names have been used. In
  the earlier, surrogate-based implementation the vectors were genuine
  *probes*: the operator was applied to them at run time and the inner
  products were taken then. In the present design nothing is probed — the
  operator is never applied to a test vector outside a test suite; the
  vectors only serve to pair against the basis elements at compile time, and
  what survives into the step is the table. That makes them test vectors in
  the sense of test functionals: canonical witnesses that extract one
  coordinate of the operator. The code says `Kind.test_vectors`, and this note
  follows it.
]

#rem("5.7")[
  Note how the count works, because it is the answer to "why is this cheap".
  There are $sum_lambda (m_lambda^L + m_lambda^R)$ test vectors, but
  $sum_lambda m_lambda^L m_lambda^R$ pairings, and each pairing costs one
  application of one basis tensor — not of the operator. For $k = 2$ that is
  $2 + 3 + 1 + 1 = 7$ vectors (per free-axis basis vector) and
  $2^2 + 3^2 + 1^2 + 1^2 = 15$ pairings. Nothing here is $D$-sized: the table is
  $O(d^2)$ scalars, the blocks it produces are $sum_lambda m_lambda^L
  m_lambda^R f_L f_R = dim cal(C)$ numbers, and all of it is a compile-time
  constant.
]

= Transport

== Definition

The blocks of @eq-wedderburn are the size-free description of an operator in
$cal(C)$. "Transport" is the operation that exploits this: re-encode a factor
list estimated at one size as the factor list of the operator with the same
blocks at another.

#defn("6.1", name: "transport")[
  Let $bold(Phi)_n : RR^d -> ⊕_lambda "Mat"(m_lambda^L f_L times m_lambda^R f_R)$
  be the dictionary of Corollary 5.4 at permuted dimension $n$, and let $n'$ be another
  stable size. Transport is
  $ tau_(n -> n') = bold(Phi)_(n')^(-1) compose bold(Phi)_n . $
]

#prop("6.2")[
  Transport is well defined and linear, it satisfies
  $bold(Phi)_(n') compose tau_(n -> n') = bold(Phi)_n$ by construction, and
  $tau_(n' -> n'') compose tau_(n -> n') = tau_(n -> n'')$. It collapses to the
  identity exactly when the basis's normalization makes the coordinates
  size-independent, which happens for $k = 1$ but not for $k = 2$ (§6.2).
]

#prop("6.3", name: "what transport preserves")[
  Let $bold(S)_n in cal(C)_n$ have blocks $bold(B)_lambda$, and let
  $bold(S)_(n')$ be the transported operator. Then

+ the blocks of $bold(S)_(n')$ are the same matrices $bold(B)_lambda$;
+ the distinct eigenvalues of $bold(S)_(n')$ are those of the $bold(B)_lambda$,
  with multiplicity $d_lambda (n')$;
+ for every function $f$ analytic on a neighbourhood of the spectrum,
  $f(bold(S)_(n'))$ is the transport of $f(bold(S)_n)$ — the functional
  calculus commutes with transport;
+ what changes is multiplicity, and hence $tr$, $det$ and norms:
  $ tr bold(S)_(n') = sum_lambda d_lambda (n') tr bold(B)_lambda , $ and
  $op("rank")$ changes too, $op("rank") bold(S)_(n') = sum_lambda d_lambda (n') op("rank") bold(B)_lambda$.
]

#proof[
  The first two are @eq-wedderburn read at $n'$: in a basis that decomposes
  $cal(S)_lambda ⊗ cal(M)_lambda$ as carrier-first, $bold(1)_(d_lambda) ⊗ bold(B)_lambda$
  is block diagonal with $d_lambda$ copies of $bold(B)_lambda$. The third
  follows from Theorem 7.1 below, and the last two from the same block-diagonal
  picture. The identity $sum_lambda d_lambda m_lambda = n^k$ (the dimension of
  the space) is the check that the multiplicities are the right ones.
]

== What transport does not preserve

Transport is not the identity on a factor list in general, and it is worth
seeing both regimes concretely.

*One permuted axis ($k = 1$).* With the library's normalization, the same
factor list gives the same blocks at every stable size. Checked on
$n = 5 -> 8$ with $f = 3$: the transported factors differ from the originals
by less than $10^(-7)$, and the blocks are bitwise equal. The reason is
@eq-k1 together with the $1 slash n$ in the normalization of the all-ones
term: the coordinates of the mean direction are already size-free.

*Two permuted axes ($k = 2$).* Not so: transporting $n = 5 -> 8$ multiplies
the eleven coefficients by factors between $0.43$ and $1.25$, and some of them
by exactly one. The blocks, of course, transport unchanged. Which coordinate picks up which factor is a property of
the basis, not of the operator — another way to see that the coordinates are
an encoding and the blocks are the object.

== Why not "just shrink the model"

The tempting shortcut: keep the factor list, evaluate it at a small size
$n_"s"$, do the linear algebra there, map back. It is wrong in general, and
the failure is exactly the size dependence above. With a random $k = 2$
symmetric operator and its factor list fixed, evaluating at $n = 5, 6, 8, 12$
gives the distinct spectra

#align(center, table(
  columns: (auto, auto, auto),
  align: (center, center, left),
  stroke: none,
  inset: (x: 6pt, y: 3pt),
  table.hline(stroke: 0.7pt),
  [*$n$*], [*$tr bold(S)$*], [*distinct spectrum*],
  table.hline(stroke: 0.4pt),
  [$5$], [$-36.445$], [$-4.472$, $-3.552$, $-2.927$, $-2.399$, $-1.089$, $2.097$, $6.844$],
  [$6$], [$-56.734$], [$-4.476$, $-3.611$, $-3.003$, $-2.399$, $-1.089$, $1.961$, $6.946$],
  [$8$], [$-108.04$], [$-4.483$, $-3.671$, $-3.107$, $-2.399$, $-1.089$, $1.770$, $7.220$],
  [$12$], [$-253.19$], [$-4.494$, $-3.674$, $-3.227$, $-2.399$, $-1.089$, $1.542$, $7.899$],
  table.hline(stroke: 0.7pt),
))

Two of the seven eigenvalues do not move, because those kinds have a
scalar block whose entry is a single coefficient; the other five drift. So the
operator at $n = 5$ and "the same operator" at $n = 12$ are genuinely
different — and a matrix function computed at the small size and mapped back
would be the function of the wrong operator. Transport is the correct version
of the shortcut: transport the *description* to the small size
($tau_(n -> n')$), do the linear algebra, transport back.

#rem("6.4")[
  Transport is used in three places. In the tests, to check that the block
  coordinates of a random operator are equal across sizes and that the dense
  spectrum at size $n$ matches the block spectrum (`test_blocks`). In the
  design of the estimator, to explain why the block path replaces the old
  "evaluate the factors at a surrogate size" path, which was silently
  transporting the wrong object. And in principle, in transfer: a metric
  estimated at one width, re-encoded at another, is the same operator there,
  up to the multiplicities that the larger width introduces.
]

= Analytic functions and small eigendecompositions

We can now state the result the whole construction exists for, and count the
cost.

#thm("7.1", name: "block functional calculus")[
  Let $bold(S) = ⊕_lambda (bold(1)_(cal(S)_lambda) ⊗ bold(B)_lambda)$ and let
  $f$ be analytic on a neighbourhood of the spectrum of $bold(S)$. Then

  $ f(bold(S)) = ⊕_lambda (bold(1)_(cal(S)_lambda) ⊗ f(bold(B)_lambda)). $ <eq-calculus>

  In words: $f(bold(S))$ is in the commutant, its blocks are the $f(bold(B)_lambda)$,
  and the carrier is untouched. The same holds for $f$ defined by the spectral
  theorem on the positive semidefinite cone ($x mapsto x^(1 slash 2)$,
  $x mapsto (x + lambda)^(-1)$, $dots$).
]

#proof[
  Two proofs, because they say different things.

  *Power series.* Multiplication is componentwise on carriers,
  $(bold(1) ⊗ bold(B))(bold(1) ⊗ bold(B)') = bold(1) ⊗ bold(B B')$, so
  $bold(S)^m = ⊕_lambda (bold(1) ⊗ bold(B)_lambda^m)$ by induction. If
  $f(z) = sum_m a_m z^m$ converges on a disc containing the spectrum, then
  $f(bold(S)) = sum_m a_m bold(S)^m = ⊕_lambda (bold(1) ⊗ sum_m a_m bold(B)_lambda^m)$,
  and the inner sum is $f(bold(B)_lambda)$. Convergence is inherited because
  the norm of $bold(1) ⊗ bold(X)$ is that of $bold(X)$.

  *Spectral theorem.* Diagonalize each block,
  $bold(B)_lambda = sum_j beta_(lambda j) bold(u)_(lambda j) bold(u)_(lambda j)^top$.
  Then the vectors $bold(e) ⊗ bold(u)_(lambda j)$, for $bold(e)$ in an
  orthonormal basis of the carrier, are an orthonormal eigenbasis of
  $bold(S)$ with eigenvalue $beta_(lambda j)$, each repeated $d_lambda$ times,
  and @eq-calculus is the spectral theorem applied in that basis.
]

#cor("7.2", name: "spectrum and eigenvectors")[
  The eigenvalues of $bold(S)$ are the eigenvalues of the blocks, and the
  eigenvectors are $bold(e) ⊗ bold(u)$: a carrier basis vector tensored with a
  block eigenvector. The width $n$ enters only through the *multiplicity*
  $d_lambda$ with which each block eigenvalue appears.
]

#cor("7.3", name: "damping is one scalar")[
  Let $bold(S)$ be positive semidefinite and $lambda > 0$. Then

  $ (bold(S) + lambda bold(1))^(-1) = ⊕_lambda (bold(1) ⊗ (bold(B)_lambda + lambda bold(1))^(-1)),
    quad quad
    (bold(S)^(1 slash 2) + lambda bold(1))^(-1) = ⊕_lambda (bold(1) ⊗ (bold(B)_lambda^(1 slash 2) + lambda bold(1))^(-1)), $

  and in either form the largest eigenvalue of the result is at most
  $1 slash lambda$, whatever the spectrum of $bold(S)$ and whatever $n$.
]

#proof[
  $lambda bold(1)_W = ⊕_lambda (bold(1)_(cal(S)_lambda) ⊗ lambda bold(1)_(cal(M)_lambda) ⊗ bold(1)_(F))$
  because the decomposition is orthogonal and the carrier action is the
  identity: this is where the fact that the blocks are matrices in *orthonormal*
  bases is used. Then apply Theorem 7.1 to $x mapsto (x + lambda)^(-1)$ and, for the
  root form, to $x mapsto x^(1 slash 2)$ followed by $x mapsto (x + lambda)^(-1)$.
  The gain bound is immediate from the spectrum: every denominator is at least
  $lambda$.
]

#rem("7.4")[
  Corollary 7.3 is why the optimizer has *one* absolute damping constant rather than
  one per kind or per block. If the carrier action were an arbitrary positive
  matrix $bold(C)_lambda$ instead of the identity, adding $lambda bold(1)$ to
  each block's spectrum would add $bold(C)_lambda ⊗ lambda bold(1)$, i.e. a
  different amount in each kind, and the "one scalar in gradient units"
  reading of the damping would be lost. The same remark explains why the
  bottom of the spectrum can be treated uniformly: a zero eigenvalue of any
  block becomes $lambda$, so no pseudo-inverse branch is needed, and the
  projection that removes roundoff-induced negative eigenvalues only removes
  roundoff (by Corollary 2.3 the metric is genuinely positive semidefinite).
]

#v(0.6em)
#block(sticky: true, below: 0.2em)[
  #text(size: 11.3pt, weight: "bold", fill: accent-2)[The algorithm]
]

#v(0.35em)
Everything above composes into six steps. Steps 1, 2, 5 and 6 touch
$D$-sized tensors but are linear contractions; step 3 is the only one where
the spectrum is touched, and it is small.

#align(center, table(
  columns: (auto, 1fr, auto),
  align: (center, left, right),
  stroke: none,
  inset: (x: 6pt, y: 3.6pt),
  table.hline(stroke: 0.8pt),
  [*step*], [*operation*], [*cost*],
  table.hline(stroke: 0.4pt),
  [1], [estimate the factors of $bold(S)$ — one contraction per component], [$O(D d)$],
  [2], [decode: $bold(B)_lambda = bold(G)_lambda^(-1) bold(P) bold(c)$, per kind], [$O(d^2)$],
  [3], [eigendecompose each assembled block $bold(B)_lambda$], [$sum_lambda (m_lambda f)^3$],
  [4], [apply $f$ to the spectrum of each block], [$O(d)$],
  [5], [re-encode: $bold(c)' = bold(P)^dagger bold(B)'$], [$O(d^2)$],
  [6], [apply the preconditioner — one contraction per component], [$O(D d)$],
  table.hline(stroke: 0.8pt),
))

Compare with the dense route: forming $bold(S)$ costs $D^2$ storage and
$O(D^3)$ for the eigendecomposition. For a first layer
$bold(W) in RR^(1000 times 784)$ with the hidden axis permuted, $D = 7.84 dot 10^5$,
so $D^3 approx 4.8 dot 10^(17)$ against $2 dot 784^3 approx 10^9$ for the two
$784 times 784$ blocks — five hundred million times cheaper, and the
multiplicity $n - 1 = 999$ of each block eigenvalue never appears in the
arithmetic. For the recurrent matrix of a vanilla RNN ($k = 2$, no free axes,
$D = n^2$) the blocks are $2 times 2$, $3 times 3$, $1 times 1$ and
$1 times 1$: fifteen numbers and four tiny eigendecompositions, at any width.

For the layer above the whole preconditioner is two $784 times 784$ inversions:
with the blocks of §4.2,

$ bold(P) = (bold(S) + lambda bold(1))^(-1)
  = Pi ⊗ (bold(B)_emptyset + lambda bold(1))^(-1)
  + Pi_perp ⊗ (bold(B)_([1]) + lambda bold(1))^(-1), $

and the step is $bold(P)$ applied to the momentum, two contractions. Nothing in
that formula knows $n = 1000$ except $Pi = bold(J) slash n$ — and $Pi$ acts as
a mean over units, not as a matrix that has to be inverted.

#rem("7.5")[
  The table's step 3 is the point of the whole exercise, and it is worth
  saying what is *not* happening there. The eigendecomposition is not a
  Krylov or Chebyshev approximation of $f(bold(S)) bold(b)$: it computes the
  whole function of the whole operator, and the reconstruction in steps 5–6 is
  exact. Nor is it a truncation of the spectrum: no eigenvalue is discarded.
  The reduction is exact because the directions that are not diagonalized are
  not approximated — they are forced to be the identity by @eq-wedderburn.
]

#rem("7.6")[
  Numerics, as implemented. The blocks are assembled in float32 but
  diagonalized in float64 (`Nx.eigh` on a float64 cast), which matters: the
  damping is often small compared with the largest eigenvalue, and the gain
  bound of Corollary 7.3 is a statement about the small end of the spectrum. Negative
  eigenvalues, which can only be roundoff on a positive semidefinite block,
  are clamped before the root form. The tables and the design matrix are
  factorized at compile time, so nothing in the step depends on a host read of
  a traced value and the whole step stays one `Rune.jit` program.
]

#rem("7.7")[
  A diagnostic warning: any quantity read off the blocks must account for the
  multiplicities of Proposition 6.3. The trace of the operator is
  $sum_lambda d_lambda tr bold(B)_lambda$, not $sum_lambda tr bold(B)_lambda$;
  the determinant and the Frobenius norm likewise. This is the one place where
  a "wrong by a factor of $n$" bug hides, and it is the same factor that made
  the surrogate of §6.3 lose the spectrum.
]

#pagebreak()

= What can go wrong

#figure(
  table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: none,
  inset: (x: 6pt, y: 4pt),
    table.hline(stroke: 0.8pt),
    [*failure*], [*symptom and guard*],
    table.hline(stroke: 0.4pt),
    [non-semisimple size],
    [Below $n = 2k$ the diagram tensors become linearly dependent; the design matrix loses rank and the block map is not invertible. The compiler raises `Invalid_argument` (`Kind.check_stable`) rather than returning a degenerate decomposition.],
    [incomplete symmetric basis],
    [Bundling $bold(T) + bold(T)^top$ spans $op("Sym")(W) ⊗ op("End")(F)$, which misses the $Lambda^2 W ⊗ Lambda^2 F$ part of the symmetric operators when $F$ has more than one entry. `Compiler.compile ~symmetric:true` detects that case and falls back to the full basis, so the estimator stays a projection.],
    [ill-conditioned basis],
    [The design matrix is a weighted zeta matrix of the partition lattice and is not orthogonal. It is factorized once with a jittered retry and an SVD pseudo-inverse fallback, in float64; the per-step solve then never sees the conditioning.],
    [non-orthonormal copies],
    [The Gram correction of @eq-blocks is not optional in general. The library's own test vectors are orthonormal (appendix A), but the table is built from the Gram matrices regardless, so changing the vectors does not silently scale the blocks.],
    [wrong size],
    [Re-evaluating a factor list at another size is not transport (§6.3); the blocks, and hence the spectrum and every function of it, change. Use `transport`, and only between stable sizes.],
    table.hline(stroke: 0.8pt),
  ),
  caption: [The failure modes, and where each one is caught.],
) <fig-failures>

= Correspondence with the code

#figure(
  [
  #set par(justify: false)
  #show raw: set text(size: 8.6pt, font: "DejaVu Sans Mono")
  #table(
    columns: (auto, auto),
    align: (left, left),
    stroke: none,
    inset: (x: 6pt, y: 3.4pt),
    table.hline(stroke: 0.8pt),
    [*mathematics*], [*library*],
    table.hline(stroke: 0.4pt),
    [$W_L, W_R$, the free spaces $F_L, F_R$], [`dims.left`, `dims.right`, `free_l`, `free_r`],
    [kinds $lambda$, carrier $d_lambda$, multiplicity $m_lambda$], [`Kind.t`, `Kind.stable_kinds`, `Kind.stable_multiplicity`, `Blocks.kinds`],
    [the commutant $cal(C)$], [`Compiler.basis`, `Basis.components`],
    [terms $bold(T)_k$, factors $c_k$], [`Term.t`, `Compiler.apply_block ~factors`, `estimate_factors`],
    [design matrix $bold(G)$, its factorization], [`Component.design_matrix`, `Compiler.build_estimator`],
    [test vectors $bold(z)^L, bold(z)^R$ (§5.1)], [`Kind.test_vectors`],
    [the table $bold(P)$ and $bold(P)^dagger$], [`Blocks.table`, `Blocks.inverse`],
    [the copy Gram $bold(G)_lambda$], [`Blocks.gram_l`, `Blocks.gram_r`],
    [blocks $bold(B)_lambda$ (per pair)], [`Blocks.block_matrices`, `Blocks.to_blocks`, `Blocks.of_blocks`],
    [transport $tau_(n -> n')$], [`Blocks.transport`, `Orbit.S.Second_order.transport`],
    [assembly over the tree (@eq-tree)], [`Orbit.S.Second_order.blocks_of_factors`, `factors_of_blocks`],
    [functions on blocks (Theorem 7.1)], [`Solve.damped_inverse_blocks`, `Solve.damped_inverse_root_blocks`],
    [the step of §7], [`Optim.prepare`, `Optim.solve`, `Optim.finish`],
    table.hline(stroke: 0.8pt),
  )
  ],
  caption: [Where each object of the note lives. The table $bold(P)$ and its
  inverse are hosted on the `Blocks.t` record; the per-pair blocks are
  assembled by the orbit layer, so the user-facing objects are the tree-level
  blocks.],
) <fig-code>

#block(
  width: 100%,
  breakable: false,
  inset: (x: 10pt, y: 9pt),
  fill: tint,
  radius: 2pt,
  [
    *The one-paragraph version.* An orbit average is the orthogonal projection
    of a gradient moment onto the commutant of the weight-space group
    (Proposition 2.2). That commutant decomposes, by Schur and Wedderburn, into a direct
    sum of small matrix algebras, and every element of it is the identity on
    the big part of each summand (Theorem 3.1). The commutant's coordinates are
    produced by a projection at compile time (§4); the matrices are recovered
    by pairing canonical test vectors — one per copy, not one per dimension —
    against the same basis (Theorem 5.2), a table fixed once and inverted exactly
    (§5.3). Because the recovered blocks are the operator's matrices in
    orthonormal bases, a function of the operator is the function of its
    blocks (Theorem 7.1), one absolute damping shifts every block uniformly (Corollary 7.3),
    and the width of the permuted axis appears only as the multiplicity with
    which each block eigenvalue is repeated. Transport is the same statement
    read backwards: the blocks are the size-free description, so an operator
    estimated at one width can be re-encoded at another (§6.2).
  ],
)

= Appendix A: the test vectors, explicitly

The library's families, for one group and no free axes, with the copy order of
`Kind.test_vectors`. Write $bold(J)$ for the all-ones $n times n$ matrix,
$bold(1)$ for the all-ones vector, and
$bold(h) = (1,-1,0,dots,0)^top slash sqrt(2)$ for the Helmert vector. Every
vector is listed unnormalized, with its norm; the code divides by it.

*$k = 1$* (vectors in $RR^n$):

#table(
  columns: (auto, auto, auto, auto),
  align: (center, center, left, center),
  stroke: none,
  inset: (x: 6pt, y: 3.2pt),
  table.hline(stroke: 0.7pt),
  [*kind*], [*copy*], [*vector*], [*norm*],
  table.hline(stroke: 0.4pt),
  [$emptyset$], [$0$], [$bold(1)$], [$sqrt(n)$],
  [$[1]$], [$0$], [$bold(h)$], [$1$],
  table.hline(stroke: 0.7pt),
)

So the trivial test vector is the uniform vector and the standard one is a
single orthonormalized difference; these are the two witnesses of the
$Pi slash Pi_perp$ decomposition.

*$k = 2$* (vectors in $RR^(n times n)$, written as matrices):

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (center, center, left, center, left),
  stroke: none,
  inset: (x: 6pt, y: 3.2pt),
  table.hline(stroke: 0.7pt),
  [*kind*], [*copy*], [*matrix*], [*norm*], [*what it is*],
  table.hline(stroke: 0.4pt),
  [$emptyset$], [$0$], [$bold(J)$], [$n$], [the parallel direction $n Pi$],
  [$emptyset$], [$1$], [$bold(1) - bold(J) slash n$], [$sqrt(n-1)$], [$n Pi_perp$, the identity on $bold(1)^perp$],
  [$[1]$], [$0$], [$bold(h) bold(1)^top$], [$sqrt(n)$], [$W$ inside $bold(1) ⊗ W$],
  [$[1]$], [$1$], [$bold(1) bold(h)^top$], [$sqrt(n)$], [$W$ inside $W ⊗ bold(1)$],
  [$[1]$], [$2$], [$op("Diag")(bold(h)) - (bold(h) bold(1)^top + bold(1) bold(h)^top) slash n$], [$sqrt(1 - 2 slash n)$], [$W$ inside $W ⊗ W$],
  [$[2]$], [$0$], [the four-cycle below], [$sqrt(8)$], [symmetric, zero diagonal, zero row sums],
  [$[1,1]$], [$0$], [$bold(e)_(01) - bold(e)_(10)$], [$sqrt(2)$], [an antisymmetrized elementary matrix],
  table.hline(stroke: 0.7pt),
)

The four-cycle is the matrix with entries $s_(01) = s_(10) = s_(23) = s_(32) = 1$
and $s_(12) = s_(21) = s_(30) = s_(03) = -1$, all others zero; it is
symmetric, traceless, has zero row sums and lies in the carrier of $[2]$. The
antisymmetric vector needs no projection for $i = 0$, $j = 1$: its row sums
already vanish, so the "remove the $bold(1) and bold(x)$ component" step of
`Kind.project_antisymmetric` is the identity there.

Three structural remarks, matching `Kind.test_vectors`:

+ every vector is the image of *one* internal vector under the copy's
  embedding: $bold(h)$ for all three copies of the standard kind, a single
  vector for the trivial kind, the four-cycle for $[2]$;
+ every vector has unit norm after the code's normalization, and the copies of
  a kind are mutually orthogonal in these examples (the pairings
  $⟨ Pi, Pi_perp ⟩$, $⟨ bold(h) bold(1)^top, bold(1) bold(h)^top ⟩$ and
  $⟨ bold(h) bold(1)^top, op("Diag")(bold(h)) - (bold(h) bold(1)^top + bold(1) bold(h)^top) slash n ⟩$
  all vanish because $bold(h)$ is zero-sum), so $bold(G)_lambda = bold(1)$;
  the correction term in the third standard copy is exactly what orthogonality
  costs: $⟨ bold(h) bold(1)^top, op("Diag")(bold(h)) ⟩ = 1$, and subtracting
  the components of $op("Diag")(bold(h))$ in the other two copies turns it
  into 0. A copy is not "some vector of the piece": it is the *canonical*
  one, and the canonical one has the other copies removed;
+ the requirements of Definition 5.1 are the reason for the apparently arbitrary
  expressions: each is the cheapest spelling of "a vector that lies in this
  copy and nowhere else".

= Appendix B: dimension counts

For reference, the three counts used above, with $k_L = k_R = k$ and
$f_L = f_R = f$:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, center, center, center),
  stroke: none,
  inset: (x: 6pt, y: 3.4pt),
  table.hline(stroke: 0.7pt),
  [*object*], [*$k = 1$*], [*$k = 2$*], [*general*],
  table.hline(stroke: 0.4pt),
  [factor coordinates $d$], [$2$], [$15$], [$sum_lambda m_lambda^2 = B_(2k)$],
  [bases of the commutant (numbers)], [$2 f^2$], [$15 f^2$], [$sum_lambda m_lambda^2 f^2$],
  [symmetric-block coordinates], [$3 f^2$], [$11 f^2$], [$sum_lambda m_lambda (m_lambda + 1) slash 2 dot f^2$],
  [test vectors, per free basis], [$2$], [$7$], [$sum_lambda m_lambda$],
  [block eigenproblems], [$2$], [$4$], [number of kinds],
  [largest block], [$f times f$], [$3 f times 3 f$], [$max_lambda m_lambda f$],
  table.hline(stroke: 0.7pt),
)

The rightmost column is the only one that matters at scale: none of the
entries grows with $n$. The Bell number in the first row is
$B_2 = 2$ and $B_4 = 15$; for $k = 3$ it would be $B_6 = 203$, still
independent of $n$.

// ------------------------------------------------------------------ refs

#heading(numbering: none)[References]

#block(inset: (left: 8pt), text(size: 9pt)[
  - A. Artemev, Y. Xia, S. Boyd, R. Yu, S. Dangel, G. Hennequin, A. Bernacchia.
    _Exploiting weight-space symmetries for approximating curvature._ ICML 2026.
    #link("https://arxiv.org/abs/2606.00442")[arXiv:2606.00442] — the estimator
    (Eq. 10), the commutant algebra, the factors.
  - T. Halverson, A. Ram. _Partition algebras._ European J. Combin. 26 (2005) —
    the algebra, its diagram basis, and the semisimplicity threshold.
  - H. Maron, H. Ben-Hamu, N. Shamir, Y. Lipman. _Invariant and equivariant
    graph networks._ ICLR 2019 — the fifteen-equality-pattern basis in the
    $n times n$ case, which is the $k = 2$ instance of §3.
  - _A seminormal form for partition algebras_, arXiv:1102.2047 — explicit
    irreducible representations, i.e. the test vectors of appendix A in
    general.
  - The `symo` sources: `lib/kind.ml` (kinds and test vectors),
    `lib/blocks.ml` (the table, the Gram, transport), `lib/compiler.ml` (the
    design matrix and its factorization), `lib/solve.ml` (the damped power on
    blocks), and `test/test_blocks.ml` (the checks quoted above).
])
