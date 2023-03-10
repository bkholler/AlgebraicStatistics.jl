# -*- Markov rings for discrete random variables -*-

using Oscar
using Base.Iterators: enumerate, product, flatten
using Combinatorics: powerset

export MarkovRing, TensorRing, ring, random_variables, unknowns, state_space, marginal, ci_ideal

"""
    MarkovRing(rvs::Pair...; unknown="p", base_ring=QQ)

The polynomial ring whose unknowns are the entries of a probability tensor.
`rvs` is a list of pairs `X => Q` where `X` is the name of a random variable
and `Q` is the list of states it takes. The polynomial ring being constructed
will have one variable for each element in the cartesian product of the `Q`s.
It is an Oscar multivariate polynomial ring whose variables are named `p[...]`
and whose `base_ring` is by default `QQ`. You can change these settings via
the optional arguments.

## Examples

``` julia-repl
julia> R = MarkovRing("A" => 1:2, "B" => 1:2, "X" => 1:2, "Y" => 1:2; base_ring=GF(17))
MarkovRing for random variables A → {1, 2}, B → {1, 2}, X → {1, 2}, Y → {1, 2} in 16 variables over Galois field with characteristic 17
```
"""
struct MarkovRing
    ring
    random_variables
    state_spaces
end

function MarkovRing(rvs::Pair...; unknown="p", base_ring=QQ)
    random_variables = [p.first for p in rvs];
    state_spaces = [p.second for p in rvs];
    return MarkovRing(
        PolynomialRing(base_ring, unknown => Tuple(state_spaces)),
        random_variables,
        state_spaces
    )
end

"""
    const TensorRing = MarkovRing

The name `TensorRing` is an alias for `MarkovRing` because that is really what a
`MarkovRing` is: the coordinate ring of tensors of a fixed format. The name
`MarkovRing` is kept for compatibility in terminology with the Macaulay2 package
`GraphicalModels`.
"""
const TensorRing = MarkovRing

function Base.show(io::IO, R::MarkovRing)
    print(io, "$(typeof(R)) for random variables ",
        join([string(x) * " → " * "{" * join(R.state_spaces[i], ", ") * "}" for (i, x) in enumerate(R.random_variables)], ", "),
        " in $(length(gens(ring(R)))) variables over $(base_ring(ring(R)))"
    )
end

"""
    ring(R::MarkovRing)

Return the Oscar multivariate polynomial ring inside the MarkovRing.

## Examples

``` julia-repl
julia> ring(R)
Multivariate Polynomial Ring in 16 variables p[1, 1, 1, 1], p[2, 1, 1, 1], p[1, 2, 1, 1], p[2, 2, 1, 1], ..., p[2, 2, 2, 2] over Galois field with characteristic 17
```
"""
function ring(R::MarkovRing)
    return R.ring[1]
end

"""
    random_variables(R::MarkovRing)

Return the list of random variables used to create the MarkovRing.

## Examples

``` julia-repl
julia> random_variables(R)
4-element Vector{String}:
 "A"
 "B"
 "X"
 "Y"
```
"""
function random_variables(R::MarkovRing)
    return R.random_variables
end

ci_statements(R::MarkovRing) = ci_statements(random_variables(R))

"""
    unknowns(R::MarkovRing)

Return the tensor of variables in the polynomial ring.

## Examples

``` julia-repl
julia> unknowns(R)
2×2×2×2 Array{gfp_mpoly, 4}:
[:, :, 1, 1] =
 p[1, 1, 1, 1]  p[1, 2, 1, 1]
 p[2, 1, 1, 1]  p[2, 2, 1, 1]

[:, :, 2, 1] =
 p[1, 1, 2, 1]  p[1, 2, 2, 1]
 p[2, 1, 2, 1]  p[2, 2, 2, 1]

[:, :, 1, 2] =
 p[1, 1, 1, 2]  p[1, 2, 1, 2]
 p[2, 1, 1, 2]  p[2, 2, 1, 2]

[:, :, 2, 2] =
 p[1, 1, 2, 2]  p[1, 2, 2, 2]
 p[2, 1, 2, 2]  p[2, 2, 2, 2]
```
"""
function unknowns(R::MarkovRing)
    return R.ring[2]
end

function find_random_variables(R::MarkovRing, K)
    idx = [findfirst(x -> cmp(string(x), string(k)) == 0, R.random_variables) for k in K]
    if (j = findfirst(r -> r == nothing, idx)) != nothing
        error("random variable $(K[j]) not found in $(typeof(R))($(join([string(x) for x in R.random_variables], ", ")))")
    end
    return idx
end

"""
    state_space(R::MarkovRing, K=random_variables(R))

Return all states that the random subvector indexed by `K` can attain
in the ring `R`. The result is a `product` iterator unless `K` has only
one element.

## Examples

``` julia-repl
julia> collect(state_space(R, ["A", "B"]))
2×2 Matrix{Tuple{Int64, Int64}}:
 (1, 1)  (1, 2)
 (2, 1)  (2, 2)
```
"""
function state_space(R::MarkovRing, K=R.random_variables)
    idx = find_random_variables(R, K)
    return length(idx) == 1 ? R.state_spaces[idx[1]] : product([R.state_spaces[i] for i in idx]...)
end

# -*- CI equations for MarkovRing -*-

# p is a permutation of 1:n and x is a vector of length n.
# This method returns the components of x permuted by p,
# e.g. apply_permutation([3,2,1], [0,1,2]) == [2,1,0].
function apply_permutation(p, x)
    px = Vector{Int64}(undef, length(p))
    for i in p
        px[p[i]] = x[i]
    end
    return px
end

"""
    marginal(R::MarkovRing, K, x)

Return a marginal as a sum of unknowns from `R`. The argument `K` lists
random variables which are fixed to the event `x`; all other random
variables in `R` are summed over their respective state spaces.

## Examples

``` julia-repl
julia> R
MarkovRing for random variables A → {1, 2}, B → {1, 2}, X → {1, 2}, Y → {1, 2} in 16 variables over Rational Field

julia> marginal(R, ["A", "X"], [1,2])
p[1, 1, 2, 1] + p[1, 2, 2, 1] + p[1, 1, 2, 2] + p[1, 2, 2, 2]
```
"""
function marginal(R::MarkovRing, K, x)
    p = unknowns(R)
    N = random_variables(R)
    M = setdiff(N, K)
    summands = Vector()
    for y in state_space(R, M)
        idx = apply_permutation(
            find_random_variables(R, [K..., M...]),
            [x..., y...]
        )
        push!(summands, p[idx...])
    end
    return length(summands) == 1 ? summands[1] : +(summands...)
end

"""
    ci_ideal(R::MarkovRing, stmts)::MPolyIdeal

Return an Oscar ideal for the conditional independence statements
given by `stmts`.

## Examples

``` julia-repl
julia> R
MarkovRing for random variables A → {1, 2}, B → {1, 2}, X → {1, 2} in 8 variables over Rational Field

julia> ci_ideal(R, [CI"X,A|B", CI"X,B|A"])
ideal(p[1, 1, 1]*p[2, 1, 2] - p[2, 1, 1]*p[1, 1, 2], p[1, 2, 1]*p[2, 2, 2] - p[2, 2, 1]*p[1, 2, 2], p[1, 1, 1]*p[1, 2, 2] - p[1, 2, 1]*p[1, 1, 2], p[2, 1, 1]*p[2, 2, 2] - p[2, 2, 1]*p[2, 1, 2])
```
"""
function ci_ideal(R::MarkovRing, stmts)::MPolyIdeal
    eqs = Vector()
    for stmt in stmts
        # The proper CI equations are the familiar 2x2 determinants in
        # marginals of the probability tensor.
        # TODO: Functional dependence not yet supported.
        QI = state_space(R, stmt.I)
        QJ = state_space(R, stmt.J)
        IJK = [stmt.I..., stmt.J..., stmt.K...]
        for z in state_space(R, stmt.K)
            for x in powerset(QI, 2, 2)
                for y in powerset(QJ, 2, 2)
                    p11 = marginal(R, IJK, [x[1], y[1], z...]);
                    p12 = marginal(R, IJK, [x[1], y[2], z...]);
                    p21 = marginal(R, IJK, [x[2], y[1], z...]);
                    p22 = marginal(R, IJK, [x[2], y[2], z...]);
                    push!(eqs, p11*p22 - p12*p21)
                end
            end
        end
    end
    return ideal(eqs)
end

# vim: set expandtab ts=4 sts=-1 sw=4 tw=0:
