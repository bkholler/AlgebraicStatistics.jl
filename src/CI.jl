# -*- Conditional independence statements -*-

using StructEquality
using Combinatorics: powerset

export CIStmt, @CI_str, ci_statements, make_elementary

"""
    CIStmt(I, J, K)
    CI"A,B|X"

A conditional independence statement asserting that `I` is independent
of `J` given `K`. These parameters are lists of names of random variables.
The sets `I` and `J` must be disjoint as this package cannot yet deal
with functional dependencies.

The literal syntax CI"I...,J...|K..." is provided for cases in which all
your variable names consist of a single character. If `I` and `J` only
consist of a single element, the comma may be omitted.

## Examples

``` julia-repl
julia> CI"AB|X"
[A ⫫ B | X]
julia> CI"1,23|45"
[1 ⫫ {2, 3} | {4, 5}]
```
"""
struct CIStmt
    I; J; K
    # TODO: We currently bail on functional dependence statements
    # because they need special care in all the algebraic parts.
    CIStmt(I, J, K) = length(intersect(I, J)) > 0 ?
        error("Functional dependence statements are not yet implemented") :
        new(I, J, K)
end
@def_structequal CIStmt

# Allow CI"12,3|456" syntax to create a CIStmt. The short syntax
# CI"12|345" for elementary CI statements is also supported and
# is assumed if there is no comma.
#
# We suppose that the semigraphoid properties apply and make (I, K)
# and (J, K) disjoint. Note that we do support functional dependencies,
# i.e., I and J may have non-empty intersection.
macro CI_str(str)
    # General syntax "12,34|567"
    m = match(r"^(.+),(.+)[|](.*)$", str)
    # Short syntax for elementary statements "12|345"
    if m == nothing
        m = match(r"^(.)(.)[|](.*)$", str)
    end
    # Give up
    if m == nothing
        throw(ArgumentError(str * " is not a CI statement"))
    end

    parse_arg(s) = unique([string(c) for c in s])
    I, J, K = map(s -> parse_arg(s), m)
    # The setdiff is allowed by the semigraphoid axioms.
    return CIStmt(setdiff(I, K), setdiff(J, K), K)
end

function Base.show(io::IO, stmt::CIStmt)
    fmt(K) = length(K) == 1 ? string(K[1]) : "{" * join([string(x) for x in K], ", ") * "}"
    print(io, "[$(fmt(stmt.I)) ⫫ $(fmt(stmt.J)) | $(fmt(stmt.K))]")
end

"""
    ci_statements(random_variables::Vector{String})
    ci_statements(R::MarkovRing)

Return a list of all elementary CI statements over a given set of
variable names or a MarkovRing. A `CIStmt(I, J, K)` is elementary
if both `I` and `J` have only one element.

As a consequence of the semigraphoid properties, these statements
are enough to describe the entire CI structure of a probability
distribution.

## Examples

``` julia-repl
julia> ci_statements(["A", "B", "X", "Y"])
24-element Vector{CIStmt}:
 [1 ⫫ 2 | {}]
 [1 ⫫ 2 | 3]
 [1 ⫫ 2 | 4]
 [1 ⫫ 2 | {3, 4}]
...
 [3 ⫫ 4 | {}]
 [3 ⫫ 4 | 1]
 [3 ⫫ 4 | 2]
 [3 ⫫ 4 | {1, 2}]
```
"""
function ci_statements(random_variables::Vector{String})
    N = 1:length(random_variables)
    stmts = Vector{CIStmt}()
    for ij in powerset(N, 2, 2)
        M = setdiff(N, ij)
        for L in powerset(M)
            push!(stmts, CIStmt(
                [random_variables[ij[1]]],
                [random_variables[ij[2]]],
                 random_variables[L]
            ))
        end
    end
    return stmts
end

"""
    make_elementary(stmt::CIStmt; semigaussoid=false)

Convert a CIStmt into an equivalent list of CIStmts's all of which
are elementary. The default operation assumes the semigraphoid axioms
and converts [I ⫫ J | K] into the list consisting of [i ⫫ j | L]
for all i in I, j in J and L between K and (I ∪ J ∪ K) ∖ {i,j}.

If `semigaussoid` is true, the stronger semigaussoid axioms are
assumed and `L` in the above procedure does not range in sets
above `K` but is fixed to `K`. Semigaussoids are also known as
compositional graphoids.

## Examples

``` julia-repl
julia> make_elementary(CI"12,34|56")
16-element Vector{CIStmt}:
 [1 ⫫ 3 | {5, 6}]
 [1 ⫫ 3 | {5, 6, 2}]
 [1 ⫫ 3 | {5, 6, 4}]
...
 [2 ⫫ 4 | {5, 6, 3}]
 [2 ⫫ 4 | {5, 6, 1, 3}]

julia> make_elementary(CI"12,34|56"; semigaussoid=true)
4-element Vector{CIStmt}:
 [1 ⫫ 3 | {5, 6}]
 [1 ⫫ 4 | {5, 6}]
 [2 ⫫ 3 | {5, 6}]
 [2 ⫫ 4 | {5, 6}]
```
"""
function make_elementary(stmt::CIStmt; semigaussoid=false)
    N = union(stmt.I, stmt.J, stmt.K)
    elts = Vector{CIStmt}()
    for i in stmt.I
        for j in stmt.J
            # If we may suppose the semigaussoid axioms, the system
            # becomes simpler. Likewise for functional dependencies
            # because (i,i|K) implies (i,i|KL) for all L:
            M = (semigaussoid || i == j) ?
                typeof(stmt.K)([]) : setdiff(N, [i, j, stmt.K...])
            for L in powerset(M)
                push!(elts, CIStmt([i], [j], union(stmt.K, L)))
            end
        end
    end
    return elts
end

# vim: set expandtab ts=4 sts=-1 sw=4 tw=0:
