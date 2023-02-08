using HomotopyContinuation
using Base.Iterators: product, flatten
using Combinatorics: powerset
using AutoHashEquals

@auto_hash_equals(
struct CIStmt
    I; J; K
end)

# Allow CI"12,3|456" syntax to create a CIStmt. The short syntax
# CI"12|345" for elementary CI statements is also supported.
#
# We suppose that the semigraphoid properties apply and make (I, K)
# and (J, K) disjoint. Note that we do support functional dependencies,
# i.e., I and J may have non-empty intersection.
macro CI_str(str)
    # General syntax "12,34|567"
    m = match(r"^(\d+),(\d+)[|](\d*)$", str)
    # Short syntax for elementary statements "12|345"
    if m == nothing
        m = match(r"^(\d)(\d)[|](\d*)$", str)
    end
    # Give up
    if m == nothing
        throw(ArgumentError(str * " is not a CI statement"))
    end

    parse_arg(S) = unique([parse(Int64, c) for c in S])
    I, J, K = map(c -> parse_arg(c), m)
    # The setdiff is allowed by the semigraphoid axioms.
    return CIStmt(setdiff(I, K), setdiff(J, K), K)
end

# Convert a CIStmt into an equivalent list of CIStmt's all of which
# are elementary, supposing the semigraphoid axioms. Elementary
# CIStmt's have I and J singletons.
function elementary_ci(A::CIStmt; compositional_graphoid=false)
    N = union(A.I, A.J, A.K)
    elts = Vector{CIStmt}()
    for i in A.I
        for j in A.J
            # If we may suppose compositional graphoid axioms,
            # the system becomes simpler. Likewise for functional
            # dependencies because (i,i|K) implies (i,i|KL):
            M = (compositional_graphoid || i == j) ?
                typeof(A.K)([]) : setdiff(N, [i, j, A.K...])
            for L in powerset(M)
                push!(elts, CIStmt([i], [j], union(A.K, L)))
            end
        end
    end
    return elts
end

struct CIRing
    varname
    states
    variables
end

function ci_ring(states...; name::Symbol=:p)::CIRing
    states = map(a -> typeof(a) <: Number ? range(0, length=a) : a, states)
    idx = product(states...)
    variables = map(i -> Variable(name, i...), [collect(idx)...])
    return CIRing(name, states, variables)
end

# Return all states that the subvector indexed by K can attain in
# the CIRing R.
function states(R::CIRing, K)
    return [collect(product([R.states[k] for k in K]...))...]
end

# Return all elementary CI statements.
#
# FIXME: powerset() enumerates subsets by cardinality where as the
# convention in CI research data is binary-counter order. These two
# orders coincide for up to four random variables, so we are good
# for the moment.
function ci_statements(R::CIRing)
    N = 1:length(R.states)
    stmts = Vector{CIStmt}()
    for ij in powerset(N, 2, 2)
        M = setdiff(N, ij)
        for L in powerset(M)
            push!(stmts, CIStmt([ij[1]], [ij[2]], L))
        end
    end
    return stmts
end

# Return a specific variable in R whose indices are in x but
# jumbled according to N, i.e., variable(R, [3,2,1], [0,1,2])
# would return p_{2,1,0}.
function variable(R::CIRing, N, x)
    @assert sort(N) == 1:length(R.states)
    px = Vector{Int64}(undef, length(R.states))
    for i in N
        px[N[i]] = x[i]
    end
    return Variable(R.varname, px...)
end

# Return a marginal as a sum of variables from R. The indices in K
# are fixed to the event x and all other indices are summed over.
function marginal(R::CIRing, K, x)
    N = 1:length(R.states)
    M = setdiff(N, K)
    summands = Vector{Expression}()
    for y in states(R, M)
        push!(summands, variable(R, [K..., M...], [x..., y...]))
    end
    return +(summands...)
end

# Return a HomotopyContinuation.ModelKit.System for the conditional
# independence and functional dependence equations for the given CI
# statements.
function ci_equations(R::CIRing, stmts)::System
    eqs = Vector{Expression}()
    for stmt in flatten([elementary_ci(stmt) for stmt in stmts])
        i, j, K = stmt.I[1], stmt.J[1], stmt.K
        if i == j
            # We use the following formulation of functional depencence which
            # yields homogeneous equations of degree (number of states of i):
            #
            # For each event x of K there is an event y of i such that p(x,y) = p(x).
            # Together with the inequalities for probability distributions
            # this implies that for exactly one y we get p(y|x) = 1 and we
            # get zero for all others. This is a functional dependence.
            #
            # XXX: The equations p(x,y) = p(x) reduce to sums of p(p,y')
            # over all y' that are not y. Without the non-negativity of
            # all p's, there are many solutions to this system and expanding
            # the product over all these sums yields a dense polynomial...
            for x in states(R, K)
                px = marginal(R, K, x)
                factors = Vector{Expression}()
                for y in R.states[i]
                    pxy = marginal(R, [K..., i], [x..., y])
                    push!(factors, expand(px - pxy))
                end
                push!(eqs, *(factors...))
            end
        else
            # The proper CI equations are the familiar 2x2 determinants in
            # marginals of the probability tensor.
            Di = R.states[i]
            Dj = R.states[j]
            ijK = [i, j, K...]
            for z in states(R, K)
                for x in powerset(Di, 2, 2)
                    for y in powerset(Dj, 2, 2)
                        p11 = marginal(R, ijK, [x[1], y[1], z...]);
                        p12 = marginal(R, ijK, [x[1], y[2], z...]);
                        p21 = marginal(R, ijK, [x[2], y[1], z...]);
                        p22 = marginal(R, ijK, [x[2], y[2], z...]);
                        push!(eqs, p11*p22 - p12*p21)
                    end
                end
            end
        end
    end
    return System(eqs)
end

# Compute the CI structure of the distribution x which has one coordinate
# corresponding to each entry in R.variables.
#
# TODO: This is based on arbitrary and hardcoded thresholds.
function ci_structure(R::CIRing, x)
    toldigits = 8
    V = R.variables
    A = Vector{CIStmt}()
    for stmt in ci_statements(R)
        for f in expressions(ci_equations(R, [stmt]))
            v = abs(evaluate(f, V => x))
            if round(v; digits=toldigits) > 0.0
                @goto next_stmt
            end
        end
        push!(A, stmt)
        @label next_stmt
    end
    return A
end

struct InferenceResult
    code
    counterexample
    nwitnesses
    nrejections
end

# Sample between N and M random probability distributions and use Newton's
# method to obtain "nearby" points on the CI variety defined by P. If the
# Newton iteration fails, the sample is ignored. If a near-counterexample
# to P => Q is found, that is returned. If N points are encountered where
# the implication is true, then success is reported. If so many samples
# are rejected that out of M iterations neither N near-witnesses nor a
# near-counterexample could be found, the test is inconclusive.
#
# TODO: This is based on arbitrary and hardcoded thresholds.
function check_inference(R::CIRing, P, Q, N=100, M=2*N; positive=false)
    @assert N <= M
    V = R.variables
    n = length(V)
    F = ci_equations(R, P)
    G = [expressions(ci_equations(R, [stmt])) for stmt in Q]
    toldigits = 6

    nsuccesses = 0
    nrejections = 0
    for i in 1:M
        # FIXME: Probably not uniform on the simplex, but do we care?
        u = abs.(randn(Float64, n))
        u = u ./ sum(u)
        res = newton(F, u)
        if !is_success(res)
            nrejections += 1
            continue
        end
        x = res.x
        # Reject if the point has negative coordinates.
        if min(real.(x)...) < 0.0
            nrejections += 1
            continue
        end
        x = abs.(x) ./ abs(sum(x))
        # Also reject if we want positive points but this one has zero entries.
        if positive && round(min(x...); digits=toldigits) == 0.0
            nrejections += 1
            continue
        end

        for H in G
            for h in H
                v = abs(evaluate(h, V => x))
                if round(v; digits=toldigits) > 0.0
                    @goto next_stmt
                end
            end
            nsuccesses += 1
            if nsuccesses >= N
                return InferenceResult(:success, nothing, nsuccesses, nrejections)
            end
            @goto next_sample
            @label next_stmt
        end
        return InferenceResult(:failure, x, nsuccesses, nrejections)
        @label next_sample
    end
    return InferenceResult(:unknown, nothing, nsuccesses, nrejections)
end

# vim: set expandtab ts=4 sts=-1 sw=4 tw=0:
