"""
search.jl — generic search funnel for discovering codes.

Mirrors `research/kit/search.py`:
  generate (spec, Hx, Hz) → screen (k≥min, d upper-bound via distance_rand, dedup by fingerprint → rank by efficiency) → pareto_frontier

All pure Julia + SparseArrays.
"""

using SparseArrays
using SHA
using Random

"""
    efficiency(n, k, d) -> Float64

Board headline figure `k*d^2 / n`.
"""
efficiency(n::Int, k::Int, d::Int) = n == 0 ? 0.0 : k * d * d / n

"""
    efficiency_weighted(n, k, d, w; a=1) -> Float64

Weight-aware efficiency `k*d^2 / (n * w^a)`.
Heavier checks (larger w) are penalized — mirrors the board's weight-class
tracks where a high-kd² code at w=16 is not equivalent to the same kd² at w=6.
`a` is the weight exponent (default 1, linear).
"""
function efficiency_weighted(n::Int, k::Int, d::Int, w::Int; a::Real = 1)::Float64
    n == 0 && return 0.0
    w <= 0 && return k * d * d / n
    return k * d * d / (n * w^a)
end

# convenience: record-based overload
efficiency_weighted(r::NamedTuple; a::Real = 1) = efficiency_weighted(r.n, r.k, r.d, r.w; a = a)

"""
    geometric_efficiency(k, d, n, rho, r) -> Float64

Geometric efficiency `4*k*d^2 / (n * rho^2 * r^4)` — planar figure of merit
that folds in qubit density rho and interaction radius r.
Mirrors `qldpc-challenge-jl/src/CSSCode.jl:geometric_efficiency`.
Returns 0.0 if any denominator is zero.
"""
function geometric_efficiency(k::Int, d::Int, n::Int, rho::Real, r::Real)::Float64
    (n == 0 || rho == 0 || r == 0) && return 0.0
    return 4 * k * d * d / (n * rho^2 * r^4)
end

# record-based overload (expects fields k,d,n,rho,r or layout-like)
function geometric_efficiency(r::NamedTuple, rho::Real, radius::Real)
    geometric_efficiency(r.k, r.d, r.n, rho, radius)
end

"""
    overhead(n, k) -> Float64

Qubit overhead `n/k` — physical per logical.
Returns `Inf` if `k == 0`.
"""
overhead(n::Int, k::Int) = k == 0 ? Inf : n / k
overhead(r::NamedTuple) = overhead(r.n, r.k)

"""
    weight_class(w) -> String

Mirrors `verify/qldpc_verify.py:weight_class` logic:
  w ≤ 4  → "weight-4"
  w ≤ 6  → "weight-6"
  w ≤ 8  → "weight-8"
  else   → "weight-9plus"
Nested — a weight-4 code also qualifies for the looser caps.
"""
function weight_class(w::Int)::String
    w <= 4 && return "weight-4"
    w <= 6 && return "weight-6"
    w <= 8 && return "weight-8"
    return "weight-9plus"
end

weight_class(Hx::AbstractMatrix, Hz::AbstractMatrix) = weight_class(max(row_weight(Hx), row_weight(Hz)))
weight_class(c::CSSCode) = weight_class(c.Hx, c.Hz)
weight_class(r::NamedTuple) = weight_class(r.w)

"""
    fingerprint(Hx, Hz) -> String

Exact-duplicate key: SHA-256 of the stacked RREFs of Hx and Hz (first 16 hex chars).
Equal fingerprint ⇒ identical stabilizer group.
"""
function fingerprint(Hx::AbstractMatrix, Hz::AbstractMatrix)::String
    Rx, _ = rref_gf2(Hx)
    Rz, _ = rref_gf2(Hz)
    b = sha256(vcat(Rx[:], [0x7c], Rz[:]))  # 0x7c = '|' separator
    return bytes2hex(b)[1:16]
end

fingerprint(c::CSSCode) = fingerprint(c.Hx, c.Hz)

"""
    screen(candidates; min_k=1, min_d=1, trials=400, seed=0, metric=efficiency, keep=nothing, verbose=false)

Screen an iterable of `(spec, Hx, Hz)` candidates (spec is any JSON-serializable
identifier, e.g. a Dict or String). For each distinct code: require `k ≥ min_k`,
estimate `d` with `distance_rand` (upper bound), require `d ≥ min_d`, and score
with `metric(n,k,d)`. Returns records sorted by score (best first), deduplicated
by fingerprint, truncated to `keep` if given.

Record shape: `NamedTuple{(:spec,:n,:k,:d,:w,:efficiency,:fingerprint)}`.
`w` is max check weight.
"""
function screen(
    candidates;
    min_k::Int = 1,
    min_d::Int = 1,
    trials::Int = 400,
    seed::Int = 0,
    metric::Function = efficiency,
    keep = nothing,
    verbose::Bool = false,
)
    seen = Dict{String,NamedTuple}()
    for (spec, Hx, Hz) in candidates
        verify_css(Hx, Hz) || continue
        k = compute_k(Hx, Hz)
        k < min_k && continue
        fp = fingerprint(Hx, Hz)
        haskey(seen, fp) && continue
        n = size(Hx, 2)
        # seed per candidate from fingerprint so sweeps are reproducible but diverse
        # use first 8 hex chars as Int
        fp_seed = seed + parse(Int, fp[1:8]; base = 16) % 100000
        d = distance_rand(Hx, Hz; trials = trials, seed = fp_seed)
        (d == typemax(Int) || d < min_d) && continue
        w = max(row_weight(Hx), row_weight(Hz))
        eff = round(metric(n, k, d); digits = 4)
        rec = (spec = spec, n = n, k = k, d = d, w = w, efficiency = eff, fingerprint = fp)
        seen[fp] = rec
        if verbose
            @info "candidate" spec n k d eff
        end
    end
    out = sort(collect(values(seen)); by = r -> r.efficiency, rev = true)
    if keep !== nothing
        out = out[1:min(keep, length(out))]
    end
    return out
end

"""
    pareto_frontier(records) -> Vector

Pareto-optimal records over (n smaller, k larger, d larger): those not dominated
by any other (no other has n'≤n, k'≥k, d'≥d with at least one strict).
"""
function pareto_frontier(records::AbstractVector)
    front = []
    for r in records
        dominated = any(
            s !== r &&
            s.n <= r.n &&
            s.k >= r.k &&
            s.d >= r.d &&
            (s.n < r.n || s.k > r.k || s.d > r.d) for s in records
        )
        !dominated && push!(front, r)
    end
    return sort(front; by = r -> (r.n, -r.k, -r.d))
end

"""
    pareto_frontier_weighted(records; a=1) -> Vector

Pareto frontier sorted by weight-aware efficiency `kd²/(n*w^a)` (best first).
Applies the same dominance filter as `pareto_frontier`, then re-sorts by
`efficiency_weighted` so a lighter-weight code can outrank a heavier one
with higher raw `kd²/n`.
"""
function pareto_frontier_weighted(records::AbstractVector; a::Real = 1)
    front = pareto_frontier(records)
    return sort(front; by = r -> efficiency_weighted(r.n, r.k, r.d, r.w; a = a), rev = true)
end

"""
    sample_bb(l, m, n_samples; weight=6, rng=Random.GLOBAL_RNG) -> Vector

Random BB enumerator: samples `n_samples` random A/B term sets of given weight
split (weight assumed 6 → 3+3) on the l×m torus. Yields `(spec, Hx, Hz)` triples
for `screen`.
"""
function sample_bb(
    l::Int,
    m::Int,
    n_samples::Int;
    weight::Int = 6,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    @assert weight == 6 "only weight-6 (3+3) random BB implemented; extend for other weights"
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    # keep a small dedup within generator
    seen_fp = Set{String}()
    for _ = 1:n_samples
        # sample 3 distinct (a,b) for A and 3 for B
        all_terms = [(a, b) for a = 0:l-1 for b = 0:m-1]
        shuffle!(rng, all_terms)
        A_terms = all_terms[1:3]
        B_terms = all_terms[4:6]
        Hx, Hz = build_bb(l, m, A_terms, B_terms)
        fp = fingerprint(Hx, Hz)
        fp in seen_fp && continue
        push!(seen_fp, fp)
        spec = Dict(
            "l" => l,
            "m" => m,
            "A_terms" => A_terms,
            "B_terms" => B_terms,
            "family" => "bb",
        )
        push!(out, (spec, Hx, Hz))
    end
    return out
end

using TestItems

@testitem "search: fingerprint dedup and efficiency" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    fp = fingerprint(Hx, Hz)
    @test length(fp) == 16
    @test efficiency(72, 12, 6) ≈ 12 * 36 / 72
    # screen dedup
    cands = [("a", Hx, Hz), ("b", Hx, Hz)]
    recs = screen(cands; trials = 20, seed = 0)
    @test length(recs) == 1
end

@testitem "search: pareto frontier" begin
    recs = [
        (spec = "a", n = 72, k = 12, d = 6, w = 6, efficiency = 3.0, fingerprint = "a"),
        (spec = "b", n = 72, k = 8, d = 6, w = 6, efficiency = 2.0, fingerprint = "b"),
        (spec = "c", n = 144, k = 12, d = 10, w = 6, efficiency = 8.33, fingerprint = "c"),
    ]
    front = pareto_frontier(recs)
    # "b" is dominated by "a" (same n, lower k)
    @test length(front) == 2
    @test any(r -> r.spec == "a", front)
    @test any(r -> r.spec == "c", front)
end

@testitem "search: weight-aware ranking" begin
    # two known codes: [[360,12,24]] w6 vs [[390,82,32]] w16
    # raw efficiency: kd^2/n  → 12*576/360=19.2  vs 82*1024/390≈215.3
    @test efficiency(360, 12, 24) ≈ 19.2 atol=1e-6
    @test efficiency(390, 82, 32) ≈ 82 * 1024 / 390 atol=1e-6
    # weighted: kd^2/(n*w^a)
    @test efficiency_weighted(360, 12, 24, 6; a=1) ≈ 19.2/6 atol=1e-6
    @test efficiency_weighted(390, 82, 32, 16; a=1) ≈ (82*1024/390)/16 atol=1e-6
    # a=2 penalizes heavier more — with a=3 the w6 code outranks w16
    @test efficiency_weighted(360, 12, 24, 6; a=3) > efficiency_weighted(390, 82, 32, 16; a=3)
    # unweighted: 390 wins; weighted a=3: 360 wins — ranking flips
    @test efficiency(390, 82, 32) > efficiency(360, 12, 24)
    @test efficiency_weighted(360, 12, 24, 6; a=3) > efficiency_weighted(390, 82, 32, 16; a=3)

    # overhead
    @test overhead(360, 12) ≈ 30.0
    @test overhead(390, 82) ≈ 390/82 atol=1e-9
    @test overhead(72, 0) == Inf

    # geometric_efficiency: 4*k*d^2/(n*rho^2*r^4)
    # example: k=12,d=6,n=72,rho=2,r=1.5 → 4*12*36/(72*4*5.0625) = 1728/(1458)=1.185...
    @test geometric_efficiency(12, 6, 72, 2.0, 1.5) ≈ 4*12*36/(72*4*1.5^4) atol=1e-9
    @test geometric_efficiency(12, 6, 0, 2.0, 1.5) == 0.0
    @test geometric_efficiency(12, 6, 72, 0.0, 1.5) == 0.0

    # weight_class mirrors verify/qldpc_verify.py
    @test weight_class(4) == "weight-4"
    @test weight_class(5) == "weight-6"
    @test weight_class(6) == "weight-6"
    @test weight_class(8) == "weight-8"
    @test weight_class(9) == "weight-9plus"
    @test weight_class(16) == "weight-9plus"
    # from matrices
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    @test weight_class(Hx, Hz) == "weight-6"
    @test weight_class(max(row_weight(Hx), row_weight(Hz))) == "weight-6"

    # pareto_frontier_weighted sorts by weighted metric
    recs = [
        (spec="w6", n=360, k=12, d=24, w=6, efficiency=19.2, fingerprint="a"),
        (spec="w16", n=390, k=82, d=32, w=16, efficiency=215.3, fingerprint="b"),
    ]
    front_w1 = pareto_frontier_weighted(recs; a=1)
    @test length(front_w1) == 2
    # a=1: w16 still outranks w6 on weighted
    @test front_w1[1].spec == "w16"
    front_w3 = pareto_frontier_weighted(recs; a=3)
    @test front_w3[1].spec == "w6"
end
