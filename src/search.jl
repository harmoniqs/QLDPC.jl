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
    min_k::Int=1,
    min_d::Int=1,
    trials::Int=400,
    seed::Int=0,
    metric::Function=efficiency,
    keep=nothing,
    verbose::Bool=false,
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
        fp_seed = seed + parse(Int, fp[1:8]; base=16) % 100000
        d = distance_rand(Hx, Hz; trials=trials, seed=fp_seed)
        (d == typemax(Int) || d < min_d) && continue
        w = max(row_weight(Hx), row_weight(Hz))
        eff = round(metric(n, k, d); digits=4)
        rec = (spec=spec, n=n, k=k, d=d, w=w, efficiency=eff, fingerprint=fp)
        seen[fp] = rec
        if verbose
            @info "candidate" spec n k d eff
        end
    end
    out = sort(collect(values(seen)); by=r -> r.efficiency, rev=true)
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
            s.n <= r.n && s.k >= r.k && s.d >= r.d &&
            (s.n < r.n || s.k > r.k || s.d > r.d)
            for s in records
        )
        !dominated && push!(front, r)
    end
    return sort(front; by=r -> (r.n, -r.k, -r.d))
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
    weight::Int=6,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    @assert weight == 6 "only weight-6 (3+3) random BB implemented; extend for other weights"
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    # keep a small dedup within generator
    seen_fp = Set{String}()
    for _ in 1:n_samples
        # sample 3 distinct (a,b) for A and 3 for B
        all_terms = [(a, b) for a in 0:l-1 for b in 0:m-1]
        shuffle!(rng, all_terms)
        A_terms = all_terms[1:3]
        B_terms = all_terms[4:6]
        Hx, Hz = build_bb(l, m, A_terms, B_terms)
        fp = fingerprint(Hx, Hz)
        fp in seen_fp && continue
        push!(seen_fp, fp)
        spec = Dict("l" => l, "m" => m, "A_terms" => A_terms, "B_terms" => B_terms, "family" => "bb")
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
    recs = screen(cands; trials=20, seed=0)
    @test length(recs) == 1
end

@testitem "search: pareto frontier" begin
    recs = [
        (spec="a", n=72, k=12, d=6, w=6, efficiency=3.0, fingerprint="a"),
        (spec="b", n=72, k=8, d=6, w=6, efficiency=2.0, fingerprint="b"),
        (spec="c", n=144, k=12, d=10, w=6, efficiency=8.33, fingerprint="c"),
    ]
    front = pareto_frontier(recs)
    # "b" is dominated by "a" (same n, lower k)
    @test length(front) == 2
    @test any(r -> r.spec == "a", front)
    @test any(r -> r.spec == "c", front)
end

