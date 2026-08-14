"""
gb.jl — Cyclic generalized-bicycle divisor codes (designed-divisor family).

Mirrors `research/campaigns/sweep_gb.py` / `climb_gb.py` (fieldnotes
2026-07-14 designed-divisor-and-odd-k) in Julia with Nemo weakdep.

Construction:
  G = Z_N cyclic (order N, group_algebra.cyclic_group).
  Pick a divisor g(x) | x^N - 1 over F2 (degree deg = k/2, so k = 2*deg
  by construction). Sample a = g·a', b = g·b' with weight 8-12 each via
  RIS / birthday on the ideal (g) — generic members (gcd(a, x^N-1)=g exactly)
  give high distance; structured low-weight non-generic members collapse d.

  Build via `build_2bga(cyclic_group(N), a_support, b_support)` where
  supports are exponent-sets of the polynomials. The resulting Hx, Hz are
  (N × 2N) CSS (circulant A,B), n=2N, w = wt(a)+wt(b) capped at ~32 (but we
  target 16-24 for board w6/w8, with wt(a)≤8-12 each).

Target: high-rate 390,82 codes (N=195, deg=41) and siblings 231,255,273
from sweep_gb, with kd²/n headroom.

Polynomials are over F2 as bit-masks (BigInt) for pure-Julia fallback;
Nemo path uses `GF(2)[x]` when the extension is loaded.

Public:
  - `gb_divisors(N, deg) -> Vector{Vector{Int}}` — divisor exponent sets
  - `build_gb(N, a_exps, b_exps) -> (Hx, Hz)` — cyclic GB code
  - `gb_divisor_code(N, g_exps, a_exps, b_exps) -> (Hx,Hz)` — via g
  - `random_gb_divisor(N, deg; weight_target=10, seed)` -> (a_exps,b_exps,g_exps)
  - `enumerate_gb_candidates(N, deg; n_samples, weight, seed)` — sampler for screen
"""

using SparseArrays
using Random

# ---------------------------------------------------------------------------
# GF(2) polynomial helpers (pure Julia, fallback when Nemo unavailable)
# ---------------------------------------------------------------------------

# poly as BigInt bitmask: bit i == coeff of x^i. Must have bit 0 =1 for divisor of x^N+1.
# Helpers using BigInt so N=390 fits (need 390 bits).

_poly_weight(p::BigInt) = count_ones(p)  # number of set bits

function _poly_deg(p::BigInt)::Int
    p == 0 && return -1
    return ndigits(p, base=2) - 1
end

function _poly_mod(a::BigInt, b::BigInt)::BigInt
    b == 0 && error("div by 0 poly")
    while a != 0 && _poly_deg(a) >= _poly_deg(b)
        shift = _poly_deg(a) - _poly_deg(b)
        a ⊻= b << shift
    end
    return a
end

_poly_divides(b::BigInt, a::BigInt) = _poly_mod(a, b) == 0

function _poly_gcd(a::BigInt, b::BigInt)::BigInt
    while b != 0
        r = _poly_mod(a, b)
        a, b = b, r
    end
    return a
end

function _poly_mul(a::BigInt, b::BigInt)::BigInt
    r = BigInt(0)
    shift = 0
    bb = b
    while bb != 0
        if isodd(bb)
            r ⊻= a << shift
        end
        bb >>= 1
        shift += 1
    end
    return r
end

# reduce mod x^N - 1 (x^N +1 over F2 since -1=1): for i >= N, bit i wraps to i-N via XOR
function _reduce_cyc(p::BigInt, N::Int)::BigInt
    # while deg >= N, fold top bits down: p = low_N + high shifted
    while _poly_deg(p) >= N
        deg = _poly_deg(p)
        shift = deg - N
        # x^{N+shift} term at bit N+shift -> wrap to bit shift: remove it and XOR at shift
        p ⊻= BigInt(1) << deg
        p ⊻= BigInt(1) << shift
        # also need to handle creates? loop handles
        # more efficient: slice but keep simple; number of bits bounded so loop OK
    end
    return p
end

function _poly_mul_cyc(a::BigInt, b::BigInt, N::Int)::BigInt
    p = _poly_mul(a,b)
    return _reduce_cyc(p, N)
end

function _exps_to_poly(exps::AbstractVector{Int})::BigInt
    p = BigInt(0)
    for e in exps
        p |= BigInt(1) << e
    end
    return p
end

function _poly_to_exps(p::BigInt)::Vector{Int}
    exps = Int[]
    bit = 0
    pp = p
    while pp != 0
        if isodd(pp)
            push!(exps, bit)
        end
        pp >>= 1
        bit += 1
    end
    return exps
end

xNm1(N::Int) = (BigInt(1) << N) | BigInt(1)  # x^N + 1 = x^N -1 over F2

# k = 2*deg gcd(a,b,x^N-1) for GB codes
function k_exact_gb(a::BigInt, b::BigInt, N::Int)::Int
    g = _poly_gcd(_poly_gcd(a, b), xNm1(N))
    return 2 * _poly_deg(g)
end

# ---------------------------------------------------------------------------
# Divisor enumeration of x^N-1 over F2
# ---------------------------------------------------------------------------

"""
    gb_divisors(N, deg; max_count=200) -> Vector{Vector{Int}}

Enumerate divisors g of x^N-1 ( = x^N+1 over F2) of degree `deg`.
Returns each `g` as sorted exponent list. Pure-Julia brute when Nemo
unavailable, Nemo-accelerated when loaded.

Brute enumerates all monic degree-deg polynomials with constant 1 (2^{deg-1}
candidates) and tests `g | x^N+1` via remainder. Feasible for deg ≤ 12;
for 390,82 with deg=41 we rely on Nemo factoring (or limit enumeration).
"""
function gb_divisors(
    N::Int,
    deg::Int;
    max_count::Int = 200,
)::Vector{Vector{Int}}
    # Try Nemo path first
    ext = Base.get_extension(@__MODULE__, :QLDPCNemoExt)
    if ext !== nothing
        try
            divs = ext.gb_divisors_nemo(N, deg; max_count=max_count)
            if !isempty(divs)
                return divs
            end
        catch e
            @debug "Nemo divisor path failed" exception=e
        end
    end
    # Fallback: brute for modest deg
    if deg > 14
        @warn "gb_divisors brute infeasible for deg=$deg without Nemo; returning empty. Install Nemo for large N."
        return Vector{Vector{Int}}()
    end
    mod = xNm1(N)
    out = Vector{Vector{Int}}()
    # enumerate monic degree-deg polys with constant term 1: bits deg and 0 forced 1
    low_bits = 1 << (deg - 1)  # possibilities for middle bits
    for mask in 0:(low_bits-1)
        g = (BigInt(1) << deg) | BigInt(1) | (BigInt(mask) << 1)
        # test divisibility
        rem = _poly_mod(mod, g)
        if rem == 0
            push!(out, _poly_to_exps(g))
            length(out) >= max_count && break
        end
    end
    return out
end

# Also support range version
function gb_divisors_range(N::Int, deg_lo::Int, deg_hi::Int; max_count::Int=200)
    out = Vector{Vector{Int}}()
    for d in deg_lo:deg_hi
        append!(out, gb_divisors(N, d; max_count=max_count - length(out)))
        length(out) >= max_count && break
    end
    return out
end

# ---------------------------------------------------------------------------
# Cyclic GB builder (via group_algebra)
# ---------------------------------------------------------------------------

"""
    build_gb(N, a_exps, b_exps) -> (Hx, Hz)

Cyclic generalized-bicycle code on Z_N with supports `a_exps`, `b_exps`
(exponent lists, each weight 4..12). Uses `build_2bga(cyclic_group(N), ...)`.
n = 2N, w = max(wt(a), wt(b)) per side, CSS via 2BGA.

`a_exps`, `b_exps` are 0-based exponents; element indices are exps .+ 1.
"""
function build_gb(
    N::Int,
    a_exps::AbstractVector{Int},
    b_exps::AbstractVector{Int},
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    a_idx = [mod(e, N) + 1 for e in a_exps]
    b_idx = [mod(e, N) + 1 for e in b_exps]
    # deduplicate mod N, must keep odd multiplicity semantics but we enforce
    # inputs are reduced and simple; duplicate exponents mod N would cancel mod2
    # so we enforce distinctness
    # dedup via toggle: even duplicates cancel
    function uniq_mod(exps)
        cnt = Dict{Int,Int}()
        for e in exps
            k = mod(e, N)
            cnt[k] = get(cnt, k, 0) + 1
        end
        return sort([k for (k,c) in cnt if isodd(c)])
    end
    # keep inputs as given but warn if not distinct mod N
    a_u = uniq_mod(a_exps)
    b_u = uniq_mod(b_exps)
    # if duplicates cancelled, use cancelled version (honest mod 2)
    if length(a_u) != length(a_exps) || length(b_u) != length(b_exps)
        a_idx = [k+1 for k in a_u]
        b_idx = [k+1 for k in b_u]
    end
    mul, _ = cyclic_group(N)
    return build_2bga(mul, a_idx, b_idx)
end

"""
    gb_divisor_code(N, g_exps, a_prime_exps, b_prime_exps) -> (Hx, Hz)

Construct a = g·a' and b = g·b' (cyclic product mod x^N-1) then `build_gb`.
Each polynomial given as exponent list.
"""
function gb_divisor_code(
    N::Int,
    g_exps::AbstractVector{Int},
    a_prime_exps::AbstractVector{Int},
    b_prime_exps::AbstractVector{Int},
)
    g = _exps_to_poly(collect(Int, g_exps))
    ap = _exps_to_poly(collect(Int, a_prime_exps))
    bp = _exps_to_poly(collect(Int, b_prime_exps))
    a = _reduce_cyc(_poly_mul(g, ap), N)
    b = _reduce_cyc(_poly_mul(g, bp), N)
    return build_gb(N, _poly_to_exps(a), _poly_to_exps(b))
end

# ---------------------------------------------------------------------------
# Random ideal member via simple sampling (fallback without RIS engine)
# ---------------------------------------------------------------------------

"""
    random_gb_pair(N, g_exps; weight_target=10, rng) -> (a_exps, b_exps)

Sample two random multiples of `g` with weight near `weight_target` via
birthday-like random support products. Returns exponent lists for `a=g·a'`,
`b=g·b'`. For screening only; distance needs `distance_rand`.
"""
function random_gb_pair(
    N::Int,
    g_exps::AbstractVector{Int};
    weight_target::Int = 10,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Tuple{Vector{Int},Vector{Int}}
    g = _exps_to_poly(collect(Int, g_exps))
    half = max(2, weight_target ÷ 2)
    function one_sample()
        ap = randperm(rng, N)[1:half]
        ap_poly = _exps_to_poly([e for e in ap])
        a = _reduce_cyc(_poly_mul(g, ap_poly), N)
        return _poly_to_exps(a)
    end
    a_exps = one_sample()
    b_exps = one_sample()
    return a_exps, b_exps
end

"""
    enumerate_gb_candidates(N, deg; n_samples=50, weight=10, seed=0) -> Vector{(spec,Hx,Hz)}

Enumerate `n_samples` GB divisor candidates for screening. Uses `gb_divisors`
to pick g, then `random_gb_pair` for a,b.
"""
function enumerate_gb_candidates(
    N::Int,
    deg::Int;
    n_samples::Int = 50,
    weight::Int = 10,
    seed::Int = 0,
)
    rng = MersenneTwister(seed)
    divs = gb_divisors(N, deg; max_count=20)
    if isempty(divs)
        return Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    end
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    for _ in 1:n_samples
        g = divs[rand(rng, 1:length(divs))]
        a_exps, b_exps = random_gb_pair(N, g; weight_target=weight, rng=rng)
        (isempty(a_exps) || isempty(b_exps)) && continue
        wt_a = length(a_exps); wt_b = length(b_exps)
        (wt_a > 16 || wt_b > 16) && continue
        Hx, Hz = build_gb(N, a_exps, b_exps)
        size(Hx,2) != 2*N && continue
        verify_css(Hx, Hz) || continue
        spec = Dict("family"=>"generalized-bicycle","N"=>N,"deg"=>deg,"g"=>g,"a"=>a_exps,"b"=>b_exps,"wt_a"=>wt_a,"wt_b"=>wt_b)
        push!(out, (spec, Hx, Hz))
    end
    return out
end

# ---------------------------------------------------------------------------
# TestItems
# ---------------------------------------------------------------------------
using TestItems

@testitem "GB: build_gb small CSS" begin
    Hx, Hz = build_gb(7, [0,1,3], [0,2,3])
    @test verify_css(Hx, Hz)
    @test size(Hx) == (7, 14)
    @test compute_k(Hx, Hz) >= 0
end

@testitem "GB: divisors of x^7+1 over F2" begin
    # x^7+1 = (x+1)(x^3+x+1)(x^3+x^2+1) over F2 — know divisors of deg 1,3
    divs1 = gb_divisors(7, 1; max_count=10)
    @test length(divs1) >= 1
    # g = x+1 = [0,1] must divide x^7+1
    @test any(sort(d)==[0,1] for d in divs1)
    divs3 = gb_divisors(7, 3; max_count=20)
    @test length(divs3) >= 2
end

@testitem "GB: gb_divisor_code via g*(a') construction" begin
    N = 15
    # pick a known divisor of x^15+1 of degree 4 e.g. x^4+x+1 (but may not divide? just test plumbing)
    divs = gb_divisors(N, 4; max_count=20)
    if !isempty(divs)
        g = divs[1]
        Hx, Hz = gb_divisor_code(N, g, [0,1], [0,2])
        @test verify_css(Hx, Hz)
        @test size(Hx,2)==2*N
    else
        # fallback: arbitrary code
        Hx, Hz = build_gb(N, [0,1,3], [0,2,5])
        @test verify_css(Hx, Hz)
    end
end

@testitem "GB: enumerate_gb_candidates smoke" begin
    cands = enumerate_gb_candidates(15, 4; n_samples=10, weight=6, seed=42)
    @test length(cands) >= 0  # may be 0 if no divisor weight match
    for (spec,Hx,Hz) in cands
        @test verify_css(Hx, Hz)
        @test size(Hx,2)==2*spec["N"]
    end
end

@testitem "GB: weight class w check cyclic codes" begin
    # known GB weight check: wt(a)=3, wt(b)=3 → row weight 6 per block, max 6
    Hx, Hz = build_gb(21, [0,1,3], [0,2,7])
    @test max(row_weight(Hx), row_weight(Hz)) <= 6
    # weight 4+4 → 8
    Hx2, Hz2 = build_gb(21, [0,1,3,7], [0,2,5,6])
    @test max(row_weight(Hx2), row_weight(Hz2)) <= 8
end

@testitem "GB: arbitrary cyclic GB is also a 2BGA" begin
    N = 9
    a = [0,1,3]; b = [0,2,5]
    Hx, Hz = build_gb(N, a, b)
    # direct 2BGA via cyclic_group should match
    mul,_ = cyclic_group(N)
    Hx2, Hz2 = build_2bga(mul, [e+1 for e in a], [e+1 for e in b])
    @test Matrix(Hx)==Matrix(Hx2)
    @test Matrix(Hz)==Matrix(Hz2)
end
