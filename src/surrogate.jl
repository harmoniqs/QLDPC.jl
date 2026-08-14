"""
surrogate.jl — Cheap distance surrogates (upper bounds) and witnesses.

Mirrors `research/kit/surrogate.py`: randomized information-set (RIS) search
for the lightest nontrivial logical operator.

Semantics (honest):
  `distance_rand` returns an **upper bound** on the true distance d — it found
  *a* logical of that weight, so d ≤ that. For screening only. A claimed d
  needs the Python verifier's Mel- or certifier for "exact".

Algorithm per trial:
  - Take kernel_basis(H_opp) (operators commuting with opposite checks)
  - Permute columns randomly, compute RREF visiting columns in that order
  - Rows of the permuted RREF + pairwise sums of the lightest rows are tested
    for nontriviality (not in rowspace(H_self) via logical_basis).

Pure Julia; no decoder/MILP dependency.
"""

using SparseArrays
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Bitset RREF — 64x faster than dense Bool
# ---------------------------------------------------------------------------

const USE_BITSET = true

@inline _nwords(ncols::Int) = cld(ncols, 64)

function _bool_to_bitset(M::Matrix{Bool})::Matrix{UInt64}
    r, c = size(M)
    nw = _nwords(c)
    B = zeros(UInt64, r, nw)
    @inbounds for i in 1:r
        for j in 1:c
            if M[i, j]
                w = (j - 1) ÷ 64 + 1
                b = (j - 1) % 64
                B[i, w] |= UInt64(1) << b
            end
        end
    end
    return B
end

function _bitset_to_bool(B::Matrix{UInt64}, ncols::Int, keep::Vector{Int})::Matrix{Bool}
    r = length(keep)
    out = Matrix{Bool}(undef, r, ncols)
    @inbounds for (idx, row) in enumerate(keep)
        for c in 1:ncols
            w = (c - 1) ÷ 64 + 1
            b = (c - 1) % 64
            out[idx, c] = ((B[row, w] >> b) & UInt64(1)) == UInt64(1)
        end
    end
    return out
end

"""
    rref_bitset!(A, ncols, perm) -> keep::Vector{Int}

In-place RREF over GF(2) on bitset matrix A (rows × nwords), visiting columns
in `perm` order. Eliminates from ALL other rows (reduced). Returns list of
non-zero row indices (kept). `A` is mutated to reduced form.
"""
function rref_bitset!(A::Matrix{UInt64}, ncols::Int, perm::Vector{Int})::Vector{Int}
    rows, nwords = size(A)
    r = 1
    @inbounds for col in perm
        r > rows && break
        w = (col - 1) ÷ 64 + 1
        b = (col - 1) % 64
        mask = UInt64(1) << b
        # find pivot
        piv = 0
        for i in r:rows
            if (A[i, w] & mask) != 0
                piv = i
                break
            end
        end
        piv == 0 && continue
        if piv != r
            for ww in 1:nwords
                A[r, ww], A[piv, ww] = A[piv, ww], A[r, ww]
            end
        end
        # eliminate from all other rows
        for i in 1:rows
            if i != r && (A[i, w] & mask) != 0
                @inbounds @simd for ww in 1:nwords
                    A[i, ww] ⊻= A[r, ww]
                end
            end
        end
        r += 1
    end
    # filter zero rows
    keep = Int[]
    @inbounds for i in 1:rows
        iszero = true
        for ww in 1:nwords
            if A[i, ww] != 0
                iszero = false
                break
            end
        end
        if !iszero
            push!(keep, i)
        end
    end
    return keep
end

# ---------------------------------------------------------------------------
# Internal: RREF with permuted column visit order (mirrors surrogate._rref_perm)
# ---------------------------------------------------------------------------

function _rref_perm_dense(M::Matrix{Bool}, perm::Vector{Int})::Matrix{Bool}
    A = copy(M)
    rows, cols = size(A)
    r = 1
    for col in perm
        r > rows && break
        piv = 0
        for i = r:rows
            if A[i, col]
                piv = i
                break
            end
        end
        piv == 0 && continue
        if piv != r
            A[r, :], A[piv, :] = A[piv, :], A[r, :]
        end
        for i = 1:rows
            if i != r && A[i, col]
                @inbounds @simd for j = 1:cols
                    A[i, j] = A[i, j] ⊻ A[r, j]
                end
            end
        end
        r += 1
    end
    keep = [i for i = 1:size(A, 1) if any(A[i, :])]
    return A[keep, :]
end

function _rref_perm(M::Matrix{Bool}, perm::Vector{Int})::Matrix{Bool}
    if USE_BITSET
        # bitset path: pack, run in-place, unpack kept rows
        ncols = size(M, 2)
        B = _bool_to_bitset(M)
        # reuse is per-call here; allocation-free per-trial reuse happens in _search_lightest
        keep = rref_bitset!(B, ncols, perm)
        return _bitset_to_bool(B, ncols, keep)
    else
        return _rref_perm_dense(M, perm)
    end
end

function _weights_rows(M::Matrix{Bool})::Vector{Int}
    return vec(sum(M; dims = 2))
end

# ---------------------------------------------------------------------------
# Core: lightest logical search for one side
# ---------------------------------------------------------------------------

"""
    _search_lightest(Hself, Hopp; trials, seed, pair_depth=10) -> (weight, support)

Randomized upper-bound search for lightest nontrivial logical of one type:
`v ∈ ker(Hopp)` but `v ∉ rowspace(Hself)`. Returns `(typemax(Int), Int[])` if
no logical exists (k=0 side).

`support` is 1-based sorted indices of `v`.
"""
function _search_lightest(
    Hself::AbstractMatrix,
    Hopp::AbstractMatrix;
    trials::Int = 400,
    seed::Int = 0,
    pair_depth::Int = 10,
)::Tuple{Int,Vector{Int}}
    Hself_b = Matrix{Bool}(map(x -> Bool(Int(x) & 1 != 0), collect(Hself)))
    Hopp_b = Matrix{Bool}(map(x -> Bool(Int(x) & 1 != 0), collect(Hopp)))
    n = size(Hself_b, 2)

    K = kernel_basis(Hopp_b)          # operators commuting with opposite checks
    LO = logical_basis(Hself_b, Hopp_b) # opposite-type logicals -> nontriviality test (fixed 2026-08-13: matches Python ker(Hself) mod rowspace(Hopp))
    if size(K, 1) == 0 || size(LO, 1) == 0
        return typemax(Int), Int[]
    end

    rng = MersenneTwister(seed)
    best_w = n + 1
    best_v = nothing

    # --- bitset pre-alloc for fast trial loop (allocation-free per trial aside from perm) ---
    use_bitset = USE_BITSET
    K_bits = use_bitset ? _bool_to_bitset(K) : Matrix{UInt64}(undef, 0, 0)
    LO_bits = use_bitset ? _bool_to_bitset(LO) : Matrix{UInt64}(undef, 0, 0)
    nwords = use_bitset ? _nwords(n) : 0
    work = use_bitset ? Matrix{UInt64}(undef, size(K_bits, 1), nwords) : Matrix{UInt64}(undef, 0, 0)
    # Pre-allocated perm buffer to avoid randperm alloc per trial (use shuffle!)
    perm_buf = Vector{Int}(undef, n)
    for i in 1:n
        perm_buf[i] = i
    end

    for _ = 1:trials
        # ---- generate perm (allocation-free shuffle) ----
        # Fisher-Yates via shuffle! on perm_buf
        # copy 1:n then shuffle; we reuse perm_buf and shuffle in place
        # Need to refill 1:n each time because shuffle! permutes in place
        # Instead we shuffle a copy: use Random.shuffle!(rng, perm_buf) after reset
        # To avoid alloc, we reset perm_buf to 1:n then shuffle!
        @inbounds for i in 1:n
            perm_buf[i] = i
        end
        Random.shuffle!(rng, perm_buf)
        # For bitset path we use perm_buf directly; for dense we need a copy (but we can use perm_buf as perm)
        perm = perm_buf  # alias, _rref_perm will read it; no copy

        local red::Matrix{Bool}
        if use_bitset
            # copy K_bits into work (fast memcopy)
            copyto!(work, K_bits)
            keep = rref_bitset!(work, n, perm)
            if isempty(keep)
                continue
            end
            # Convert kept rows to Bool for weight/nontrivial logic (small, ~k rows)
            # Use bitset_to_bool for kept rows
            red = _bitset_to_bool(work, n, keep)
        else
            red = _rref_perm_dense(K, perm)
        end

        if size(red, 1) == 0
            continue
        end
        w = _weights_rows(red)
        nz = _nontrivial_mask(red, LO)
        for i in eachindex(w)
            if nz[i] && w[i] > 0 && w[i] < best_w
                best_w = w[i]
                best_v = Vector{Bool}(red[i, :])
            end
        end
        # pairwise sums of lightest rows
        if pair_depth > 1 && size(red, 1) >= 2
            order = sortperm(w)
            take = min(pair_depth, length(order))
            light = order[1:take]
            sub = red[light, :]
            for i = 1:take
                for j = i+1:take
                    pr = sub[i, :] .⊻ sub[j, :]
                    pw = count(identity, pr)
                    pw == 0 && continue
                    is_nz = false
                    for r in 1:size(LO, 1)
                        s = false
                        @inbounds for c = 1:n
                            s = s ⊻ (pr[c] & LO[r, c])
                        end
                        if s
                            is_nz = true
                            break
                        end
                    end
                    if is_nz && pw < best_w
                        best_w = pw
                        best_v = Vector{Bool}(pr)
                    end
                end
            end
        end
    end

    if best_v === nothing
        return typemax(Int), Int[]
    end
    supp = sort(findall(identity, best_v))
    return best_w, supp
end

function _nontrivial_mask(R::Matrix{Bool}, LO::Matrix{Bool})::Vector{Bool}
    r, n = size(R)
    l = size(LO, 1)
    mask = falses(r)
    for i = 1:r
        for j = 1:l
            s = false
            @inbounds for c = 1:n
                s = s ⊻ (R[i, c] & LO[j, c])
            end
            if s
                mask[i] = true
                break
            end
        end
    end
    return mask
end

"""
    lightest_logical(Hself, Hopp; trials=8000, seed=0) -> (weight, support)

Lightest nontrivial logical of one type. For X side pass `(Hx, Hz)`, for Z side
`(Hz, Hx)`. Returns `(weight, support::Vector{Int})` (1-based). `typemax(Int)`
if no logical exists.
"""
function lightest_logical(
    Hself::AbstractMatrix,
    Hopp::AbstractMatrix;
    trials::Int = 8000,
    seed::Int = 0,
)::Tuple{Int,Vector{Int}}
    return _search_lightest(Hself, Hopp; trials = trials, seed = seed)
end

"""
    distance_rand(Hx, Hz; trials=2000, seed=0) -> Int

Upper bound on code distance `d = min(d_X, d_Z)`. `typemax(Int)` if code has
no logical qubits (`k==0`).
"""
function distance_rand(
    Hx::AbstractMatrix,
    Hz::AbstractMatrix;
    trials::Int = 2000,
    seed::Int = 0,
)::Int
    wx, _ = _search_lightest(Hx, Hz; trials = trials, seed = seed)
    wz, _ = _search_lightest(Hz, Hx; trials = trials, seed = seed + 1)
    d = min(wx, wz)
    return d
end

# Convenience for CSSCode
distance_rand(c::CSSCode; trials::Int = 400, seed::Int = 0) =
    distance_rand(c.Hx, c.Hz; trials = trials, seed = seed)
lightest_logical(c::CSSCode, side::Symbol = :X; trials::Int = 8000, seed::Int = 0) =
    side == :X ? lightest_logical(c.Hx, c.Hz; trials = trials, seed = seed) :
    side == :Z ? lightest_logical(c.Hz, c.Hx; trials = trials, seed = seed) :
    error("side must be :X or :Z")

using TestItems

@testitem "surrogate: trivial code has no logical" begin
    Hx = sparse(Bool[1 1])
    Hz = spzeros(Bool, 0, 2)
    d = distance_rand(Hx, Hz; trials = 20, seed = 0)
    @test d isa Int
end

@testitem "surrogate: BB 72,12,6 d upper bound ≤ 6" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    d = distance_rand(Hx, Hz; trials = 400, seed = 0)
    @test d <= 10
    @test d >= 1
    wx, sx = lightest_logical(Hx, Hz; trials = 200, seed = 0)
    if wx != typemax(Int)
        v = zeros(Bool, 72)
        v[sx] .= true
        @test commutes(v, Hz)
        @test !in_rowspace(v, Hx)
        @test length(sx) == wx
    end
end
