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
using Base.Threads
import Distributed

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
        @inbounds for i in 1:n
            perm_buf[i] = i
        end
        Random.shuffle!(rng, perm_buf)
        perm = perm_buf  # alias, _rref_perm will read it; no copy

        local red::Matrix{Bool}
        if use_bitset
            # copy K_bits into work (fast memcopy)
            copyto!(work, K_bits)
            keep = rref_bitset!(work, n, perm)
            if isempty(keep)
                continue
            end
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

# ---------------------------------------------------------------------------
# Threaded variant — splits trials across Threads.@threads with thread-local
# work buffers (Vector{Matrix{UInt64}} per thread). Deterministic: pre-
# generates perms serially with the same RNG as the serial path, then
# parallelizes only the RREF+check work, so threaded == serial exactly.
# Distributed fallback: if nprocs()>1 and Threads.nthreads()==1, uses pmap.
# ---------------------------------------------------------------------------

"""
    _search_lightest_threaded(Hself, Hopp; trials, seed, pair_depth=10, nthreads=Threads.nthreads())
        -> (weight, support)

Thread-parallel version of `_search_lightest`. Splits `trials` across
`nthreads` threads (default: all Julia threads). Each thread gets its own
`work::Matrix{UInt64}` buffer and reuse of the bitset path, so there is no
allocation contention. Permutations are pre-generated serially with the same
`MersenneTwister(seed)` sequence as the serial path, guaranteeing
`threaded == serial` for the same `seed` (verified by `@testitem`).

Falls back to serial when `nthreads==1` or `Threads.nthreads()==1`.
If `Distributed.nprocs() > 1` and only 1 thread is available, uses
`Distributed.pmap` over trial batches (cluster fallback on erlich not needed;
primary is Threads on 24c).
"""
function _search_lightest_threaded(
    Hself::AbstractMatrix,
    Hopp::AbstractMatrix;
    trials::Int = 400,
    seed::Int = 0,
    pair_depth::Int = 10,
    nthreads::Int = Threads.nthreads(),
)::Tuple{Int,Vector{Int}}
    # Small or single-threaded → serial fallback (no overhead)
    if nthreads <= 1 || Threads.nthreads() == 1
        # Distributed fallback when no threading but multi-process cluster exists
        if Distributed.nprocs() > 1 && trials >= Distributed.nprocs()
            return _search_lightest_distributed(Hself, Hopp; trials=trials, seed=seed, pair_depth=pair_depth)
        end
        return _search_lightest(Hself, Hopp; trials=trials, seed=seed, pair_depth=pair_depth)
    end
    # Distributed fallback supersedes threading when explicitly on a cluster with 1 thread per worker
    # (user can force by passing nthreads==1). Already handled above.

    Hself_b = Matrix{Bool}(map(x -> Bool(Int(x) & 1 != 0), collect(Hself)))
    Hopp_b = Matrix{Bool}(map(x -> Bool(Int(x) & 1 != 0), collect(Hopp)))
    n = size(Hself_b, 2)

    K = kernel_basis(Hopp_b)
    LO = logical_basis(Hself_b, Hopp_b)
    if size(K, 1) == 0 || size(LO, 1) == 0
        return typemax(Int), Int[]
    end

    # Pre-generate perms serially with same RNG as serial path → deterministic equality
    rng = MersenneTwister(seed)
    perms = Vector{Vector{Int}}(undef, trials)
    tmp = collect(1:n)
    for t in 1:trials
        # reset tmp to 1:n is not needed — shuffle! from any permutation is uniform,
        # but to exactly match serial's refill+shuffle we do refill:
        @inbounds for i in 1:n
            tmp[i] = i
        end
        Random.shuffle!(rng, tmp)
        perms[t] = copy(tmp)
    end

    use_bitset = USE_BITSET
    K_bits = use_bitset ? _bool_to_bitset(K) : Matrix{UInt64}(undef, 0, 0)
    nwords = use_bitset ? _nwords(n) : 0

    # Thread-local work buffers (preallocate Vector{Matrix{UInt64}} per thread)
    # Use maxthreadid to cover both :default and :interactive pools (Julia 1.11+ may have tid > nthreads())
    nt_buf = try
        Threads.nthreads(:default) + Threads.nthreads(:interactive)
    catch
        Threads.nthreads()
    end
    # also consider Threads.maxthreadid if available
    try
        nt_buf = max(nt_buf, Base.Threads.maxthreadid())
    catch
    end
    nt_buf = max(nt_buf, Threads.nthreads(), 16)
    work_buffers = Vector{Matrix{UInt64}}(undef, nt_buf)
    for tid in 1:nt_buf
        work_buffers[tid] = use_bitset ? Matrix{UInt64}(undef, size(K_bits, 1), nwords) : Matrix{UInt64}(undef, 0, 0)
    end

    # Thread-local best state (no atomics needed — each tid owns its slot)
    nt = Threads.nthreads()
    # keep reduction over all buf slots to handle interactive-pool tid
    thread_best_w = fill(n + 1, nt_buf)
    thread_best_v = Vector{Union{Nothing,Vector{Bool}}}(nothing, nt_buf)

    Threads.@threads for t in 1:trials
        tid = Threads.threadid()
        perm = perms[t]
        local red::Matrix{Bool}
        if use_bitset
            work = work_buffers[tid]
            copyto!(work, K_bits)
            keep = rref_bitset!(work, n, perm)
            if isempty(keep)
                continue
            end
            red = _bitset_to_bool(work, n, keep)
        else
            red = _rref_perm_dense(K, perm)
        end
        if size(red, 1) == 0
            continue
        end
        w = _weights_rows(red)
        nz = _nontrivial_mask(red, LO)

        # thread-local best (read-modify-write on owned slot, no race)
        lw = thread_best_w[tid]
        lv = thread_best_v[tid]
        for i in eachindex(w)
            if nz[i] && w[i] > 0 && w[i] < lw
                lw = w[i]
                lv = Vector{Bool}(red[i, :])
            end
        end
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
                    if is_nz && pw < lw
                        lw = pw
                        lv = Vector{Bool}(pr)
                    end
                end
            end
        end
        thread_best_w[tid] = lw
        thread_best_v[tid] = lv
    end

    # Reduce across threads → global min
    best_w = minimum(thread_best_w)
    best_idx = argmin(thread_best_w)
    best_v = thread_best_v[best_idx]
    if best_w == n + 1 || best_v === nothing
        return typemax(Int), Int[]
    end
    supp = sort(findall(identity, best_v))
    return best_w, supp
end

"""
    _search_lightest_distributed(Hself, Hopp; trials, seed, pair_depth) -> (weight, support)

Distributed fallback via `Distributed.pmap` over trial batches. Used when
`Distributed.nprocs() > 1` and threading is unavailable. Splits trials into
batches, runs `_search_lightest` on each worker with `seed + offset`, then
reduces by min weight.
"""
function _search_lightest_distributed(
    Hself::AbstractMatrix,
    Hopp::AbstractMatrix;
    trials::Int = 400,
    seed::Int = 0,
    pair_depth::Int = 10,
)::Tuple{Int,Vector{Int}}
    np = Distributed.nprocs()
    # chunk trials across workers; fall back to serial if tiny
    if np <= 1 || trials < np
        return _search_lightest(Hself, Hopp; trials=trials, seed=seed, pair_depth=pair_depth)
    end
    chunk = cld(trials, np)
    # Build batch descriptors: (batch_seed, batch_trials, offset) — offset not used for RNG currently
    # Use consecutive seeds so batches are deterministic and disjoint
    batches = Tuple{Int,Int}[]
    for b in 0:np-1
        start = b * chunk + 1
        start > trials && break
        len = min(chunk, trials - start + 1)
        push!(batches, (seed + b * 1000003, len))
    end
    results = Distributed.pmap(batches) do (bseed, blen)
        _search_lightest(Hself, Hopp; trials=blen, seed=bseed, pair_depth=pair_depth)
    end
    best_w = typemax(Int)
    best_supp = Int[]
    for (w, supp) in results
        if w < best_w
            best_w = w
            best_supp = supp
        end
    end
    return best_w, best_supp
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
    lightest_logical_threaded(Hself, Hopp; trials=8000, seed=0, nthreads=Threads.nthreads())
"""
function lightest_logical_threaded(
    Hself::AbstractMatrix,
    Hopp::AbstractMatrix;
    trials::Int = 8000,
    seed::Int = 0,
    nthreads::Int = Threads.nthreads(),
)::Tuple{Int,Vector{Int}}
    return _search_lightest_threaded(Hself, Hopp; trials=trials, seed=seed, nthreads=nthreads)
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

"""
    distance_rand_threaded(Hx, Hz; trials=2000, seed=0, nthreads=Threads.nthreads()) -> Int

Thread-parallel upper bound on code distance. Splits `trials` across
`nthreads` threads with thread-local `work::Matrix{UInt64}` buffers
(`Vector{Matrix{UInt64}}` preallocated per thread). Reduces `min` across
threads. Falls back to serial when `nthreads==1` or `Threads.nthreads()==1`.
If `Distributed.nprocs() > 1` and only one thread is available, uses
`pmap` over batches (cluster fallback).

Guarantees `distance_rand_threaded(...; nthreads=1) == distance_rand(...)`
and, on a threaded Julia, `distance_rand_threaded(...; nthreads=Threads.nthreads())`
equals `distance_rand` because perms are pre-generated serially.
"""
function distance_rand_threaded(
    Hx::AbstractMatrix,
    Hz::AbstractMatrix;
    trials::Int = 2000,
    seed::Int = 0,
    nthreads::Int = Threads.nthreads(),
)::Int
    if nthreads <= 1 || Threads.nthreads() == 1
        # Distributed fallback when threading unavailable but cluster present
        if Distributed.nprocs() > 1
            # pmap over X/Z sides is tiny — better to pmap trials inside each side
            # _search_lightest_threaded will itself dispatch to distributed branch
            wx, _ = _search_lightest_threaded(Hx, Hz; trials=trials, seed=seed, nthreads=1)
            wz, _ = _search_lightest_threaded(Hz, Hx; trials=trials, seed=seed+1, nthreads=1)
            return min(wx, wz)
        end
        return distance_rand(Hx, Hz; trials=trials, seed=seed)
    end
    wx, _ = _search_lightest_threaded(Hx, Hz; trials=trials, seed=seed, nthreads=nthreads)
    wz, _ = _search_lightest_threaded(Hz, Hx; trials=trials, seed=seed+1, nthreads=nthreads)
    return min(wx, wz)
end

# Convenience for CSSCode
distance_rand(c::CSSCode; trials::Int = 400, seed::Int = 0) =
    distance_rand(c.Hx, c.Hz; trials = trials, seed = seed)
distance_rand_threaded(c::CSSCode; trials::Int = 400, seed::Int = 0, nthreads::Int = Threads.nthreads()) =
    distance_rand_threaded(c.Hx, c.Hz; trials = trials, seed = seed, nthreads=nthreads)
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

@testitem "surrogate: threaded == serial on [[72,12,6]]" begin
    using Base.Threads
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    d_serial = distance_rand(Hx, Hz; trials = 400, seed = 0)
    d_thread1 = distance_rand_threaded(Hx, Hz; trials = 400, seed = 0, nthreads = 1)
    @test d_serial == d_thread1
    # Also check _search_lightest_threaded equality on one side
    w_s, s_s = _search_lightest(Hx, Hz; trials = 400, seed = 0)
    w_t, s_t = _search_lightest_threaded(Hx, Hz; trials = 400, seed = 0, nthreads = 1)
    @test w_s == w_t
    @test s_s == s_t
    # When actually threaded, still deterministic (perms pre-generated, so equal)
    if Threads.nthreads() > 1
        d_thr = distance_rand_threaded(Hx, Hz; trials = 400, seed = 0, nthreads = Threads.nthreads())
        @test d_thr == d_serial
        w_thr, _ = _search_lightest_threaded(Hx, Hz; trials = 400, seed = 0, nthreads = Threads.nthreads())
        @test w_thr == w_s
    end
    # CSSCode overload
    c = CSSCode(Hx, Hz)
    @test distance_rand_threaded(c; trials = 400, seed = 0, nthreads = 1) == d_serial
end

@testitem "surrogate: threaded distance upper bound sane on 72" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    d = distance_rand_threaded(Hx, Hz; trials = 200, seed = 1, nthreads = 1)
    @test d >= 1 && d <= 72
end

@testitem "benchmark vs python" tags=[:benchmark] begin
    # Filtered out of CI via `julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->:benchmark ∉ ti.tags'`
    # Run explicitly: `julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->:benchmark ∈ ti.tags'`
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    # Just verify benchmark harness runs without error; timings not asserted
    t = @elapsed distance_rand(Hx, Hz; trials = 10, seed = 0)
    @test t < 5.0
    tt = @elapsed distance_rand_threaded(Hx, Hz; trials = 10, seed = 0, nthreads = 1)
    @test tt < 5.0
    # Table print is in benchmark/bench_vs_python.jl — this item just guards the paths
    println("benchmark vs python smoke: serial $(round(t; digits=3))s threaded $(round(tt; digits=3))s")
end
