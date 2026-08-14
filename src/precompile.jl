"""
precompile.jl — PrecompileTools workload for QLDPC.jl

Warms up the heavy paths so `using QLDPC` is fast and the first
`distance_rand` doesn't pay JIT. Covers:
  - BB construction [[72,12,6]]
  - CSS checks + GF(2) rank (compute_k)
  - distance_rand (RIS) — 10 trials small to avoid long precompile
  - threaded variant (nthreads=1 fallback)
  - rank_gf2_fast path

Included from src/QLDPC.jl via `include("precompile.jl")`.
"""

import PrecompileTools

PrecompileTools.@setup_workload begin
    # Put code that generates the workload into @compile_workload
    Hx_f = Hz_f = nothing
    PrecompileTools.@compile_workload begin
        # BB build + CSS — the hot path for search
        Hx_f, Hz_f = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
        verify_css(Hx_f, Hz_f)
        compute_k(Hx_f, Hz_f)
        rank_gf2(Hx_f)
        rank_gf2_fast(Hx_f)
        rref_gf2(Hx_f)
        kernel_basis(Hx_f)
        logical_basis(Hx_f, Hz_f)

        # Distance surrogates — 10 trials keeps precompile < ~2s
        distance_rand(Hx_f, Hz_f; trials = 10, seed = 0)
        distance_rand_threaded(Hx_f, Hz_f; trials = 10, seed = 0, nthreads = 1)
        lightest_logical(Hx_f, Hz_f; trials = 10, seed = 0)
        _search_lightest(Hx_f, Hz_f; trials = 10, seed = 0)
        _search_lightest_threaded(Hx_f, Hz_f; trials = 10, seed = 0, nthreads = 1)

        # CSSCode + fingerprint + efficiency
        c = CSSCode(Hx_f, Hz_f)
        ncode(c); kcode(c); weight(c)
        fingerprint(Hx_f, Hz_f)
        efficiency(72, 12, 6)

        # Bitset primitives
        _nwords(72)
        # rref_bitset workload already covered via distance_rand bitset path
    end
end
