<div align="center">
  <a href="https://github.com/harmoniqs/QLDPC.jl">
    <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
  </a>
  <a href="https://github.com/harmoniqs/QLDPC.jl/actions/workflows/CI.yml?query=branch%3Amain">
    <img src="https://github.com/harmoniqs/QLDPC.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
  </a>
  <a href="https://codecov.io/gh/harmoniqs/QLDPC.jl">
    <img src="https://codecov.io/gh/harmoniqs/QLDPC.jl/branch/main/graph/badge.svg" alt="Coverage"/>
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
  </a>
</div>

# QLDPC.jl

**Julia toolkit for qLDPC codes — BB, CSS, distance, search, validation (challenge kit).**

`QLDPC.jl` is the Julia companion to the [qLDPC challenge](https://github.com/harmoniqs/qldpc-challenge): a small, auditable toolkit that mirrors the Python `research/kit` (BB construction, CSS checks, GF(2) rank, surrogate distance, search funnel, submission packaging) in idiomatic Julia with `SparseArrays`, `TestItems`, and `Literate` docs.

Quickstart:

```julia
using QLDPC

# [[72,12,6]] BB code (Bravyi et al. gross code)
Hx, Hz = build_bb(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
@assert verify_css(Hx, Hz)
k = compute_k(Hx, Hz)           # 12
d = distance_rand(Hx, Hz; trials=400, seed=0)  # upper bound ≤ 6

# search funnel
recs = screen([(("demo", Hx, Hz))]; min_k=1, trials=200)
```

## Installation

```julia
] add QLDPC
```

Or dev:

```julia
using Pkg; Pkg.develop(path="~/armonia/repos/QLDPC.jl")
```

## Layout

- `src/css.jl` — `verify_css`, `compute_k`, `CSSCode`, GF(2) RREF/rank/kernel/logical_basis
- `src/bb.jl` — `build_bb`, `poly_matrix`, `KNOWN_CODES` (bivariate-bicycle torus)
- `src/surrogate.jl` — `distance_rand`, `lightest_logical` (randomized information set RIS)
- `src/search.jl` — `screen`, `pareto_frontier`, `fingerprint`, `efficiency`
- `src/submit.jl` — `make_submission`, `validate_candidate`, `save_submission`

The **gate stays Python** — `verify/` (schema + CSS + witness + refutation) is the trust anchor. This package is the builder + cheap estimator.

## Performance — Threads + Sysimage (24c)

`distance_rand` has a thread-parallel twin that splits the 400 RIS trials across
cores with thread-local bitset work buffers (`Vector{Matrix{UInt64}}` per thread)
and deterministic perms, so `distance_rand_threaded(...; nthreads=1) == distance_rand(...)`
and on a threaded Julia `distance_rand_threaded` == serial:

```julia
using QLDPC
Hx, Hz = build_bb(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
distance_rand_threaded(Hx, Hz; trials=400, seed=0)  # uses Threads.nthreads() by default
# Distributed fallback: if nprocs() > 1 and only 1 thread, uses pmap over batches
```

On a 10-core M4 (local, `--threads=10`): `n=288, trials=400` goes `0.31s → 0.086s`
(**≈3.6×**, `n=144` ≈1.9×, `n=72` ≈2.1×). On erlich (24c) expect ≈5–7× (linear in
trials until memory bandwidth). See `benchmark/bench_vs_python.jl` for the
`n=72,144,288` table (Julia vs Python `research/kit`).

Precompile / sysimage — `src/precompile.jl` (`PrecompileTools.@setup_workload`)
warms BB 72,12,6 + `distance_rand` (10 trials) so `Pkg.precompile` already cuts
first-call JIT. For max performance (no JIT on `using QLDPC`):

```julia
julia --project=. -e 'using Pkg; Pkg.add("PackageCompiler")'
julia --project=. scripts/build_sysimage.jl          # → QLDPC.so  (~45-60 MB)
julia --sysimage QLDPC.so --project=. -e 'using QLDPC; ...'
```

`Pkg.precompile` is enough for CI; the `QLDPC.so` sysimage is for interactive/
benchmark use. See `scripts/build_sysimage.jl` for options.

## Testing

```julia
julia --project=. -e 'using TestItemRunner; @run_package_tests()'           # threads=1 path also tested
julia --project=. --threads auto -e 'using TestItemRunner; @run_package_tests()'  # threaded
# benchmark smoke (filtered out of CI):
julia --project=. -e 'using TestItemRunner; @run_package_tests filter=ti->:benchmark in ti.tags'
```

## License

MIT — see [LICENSE](LICENSE).

*"Technologies are ways of commandeering nature." — Simone de Beauvoir*
