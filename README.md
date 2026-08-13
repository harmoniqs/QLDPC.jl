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

## Testing

```julia
julia --project=. -e 'using TestItemRunner; @run_package_tests()'
```

## License

MIT — see [LICENSE](LICENSE).

*"Technologies are ways of commandeering nature." — Simone de Beauvoir*
