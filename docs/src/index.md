```@meta
CurrentModule = QLDPC
```

# QLDPC.jl

Julia toolkit for qLDPC codes — BB, CSS, distance, search, validation (challenge kit).

Mirrors `research/kit` (BB, CSS, surrogate, search, submit) in idiomatic Julia with `SparseArrays`.

## Quickstart

```julia
using QLDPC
Hx, Hz = build_bb(6, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)])
@assert verify_css(Hx, Hz)
compute_k(Hx, Hz)  # 12
distance_rand(Hx, Hz; trials=400, seed=0)  # upper bound ≤ 6
```

See `Quickstart` (generated from `docs/literate/quickstart.jl`) and `API`.

