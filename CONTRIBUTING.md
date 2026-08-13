# Contributing to QLDPC.jl

We welcome contributions! Practices follow [SciML ColPrac](https://github.com/SciML/ColPrac) and the harmoniqs style (Piccolo.jl / DirectTrajOpt.jl).

## Development Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Tests

Tests live **in the same file as the code** via [TestItems.jl](https://github.com/JuliaTesting/TestItems.jl):

```julia
@testitem "verify CSS" begin
    Hx, Hz = build_bb(6,6,[(3,0),(0,1),(0,2)],[(0,3),(1,0),(2,0)])
    @test verify_css(Hx, Hz)
end
```

Run:

```bash
julia --project=. -e 'using TestItemRunner; @run_package_tests()'
julia --project=. -e 'using TestItemRunner; @run_package_tests(filter=t->:aqua in t.tags)'
```

## Formatting

```julia
using JuliaFormatter; format(".")
```

CI fails if not formatted.

## Docs

```bash
julia --project=docs docs/make.jl
```

Literate sources live in `docs/literate/`.

