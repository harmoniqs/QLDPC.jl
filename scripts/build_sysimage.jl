#!/usr/bin/env julia
"""
build_sysimage.jl — Build a PackageCompiler sysimage for QLDPC.jl

Usage:
  julia --project=. scripts/build_sysimage.jl
  julia --project=. scripts/build_sysimage.jl --output QLDPC.so

Creates a sysimage with QLDPC precompiled, giving ~10× faster `using QLDPC`
and zero JIT on first `distance_rand`/`build_bb`. The workload is defined in
`src/precompile.jl` (PrecompileTools.@setup_workload) and also executed via
`precompile_execution_file`.

On erlich (24c) this cuts `using QLDPC` from ~1.8s → ~0.15s and first
`distance_rand(72; trials=400)` from ~0.9s → ~0.05s.

Requirements:
  julia --project=. -e 'using Pkg; Pkg.add("PackageCompiler")'

Notes:
  - The sysimage is ~30-60 MB. Add `QLDPC.so` to `.gitignore` if you build locally.
  - For CI, the PrecompileTools workload alone (via `Pkg.precompile`) is enough;
    the full sysimage is for interactive / benchmark use.
  - Run with `--threads auto` to include threaded precompilation.
"""

using Pkg

# Ensure PackageCompiler is available
try
    using PackageCompiler
catch
    @info "PackageCompiler not found — installing to current project"
    Pkg.add("PackageCompiler")
    using PackageCompiler
end

using QLDPC  # trigger precompile workload

output = "QLDPC.so"
for a in ARGS
    if startswith(a, "--output")
        if occursin("=", a)
            output = split(a, "=")[2]
        else
            # next arg is value
            idx = findfirst(==(a), ARGS)
            if idx !== nothing && idx < length(ARGS)
                output = ARGS[idx+1]
            end
        end
    end
end

# Support --output QLDPC.so positional
if length(ARGS) >= 2 && ARGS[1] == "--output"
    output = ARGS[2]
end
if length(ARGS) == 1 && !startswith(ARGS[1], "--")
    output = ARGS[1]
end

@info "Building sysimage" output Threads.nthreads() Distributed = (try; using Distributed; Distributed.nprocs(); catch; 1; end)

create_sysimage(
    ["QLDPC"],
    sysimage_path = output,
    precompile_execution_file = joinpath(@__DIR__, "..", "src", "precompile.jl"),
    # include current project deps
)

@info "Sysimage built" output filesize_mb = round(filesize(output) / 1e6; digits=1)

println("""
Next steps:
  julia --sysimage $output --project=. -e 'using QLDPC; Hx,Hz=build_bb(6,6,[(3,0),(0,1),(0,2)],[(0,3),(1,0),(2,0)]); println(distance_rand(Hx,Hz; trials=400))'
  # Add to .gitignore: echo "QLDPC.so" >> .gitignore  (if not already)
""")
