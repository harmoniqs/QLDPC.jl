using Documenter
using Literate

# Generate markdown from literate sources
lit_src = joinpath(@__DIR__, "literate")
lit_out = joinpath(@__DIR__, "src", "generated")
mkpath(lit_out)
for jl in filter(f -> endswith(f, ".jl"), readdir(lit_src; join=true))
    Literate.markdown(jl, lit_out)
end

makedocs(;
    modules=[QLDPC],
    authors="Harmoniqs",
    repo="https://github.com/harmoniqs/QLDPC.jl/blob/{commit}{path}#{line}",
    sitename="QLDPC.jl",
    format=Documenter.HTML(;
        canonical="https://harmoniqs.github.io/QLDPC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Quickstart" => "generated/quickstart.md",
        "API" => "lib.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/harmoniqs/QLDPC.jl",
    devbranch="main",
)
