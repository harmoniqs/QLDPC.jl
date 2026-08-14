@testitem "Aqua quality assurance" tags = [:aqua] begin
    using Aqua, QLDPC
    Aqua.test_all(
        QLDPC;
        stale_deps = (ignore = [:JLD2, :TestItemRunner, :BenchmarkTools, :PackageCompiler],),
        deps_compat = (ignore = [:Test, :Aqua, :JET, :Distributed, :BenchmarkTools, :PackageCompiler, :ChunkSplitters, :OhMyThreads, :Polyester, :StaticArrays],),
    )
end
