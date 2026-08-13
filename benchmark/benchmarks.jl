using BenchmarkTools
using QLDPC

SUITE = BenchmarkGroup()

SUITE["bb"] = BenchmarkGroup()
SUITE["bb"]["build_72"] =
    @benchmarkable build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
SUITE["bb"]["verify_css"] = @benchmarkable begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    verify_css(Hx, Hz)
end

SUITE["distance"] = BenchmarkGroup()
SUITE["distance"]["rand_72_400"] = @benchmarkable begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    distance_rand(Hx, Hz; trials = 400, seed = 0)
end

# Run with: julia --project=benchmark benchmark/benchmarks.jl
if abspath(PROGRAM_FILE) == @__FILE__
    BenchmarkTools.tune!(SUITE)
    results = BenchmarkTools.run(SUITE; verbose = true)
    BenchmarkTools.display(results)
end
