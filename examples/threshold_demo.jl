#!/usr/bin/env julia --project=. --threads=auto
# threshold_demo.jl — BP+OSD threshold curve for a BB code (LDPCDecoders weakdep)
#
# Runs `threshold_estimate` on the [[72,12,6]] and [[144,12,12]] BB codes,
# prints the LER vs p curve, and shows weight-aware ranking vs raw kd²/n.
#
# The decoder path is optional: with `LDPCDecoders.jl` installed the curve
# uses BP+OSD (`QLDPCDecodeExt`); without it the same call falls back to
# syndrome-zero Monte Carlo (lower bound) so the demo always runs.
#
# Usage:
#   julia --project=. examples/threshold_demo.jl
#   julia --project=. --threads=auto examples/threshold_demo.jl
#   julia --project=. -e 'using Pkg; Pkg.add("LDPCDecoders")'  # then re-run for BP+OSD
#
# See also: `src/decode.jl` (`bp_osd_distance`, `threshold_estimate`, `circuit_ler`),
#           `src/search.jl` (`efficiency_weighted`, `geometric_efficiency`, `overhead`).

using QLDPC
using Printf

println("=== QLDPC.jl threshold_demo — BP+OSD (LDPCDecoders weakdep) ===\n")

# Check whether the LDPCDecoders extension is loaded
has_ldpc = Base.get_extension(QLDPC, :QLDPCDecodeExt) !== nothing
println("LDPCDecoders available: $has_ldpc")
if !has_ldpc
    println("  (install with `] add LDPCDecoders` for BP+OSD; using fallback)")
end
println()

# --- Pick BB codes from KNOWN_CODES ---

codes = [
    ("[[72,12,6]]", KNOWN_CODES["[[72,12,6]]"]),
    ("[[144,12,12]]", KNOWN_CODES["[[144,12,12]]"]),
]

# weight-aware ranking table
println("--- Weight-aware ranking ---")
println("code          n    k    d   w   kd²/n   kd²/(n*w)  kd²/(n*w³)  n/k  class")
for (label, p) in codes
    Hx, Hz = build_bb(p.l, p.m, p.A, p.B)
    n = size(Hx, 2)
    k = compute_k(Hx, Hz)
    d = distance_rand(Hx, Hz; trials=400, seed=0)
    w = max(row_weight(Hx), row_weight(Hz))
    eff = efficiency(n, k, d)
    eff_w1 = efficiency_weighted(n, k, d, w; a=1)
    eff_w3 = efficiency_weighted(n, k, d, w; a=3)
    ov = overhead(n, k)
    wc = weight_class(w)
    @printf("%-12s %4d %4d %4d %3d  %6.2f   %6.3f    %6.4f  %5.2f  %s\n",
        label, n, k, d, w, eff, eff_w1, eff_w3, ov, wc)
end

# Synthetic comparison from the issue: [[360,12,24]] w6 vs [[390,82,32]] w16
println("\n--- Synthetic: [[360,12,24]] w6 vs [[390,82,32]] w16 ---")
for (n, k, d, w) in [(360, 12, 24, 6), (390, 82, 32, 16)]
    eff = efficiency(n, k, d)
    eff_w1 = efficiency_weighted(n, k, d, w; a=1)
    eff_w3 = efficiency_weighted(n, k, d, w; a=3)
    @printf("[[%d,%d,%d]] w%d  kd²/n=%.2f  weighted(a=1)=%.2f  weighted(a=3)=%.4f  overhead=%.2f  %s\n",
        n, k, d, w, eff, eff_w1, eff_w3, overhead(n, k), weight_class(w))
end
println("  raw kd²/n favors 390,82; weighted a=3 favors 360,12 (lighter checks win)")

# geometric efficiency example
println("\n--- Geometric efficiency ---")
for (label, p) in codes
    Hx, Hz = build_bb(p.l, p.m, p.A, p.B)
    k = compute_k(Hx, Hz)
    d = distance_rand(Hx, Hz; trials=400, seed=0)
    n = size(Hx, 2)
    # fake planar layout for demo: rho=2, r=5.0 (bilayer tile)
    geo = geometric_efficiency(k, d, n, 2.0, 5.0)
    @printf("%-12s  4kd²/(n rho² r⁴) = %.6f  (k=%d d=%d n=%d rho=2 r=5)\n", label, geo, k, d, n)
end

println("\n--- Threshold curves (BP+OSD or fallback) ---")
for (label, p) in codes
    Hx, Hz = build_bb(p.l, p.m, p.A, p.B)
    k = compute_k(Hx, Hz)
    d_rand = distance_rand(Hx, Hz; trials=400, seed=0)
    d_bp = bp_osd_distance(Hx, Hz; trials=200, seed=0)
    println("\n$label  Hx=$(size(Hx)) Hz=$(size(Hz))  k=$k  d_rand≈$d_rand  d_bp≈$d_bp  $(d_bp == d_rand ? "(agree)" : "(Δ=$(abs(d_bp-d_rand)))")")

    # threshold curve — 5 points, 1000 trials each
    p_range = 0.005:0.005:0.03
    est = threshold_estimate(Hx, Hz; p_range = p_range, trials = 1000, seed = 42)
    println("  p      LER        n_logical/trials")
    for e in est
        @printf("  %.3f  %.4f     %4d/%d\n", e.p, e.ler, e.n_logical, e.n_trials)
    end

    # circuit-level placeholder
    cl = circuit_ler(Hx, Hz; p = 0.01, trials = 500, seed = 1)
    println("  circuit_ler(p=0.01): ler=$(cl.ler)  note: $(cl.note)")

    # pareto demo
    recs = [
        (spec=label, n=size(Hx,2), k=k, d=d_rand, w=max(row_weight(Hx), row_weight(Hz)), efficiency=efficiency(size(Hx,2), k, d_rand), fingerprint="demo"),
    ]
    front = pareto_frontier(recs)
    front_w = pareto_frontier_weighted(recs; a=1)
    println("  pareto_frontier: $(length(front)) code(s); weighted: $(length(front_w)) code(s)")
end

println("\nDone. With LDPCDecoders, `bp_osd_distance` and `threshold_estimate` use BP+OSD;")
println("without it they fall back to RIS / syndrome-zero counting (same API).")
println("For full circuit-level LER, wire Stim/QuantumClifford + LDPCDecoders fault-tolerant BP.")
