"""
bench_vs_python.jl — Julia vs Python timing for build_bb + distance_rand

For n=72,144,288 (BB codes from KNOWN_CODES), times Julia's `build_bb` +
`distance_rand` vs Python's `research/kit` (if available) and prints a table.

Run:
  julia --project=. benchmark/bench_vs_python.jl
  julia --project=. --threads auto benchmark/bench_vs_python.jl   # threaded
  julia --project=benchmark benchmark/bench_vs_python.jl          # via benchmark env

The Python side shells out to `python3 -c "..."` importing `research.kit.bb`
and `research.kit.surrogate` (same RIS). If Python kit not on PYTHONPATH,
the Python column shows "—" and Julia times are still reported.

Like the previous profile (research/profile.py), we report median of 3 runs
per (n, trials=400).
"""

using BenchmarkTools
using QLDPC
using Printf

# Known BB codes keyed by n
const CODES = [
    ("[[72,12,6]]",  6,  6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)], 72),
    ("[[144,12,12]]", 12, 6, [(3,0),(0,1),(0,2)], [(0,3),(1,0),(2,0)], 144),
    ("[[288,12,18]]", 12,12, [(3,0),(0,2),(0,7)], [(0,3),(1,0),(2,0)], 288),
]

function time_julia(l, m, A, B; trials=400, seed=0)
    Hx, Hz = build_bb(l, m, A, B)
    # warm up once (JIT)
    distance_rand(Hx, Hz; trials=10, seed=seed)
    # median of 3
    ts = Float64[]
    for _ in 1:3
        t = @elapsed distance_rand(Hx, Hz; trials=trials, seed=seed)
        push!(ts, t)
    end
    return median(ts), size(Hx,2), compute_k(Hx, Hz)
end

function time_julia_threaded(l, m, A, B; trials=400, seed=0)
    Hx, Hz = build_bb(l, m, A, B)
    distance_rand_threaded(Hx, Hz; trials=10, seed=seed, nthreads=1)
    if Threads.nthreads() == 1
        return nothing
    end
    ts = Float64[]
    for _ in 1:3
        t = @elapsed distance_rand_threaded(Hx, Hz; trials=trials, seed=seed)
        push!(ts, t)
    end
    return median(ts)
end

function time_python(l, m, A, B; trials=400, seed=0)
    # Try to run Python kit via shell; return seconds or nothing
    py = Sys.which("python3")
    py === nothing && return nothing
    # Build python snippet that imports research.kit if available
    # We pass terms as Python lists of tuples
    a_str = string(A)
    b_str = string(B)
    # Python code: build_bb + surrogate.distance_rand
    code = """
import time, sys, os
# candidate challenge roots (local checkout next to QLDPC.jl, erlich ~/qldpc-challenge, env QLDPC_CHALLENGE)
_candidates = [
    os.environ.get("QLDPC_CHALLENGE", ""),
    os.path.join(os.getcwd(), "..", "qldpc-challenge"),
    os.path.join(os.getcwd(), "..", "..", "qldpc-challenge"),
    "/home/aaron/qldpc-challenge",
    os.path.expanduser("~/armonia/repos/qldpc-challenge"),
    os.path.expanduser("~/qldpc-challenge"),
    ".",
]
for _r in _candidates:
    if _r and os.path.isdir(_r):
        for _sub in ["research/kit", "verify", ""]:
            _p = os.path.join(_r, _sub) if _sub else _r
            if _p not in sys.path:
                sys.path.insert(0, _p)
try:
    from research.kit.bb import build_bb
    from research.kit.surrogate import distance_rand
except Exception:
    try:
        from bb import build_bb
        from surrogate import distance_rand
    except Exception as e:
        print("PY_NA", e, file=sys.stderr)
        sys.exit(0)
for _ in range(1):
    Hx, Hz = build_bb($l, $m, $a_str, $b_str)
    distance_rand(Hx, Hz, trials=10, seed=$seed)
ts=[]
for _ in range(3):
    Hx, Hz = build_bb($l, $m, $a_str, $b_str)
    t0=time.time()
    d=distance_rand(Hx, Hz, trials=$trials, seed=$seed)
    ts.append(time.time()-t0)
import statistics
print(statistics.median(ts))
"""
    tmp = tempname()
    write(tmp, code)
    try
        out = read(`$py $tmp`, String)
        # parse first float line
        m = match(r"([0-9]+\.[0-9]+)", out)
        m === nothing && return nothing
        return parse(Float64, m.captures[1])
    catch e
        @debug "python timing failed" exception=e
        return nothing
    finally
        try rm(tmp) catch end
    end
end

function main()
    trials = 400
    seed = 0
    println("# Julia vs Python — build_bb + distance_rand (trials=$trials)")
    println("# Threads: $(Threads.nthreads()), Julia $(VERSION), $(Sys.CPU_NAME)")
    println()
    @printf("%-14s %-4s %-6s %-12s %-12s %-10s\n", "code", "n", "k", "julia (s)", "julia-thr(s)", "python (s)")
    println("-"^78)
    for (name, l, m, A, B, n) in CODES
        tj, _, k = time_julia(l, m, A, B; trials=trials, seed=seed)
        tjt = time_julia_threaded(l, m, A, B; trials=trials, seed=seed)
        tp = time_python(l, m, A, B; trials=trials, seed=seed)
        # format
        tj_s = @sprintf("%.3f", tj)
        tjt_s = tjt === nothing ? "—" : @sprintf("%.3f", tjt)
        tp_s = tp === nothing ? "—" : @sprintf("%.3f", tp)
        if tjt !== nothing
            speedup = tj / tjt
            tjt_s *= @sprintf(" (%.1fx)", speedup)
        end
        @printf("%-14s %-4d %-6d %-12s %-12s %-10s\n", name, n, k, tj_s, tjt_s, tp_s)
    end
    println()
    println("Notes:")
    println("  - julia-thr uses distance_rand_threaded with Threads.nthreads()=$(Threads.nthreads()) (per-thread work buffers, deterministic perms).")
    println("  - python column shells out to research/kit; '—' means kit not on PYTHONPATH.")
    println("  - median of 3 runs; first warm-up (JIT) excluded.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
