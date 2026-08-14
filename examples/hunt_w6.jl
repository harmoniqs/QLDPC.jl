#!/usr/bin/env julia --project=. --threads=24
# hunt_w6.jl — BB unrestricted w6 260-360 threaded hunter (Julia, 24c)
#
# Targets the same objective as research/candidates/bb_unrestricted_w6.py:
#   beats [[360,12,24]]  kd²/n = 19.2   (unrestricted x w6, l,m 12..18)
#
# Uses QLDPC.sample_bb + distance_rand_threaded (Threads.nthreads() = 24)
# Expect 5-7x speedup vs Python surrogate (see benchmark/bench_vs_python.jl):
#   n=288 trials=400:  python 4.1s → julia serial 0.18s (22x) → julia-thr 0.025s (7x more)
#
# Run:
#   julia --project=. --threads=24 examples/hunt_w6.jl              # forever
#   julia --project=. --threads=24 examples/hunt_w6.jl --once       # one cycle (smoke)
#   QLDPC_CHALLENGE=~/armonia/repos/qldpc-challenge julia --threads=24 --project=. examples/hunt_w6.jl
#
# Validates via QLDPC.validate_candidate + optional Python gate
#   (if qldpc-challenge/verify/validate_candidate.py exists, shells to it;
#    otherwise Julia-only checks). Saves advancing candidates to
#   research/candidates/bb-w6-<n>-<k>-<d>-l<l>m<m>-c<cycle>.json
#
# Environment: QLDPC_CHALLENGE (challenge root), QLDPC_JL (this repo),
# JULIA_PROJECT, Threads.nthreads().

using QLDPC
using Random
using Printf
using JSON
using Dates

const TARGET_EFF = 19.2
const L_RANGE = (12, 18)
const M_RANGE = (12, 18)
const BATCH = 500
const TRIALS_SCREEN = 600
const TRIALS_CONFIRM = 800
const MIN_K = 8
const MIN_D_SCREEN = 12

# --- challenge root discovery (for saving + Python validation) ---
function find_challenge_root()::Union{String,Nothing}
    candidates = String[]
    for p in (
        get(ENV, "QLDPC_CHALLENGE", ""),
        joinpath(pwd(), "..", "qldpc-challenge"),
        joinpath(@__DIR__, "..", "..", "qldpc-challenge"),
        expanduser("~/armonia/repos/qldpc-challenge"),
        expanduser("~/qldpc-challenge"),
        "/home/aaron/qldpc-challenge",
        pwd(),
    )
        isempty(p) && continue
        ap = abspath(p)
        if isdir(joinpath(ap, "verify")) && isdir(joinpath(ap, "research"))
            return ap
        end
        push!(candidates, ap)
    end
    return nothing
end

const CHALLENGE_ROOT = find_challenge_root()
const CANDIDATES_DIR = CHALLENGE_ROOT === nothing ? joinpath(pwd(), "research", "candidates") : joinpath(CHALLENGE_ROOT, "research", "candidates")
const VERIFY_DIR = CHALLENGE_ROOT === nothing ? nothing : joinpath(CHALLENGE_ROOT, "verify")

# sample_bb with l,m range (not in QLDPC.sample_bb which fixes l,m)
function sample_bb_range(l_range, m_range, n_samples; weight=6, rng=Random.GLOBAL_RNG)
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    seen_fp = Set{String}()
    for _ in 1:n_samples
        l = rand(rng, l_range[1]:l_range[2])
        m = rand(rng, m_range[1]:m_range[2])
        all_terms = [(a, b) for a in 0:l-1 for b in 0:m-1]
        shuffle!(rng, all_terms)
        A_terms = all_terms[1:3]
        B_terms = all_terms[4:6]
        Hx, Hz = build_bb(l, m, A_terms, B_terms)
        fp = fingerprint(Hx, Hz)
        fp in seen_fp && continue
        push!(seen_fp, fp)
        spec = Dict("l"=>l, "m"=>m, "A"=>A_terms, "B"=>B_terms, "family"=>"bb")
        # Also accept "A_terms"/"B_terms" alias for Python compat
        spec["A_terms"] = A_terms
        spec["B_terms"] = B_terms
        push!(out, (spec, Hx, Hz))
    end
    return out
end

function efficiency(n,k,d)
    k*d*d/n
end

# threaded screen (mirrors QLDPC.screen but uses distance_rand_threaded)
function screen_threaded(candidates; min_k=1, min_d=1, trials=400, seed=0, metric=efficiency, keep=nothing, verbose=false)
    seen = Dict{String,NamedTuple}()
    # pre-shuffle candidates? keep order
    for (spec, Hx, Hz) in candidates
        verify_css(Hx, Hz) || continue
        k = compute_k(Hx, Hz)
        k < min_k && continue
        fp = fingerprint(Hx, Hz)
        haskey(seen, fp) && continue
        n = size(Hx, 2)
        fp_seed = seed + parse(Int, fp[1:8]; base=16) % 100000
        # threaded distance (deterministic perms, per-thread buffers)
        d = distance_rand_threaded(Hx, Hz; trials=trials, seed=fp_seed)
        (d == typemax(Int) || d < min_d) && continue
        w = max(row_weight(Hx), row_weight(Hz))
        eff = round(metric(n,k,d); digits=4)
        rec = (spec=spec, n=n, k=k, d=d, w=w, efficiency=eff, fingerprint=fp)
        seen[fp] = rec
        if verbose
            @info "screen" spec n k d eff
        end
    end
    out = sort(collect(values(seen)); by=r->r.efficiency, rev=true)
    if keep !== nothing
        out = out[1:min(keep, length(out))]
    end
    return out
end

function validate_via_python(doc::Dict)::Union{Dict,Nothing}
    VERIFY_DIR === nothing && return nothing
    vc = joinpath(VERIFY_DIR, "validate_candidate.py")
    isfile(vc) || return nothing
    tmp = tempname()*".json"
    open(tmp, "w") do io
        JSON.print(io, doc, 4)
    end
    # try python3 validate
    try
        out = read(`python3 $vc $tmp`, String)
        # validate_candidate.py prints JSON verdict
        # try to parse last JSON object
        # fallback: just check passed field via direct python call that returns JSON
        # Use python -c bridge for structured return
        py = """
import json, sys
sys.path.insert(0, $(repr(joinpath(CHALLENGE_ROOT, "research", "kit"))))
sys.path.insert(0, $(repr(joinpath(CHALLENGE_ROOT, "verify"))))
from validate_candidate import validate_candidate
doc=json.load(open($(repr(tmp))))
v=validate_candidate(doc)
print(json.dumps(v))
"""
        ptmp = tempname()*".py"
        write(ptmp, py)
        j = read(`python3 $ptmp`, String)
        return JSON.parse(j)
    catch e
        @warn "python validation failed, falling back to Julia-only" exception=e
        return nothing
    end
end

function main(; once=false)
    println("[w6-hunt-jl] threads=$(Threads.nthreads()) julia=$(VERSION) target kd²/n > $TARGET_EFF  l,m $(L_RANGE)..$(M_RANGE) w6")
    println("[w6-hunt-jl] challenge_root=$(CHALLENGE_ROOT === nothing ? "(none, saving to $CANDIDATES_DIR)" : CHALLENGE_ROOT)")
    println("[w6-hunt-jl] candidates_dir=$CANDIDATES_DIR  verify_dir=$(VERIFY_DIR)")
    mkpath(CANDIDATES_DIR)
    cycle = 0
    total = 0
    rng = MersenneTwister(rand(UInt32))
    while true
        cycle += 1
        seed = rand(rng, 0:2^31-1) ⊻ floor(Int, time()) ⊻ cycle
        t0 = time()
        println("\n[cycle $cycle] seed=$seed sampling $BATCH BB l,m $(L_RANGE)..$(M_RANGE) (w6, 3+3)  trials_screen=$TRIALS_SCREEN  thr=$(Threads.nthreads())  $(Dates.now())")
        flush(stdout)
        cands_raw = sample_bb_range(L_RANGE, M_RANGE, BATCH; rng=rng)
        recs = screen_threaded(cands_raw; min_k=MIN_K, min_d=MIN_D_SCREEN, trials=TRIALS_SCREEN, seed=seed)
        total += BATCH
        dt = time()-t0
        @printf("[cycle %d] screened %d k>=%d d>=%d in %.1fs  total=%d  rate=%.1f/sec\n", cycle, length(recs), MIN_K, MIN_D_SCREEN, dt, total, BATCH/dt)
        if isempty(recs)
            once && break
            sleep(0.5)
            continue
        end
        # hot list near target
        hot = filter(r -> r.k*r.d*r.d/r.n > TARGET_EFF - 2, recs)
        hot = sort(hot; by=r->r.k*r.d*r.d/r.n, rev=true)[1:min(5, length(hot))]
        for r in hot
            eff = r.k*r.d*r.d/r.n
            @printf("  HOT [[%d,%d,%d]] eff=%.2f w=%d l=%d m=%d fp=%s\n", r.n, r.k, r.d, eff, r.w, r.spec["l"], r.spec["m"], r.fingerprint)
        end
        cands = filter(r -> 260 <= r.n <= 648 && r.w <= 6 && 12 <= r.k <= 24 && r.d >= 16, recs)
        @printf("  -> %d cands to validate (n 260..648, 12<=k<=24, d>=16, w<=6)\n", length(cands))
        for r in cands[1:min(10, length(cands))]
            # re-estimate with more trials, threaded
            spec = r.spec
            # rebuild to ensure Hx/Hz match spec (sample may have dedup)
            A = spec["A"]; B = spec["B"]
            # A/B may be Vector{Tuple} or Vector{Vector}
            A_t = [(Int(a[1]), Int(a[2])) for a in A]
            B_t = [(Int(b[1]), Int(b[2])) for b in B]
            Hx, Hz = build_bb(spec["l"], spec["m"], A_t, B_t)
            println("    re-estimating d trials=$TRIALS_CONFIRM (thr=$(Threads.nthreads())) ...")
            flush(stdout)
            d2 = distance_rand_threaded(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            eff2 = r.k*d2*d2/r.n
            @printf("  check [[%d,%d,%d]] eff=%.2f (was d=%d) l=%d m=%d\n", r.n, r.k, d2, eff2, r.d, spec["l"], spec["m"])
            if d2 < 18
                continue
            end
            if eff2 < TARGET_EFF && r.k < 12
                continue
            end
            doc = make_submission(Hx, Hz;
                name="[[$(r.n),$(r.k),$d2]] bb-w6",
                construction="BB l=$(spec["l"]) m=$(spec["m"]) w6 (Julia threaded hunter)",
                authors=["@aarontrowbridge"],
                family="bivariate_bicycle",
                references=["arXiv:2308.07915"],
                confidence="upper_bound",
                trials=TRIALS_CONFIRM,
                seed=seed,
            )
            doc["provenance"] = get(doc, "provenance", Dict())
            doc["provenance"]["model"] = "QLDPC.jl threaded hunter"
            doc["provenance"]["threads"] = Threads.nthreads()
            # lightweight Julia validate
            try
                validate_candidate(doc)
            catch e
                @warn "Julia validate failed" exception=e
                continue
            end
            # Python gate if available
            v = validate_via_python(doc)
            if v !== nothing
                passed = get(v, "passed", false)
                adv = get(get(v, "novelty", Dict()), "board_advancing", false)
                labels = get(v, "labels", nothing)
                println("    validate passed=$passed adv=$adv labels=$labels")
                if !(passed && adv)
                    continue
                end
            else
                # no Python verifier: just check board_advancing via Julia efficiency
                if eff2 < TARGET_EFF
                    # not advancing, skip unless we want fieldnote
                    println("    (no Python gate) eff2=$eff2 < $TARGET_EFF, skip save")
                    continue
                end
            end
            fname = "bb-w6-$(r.n)-$(r.k)-$(d2)-l$(spec["l"])m$(spec["m"])-c$(cycle)-jl.json"
            out = joinpath(CANDIDATES_DIR, fname)
            save_submission(doc, out)
            println("    SAVED $out")
            flush(stdout)
        end
        flush(stdout)
        once && break
    end
end

# CLI: --once for smoke test
if abspath(PROGRAM_FILE) == @__FILE__
    once = "--once" in ARGS
    main(; once=once)
end
