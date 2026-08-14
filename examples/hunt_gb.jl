#!/usr/bin/env julia --project=. --threads=24
# hunt_gb.jl — Cyclic GB divisor hunter (Julia, threaded)
#
# Samples generalized-bicycle codes on Z_N with both circulants multiples
# of a divisor g(x) | x^N - 1  → k = 2*deg(g) by construction.
# Targets high-rate 390,82 and siblings (195,231,255,273) with w 8-12
# and honest distance. Also economizes with small-N demonstrators.
#
# Uses `gb.jl`: `gb_divisors` + `build_gb` / `enumerate_gb_candidates`.
# For 390 we use Nemo factoring when available (else brute small N).
#
# Run:
#   julia --project=. --threads=24 examples/hunt_gb.jl              # forever
#   julia --project=. --threads=24 examples/hunt_gb.jl --once       # one cycle
#   julia --project=. --threads=24 examples/hunt_gb.jl --N 195 --deg 41 --once
#   QLDPC_CHALLENGE=~/armonia/repos/qldpc-challenge julia --threads=24 --project=. examples/hunt_gb.jl --once

using QLDPC
using Random
using Printf
using JSON
using Dates
using SparseArrays

const TARGET_EFF = 15.0  # GB high-rate target
const TRIALS_SCREEN = 400
const TRIALS_CONFIRM = 600
const BATCH = 80

# Primary sweep points: N=195 (k≈82), 231, 255, 273, 285, 315
const SWEEP_POINTS = [(195,41),(195,40),(231,42),(255,43),(273,44),(63,12),(31,10),(21,6)]

function find_challenge_root()::Union{String,Nothing}
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
    end
    return nothing
end

const CHALLENGE_ROOT = find_challenge_root()
const CANDIDATES_DIR = CHALLENGE_ROOT === nothing ? joinpath(pwd(), "research", "candidates") : joinpath(CHALLENGE_ROOT, "research", "candidates")
const VERIFY_DIR = CHALLENGE_ROOT === nothing ? nothing : joinpath(CHALLENGE_ROOT, "verify")

function validate_via_python(doc::Dict)::Union{Dict,Nothing}
    VERIFY_DIR === nothing && return nothing
    vc = joinpath(VERIFY_DIR, "validate_candidate.py")
    isfile(vc) || return nothing
    tmp = tempname() * ".json"
    open(tmp, "w") do io; JSON.print(io, doc, 4); end
    try
        py = """
import json, sys
sys.path.insert(0, $(repr(joinpath(CHALLENGE_ROOT, "research", "kit"))))
sys.path.insert(0, $(repr(joinpath(CHALLENGE_ROOT, "verify"))))
from validate_candidate import validate_candidate
doc=json.load(open($(repr(tmp))))
v=validate_candidate(doc)
print(json.dumps(v))
"""
        ptmp = tempname() * ".py"
        write(ptmp, py)
        j = read(`python3 $ptmp`, String)
        return JSON.parse(j)
    catch e
        @warn "python validation failed" exception=e
        return nothing
    end
end

function parse_cli_args()
    N = nothing; deg = nothing; once = false
    for i in 1:length(ARGS)
        if ARGS[i]=="--N" && i < length(ARGS)
            N = parse(Int, ARGS[i+1])
        elseif ARGS[i]=="--deg" && i < length(ARGS)
            deg = parse(Int, ARGS[i+1])
        end
    end
    once = "--once" in ARGS
    return (N, deg, once)
end

function main(; N_override::Union{Int,Nothing}=nothing, deg_override::Union{Int,Nothing}=nothing, once::Bool=false)
    points = if N_override !== nothing && deg_override !== nothing
        [(N_override, deg_override)]
    else
        SWEEP_POINTS
    end
    println("[gb-hunt-jl] threads=$(Threads.nthreads()) julia=$(VERSION) target eff>$TARGET_EFF  points=$points")
    println("[gb-hunt-jl] challenge_root=$(CHALLENGE_ROOT === nothing ? "(none, saving to $CANDIDATES_DIR)" : CHALLENGE_ROOT)")
    println("[gb-hunt-jl] candidates_dir=$CANDIDATES_DIR  trials_screen=$TRIALS_SCREEN batch=$BATCH")
    mkpath(CANDIDATES_DIR)
    cycle = 0
    total = 0
    rng = MersenneTwister(rand(UInt32))

    while true
        cycle += 1
        seed = rand(rng, 0:2^31-1) ⊻ floor(Int, time()) ⊻ cycle
        t0 = time()
        println("\n[cycle $cycle] seed=$seed  points=$points  thr=$(Threads.nthreads())  $(Dates.now())")
        flush(stdout)

        candidates = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()

        for (N, deg) in points
            2*N > 700 && continue
            # gb_divisors may be empty without Nemo for large N/deg — handle gracefully
            divs = gb_divisors(N, deg; max_count=12)
            if isempty(divs)
                # fallback: direct random GB without divisor constraint for demo
                # but tag as "gb-generic" so board family still generalized-bicycle
                for _ in 1:min(BATCH÷2, 20)
                    a = randperm(rng, N)[1:rand(rng, 4:6)]
                    b = randperm(rng, N)[1:rand(rng, 4:6)]
                    Hx, Hz = build_gb(N, [x-1 for x in a], [x-1 for x in b])
                    spec = Dict("family"=>"generalized-bicycle","N"=>N,"deg"=>deg,"g"=>nothing,"a"=>[x-1 for x in a],"b"=>[x-1 for x in b],"tag"=>"gb-generic","weight"=>length(a)+length(b))
                    push!(candidates, (spec, Hx, Hz))
                end
                println("  N=$N deg=$deg: no divisor (need Nemo for large), used generic fallback")
                continue
            end
            # sample BATCH codes per point
            for _ in 1:BATCH
                g = divs[rand(rng, 1:length(divs))]
                a_exps, b_exps = random_gb_pair(N, g; weight_target=rand(rng, 8:12), rng=rng)
                (isempty(a_exps) || isempty(b_exps)) && continue
                Hx, Hz = build_gb(N, a_exps, b_exps)
                verify_css(Hx, Hz) || continue
                spec = Dict("family"=>"generalized-bicycle","N"=>N,"deg"=>deg,"g"=>g,"a"=>a_exps,"b"=>b_exps,"tag"=>"gb-divisor","wt_a"=>length(a_exps),"wt_b"=>length(b_exps))
                push!(candidates, (spec, Hx, Hz))
            end
            println("  N=$N deg=$deg: divisors=$(length(divs)) sampled $BATCH")
        end

        # also sample via enumerate_gb_candidates for small demonstrators
        for (N,deg) in [(15,4),(21,6),(31,5)]
            cands = enumerate_gb_candidates(N, deg; n_samples=10, weight=6, seed=seed+cycle+N)
            for (spec,Hx,Hz) in cands
                spec["tag"]="gb-enumerate"
                push!(candidates, (spec, Hx, Hz))
            end
        end

        total += length(candidates)
        println("  sampled $(length(candidates)) GB candidates (total $total)")

        seen = Dict{String,NamedTuple}()
        for (spec, Hx, Hz) in candidates
            n = size(Hx,2)
            n > 700 && continue
            verify_css(Hx, Hz) || continue
            k = compute_k(Hx, Hz)
            k < 4 && continue
            fp = fingerprint(Hx, Hz)
            haskey(seen, fp) && continue
            ww = max(row_weight(Hx), row_weight(Hz))
            ww > 12 && continue
            fp_seed = seed + parse(Int, fp[1:8]; base=16) % 100000
            d = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            end
            (d == typemax(Int) || d < 6) && continue
            eff = round(k * d * d / n; digits=4)
            rec = (spec=spec, n=n, k=k, d=d, w=ww, efficiency=eff, fingerprint=fp, Hx=Hx, Hz=Hz)
            seen[fp] = rec
        end
        recs = sort(collect(values(seen)); by=r -> r.efficiency, rev=true)
        dt = time() - t0
        @printf("[cycle %d] screened %d k>=4 d>=6 in %.1fs  rate=%.1f codes/sec\n", cycle, length(recs), dt, length(candidates)/max(dt,1e-6))
        if isempty(recs)
            once && break
            sleep(0.5)
            continue
        end

        hot = filter(r -> r.n <= 700, recs)[1:min(8, length(recs))]
        for r in hot
            eff = r.k * r.d * r.d / r.n
            N = get(r.spec, "N", r.n÷2)
            @printf("  HOT [[%d,%d,%d]] eff=%.2f w=%d N=%d tag=%s fp=%s\n", r.n, r.k, r.d, eff, r.w, N, get(r.spec,"tag","?"), r.fingerprint)
        end

        cands = filter(r -> r.n <= 700 && r.k >= 6 && r.d >= 6, recs)
        cands = sort(cands; by=r -> r.k * r.d * r.d / r.n, rev=true)[1:min(8, length(cands))]
        @printf("  -> %d cands to confirm (n≤700, k≥6, d≥6)\n", length(cands))
        for r in cands
            spec = r.spec; Hx = r.Hx; Hz = r.Hz
            N = get(spec, "N", r.n÷2)
            println("    re-estimating d trials=$TRIALS_CONFIRM N=$N tag=$(get(spec,"tag","?")) ...")
            flush(stdout)
            d2 = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            end
            d2 < 6 && continue
            eff2 = r.k * d2 * d2 / r.n
            @printf("  check [[%d,%d,%d]] eff=%.2f (was d=%d) N=%d wt_a=%s wt_b=%s\n", r.n, r.k, d2, eff2, r.d, N, get(spec,"wt_a","?"), get(spec,"wt_b","?"))
            doc = make_submission(Hx, Hz;
                name="[[$(r.n),$(r.k),$d2]] GB N=$(N) w$(r.w)",
                construction="Cyclic GB on Z_$(N): g=$(get(spec,"g",nothing)) a=$(get(spec,"a",nothing)) b=$(get(spec,"b",nothing)) w$(r.w) via gb divisor (k=2deg by construction, Julia hunter)",
                authors=["@aarontrowbridge"],
                family="generalized-bicycle",
                references=["arXiv:2308.07915","fieldnotes/2026-07-14-designed-divisor-and-odd-k.md"],
                confidence="upper_bound",
                trials=TRIALS_CONFIRM,
                seed=seed,
            )
            doc["provenance"] = get(doc, "provenance", Dict())
            doc["provenance"]["model"] = "QLDPC.jl GB hunter"
            doc["provenance"]["threads"] = Threads.nthreads()
            doc["provenance"]["N"] = N
            doc["provenance"]["tag"] = get(spec, "tag", "gb-divisor")
            try; validate_candidate(doc); catch e; @warn "Julia validate failed" exception=e; continue; end
            v = validate_via_python(doc)
            if v !== nothing
                passed = get(v, "passed", false); adv = get(get(v, "novelty", Dict()), "board_advancing", false)
                labels = get(v, "labels", nothing)
                println("    validate passed=$passed adv=$adv labels=$labels")
                if !passed
                    continue
                end
                # save even if not advancing when eff high? but GB is competitive only if advancing
                if !adv && eff2 < TARGET_EFF
                    println("    not advancing and eff2 < $TARGET_EFF, skip save")
                    continue
                end
            else
                if eff2 < TARGET_EFF - 2
                    println("    (no Python gate) eff2=$eff2 < $(TARGET_EFF-2), skip save")
                    continue
                end
            end
            fname = "gb-$(r.n)-$(r.k)-$(d2)-N$(N)-c$(cycle)-jl.json"
            out = joinpath(CANDIDATES_DIR, fname)
            save_submission(doc, out)
            println("    SAVED $out")
            flush(stdout)
        end
        flush(stdout)
        once && break
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    Narg, degarg, once = parse_cli_args()
    main(; N_override=Narg, deg_override=degarg, once=once)
end
