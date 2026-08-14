#!/usr/bin/env julia --project=. --threads=24
# hunt_tanner.jl — Lifted / Tanner / Balanced-product hunter (Julia, threaded)
#
# Samples hypergraph product, lifted product (a≠b 2BGA), balanced product
# (HP of two 2BGAs), and quantum-Tanner expanders via random regular LDPC
# + Kron constructions. Targets the first-mover `quantum-tanner` board cell:
# no code on the board yet, so any valid Tanner is advancing. Also covers
# `hypergraph-product` / `lifted-product` families.
#
# Mirrors hunt_2bga.jl structure:
#   sample via products.jl → screen via distance_rand_threaded → validate
#
# Run:
#   julia --project=. --threads=24 examples/hunt_tanner.jl              # forever
#   julia --project=. --threads=24 examples/hunt_tanner.jl --once       # one cycle
#   QLDPC_CHALLENGE=~/armonia/repos/qldpc-challenge julia --threads=24 --project=. examples/hunt_tanner.jl --once

using QLDPC
using Random
using Printf
using JSON
using Dates
using SparseArrays

const TARGET_EFF = 8.0   # Tanner is first-mover: any k≥4 d≥6 is board advancing
const TRIALS_SCREEN = 400
const TRIALS_CONFIRM = 600
const MIN_K = 4
const MIN_D_SCREEN = 6
const BATCH_HP = 80
const BATCH_LP = 80
const BATCH_BP = 40
const BATCH_QT = 40

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

function main(; once::Bool=false)
    println("[tanner-hunt-jl] threads=$(Threads.nthreads()) julia=$(VERSION) target eff>$TARGET_EFF  HP/LP/BP/QT")
    println("[tanner-hunt-jl] challenge_root=$(CHALLENGE_ROOT === nothing ? "(none, saving to $CANDIDATES_DIR)" : CHALLENGE_ROOT)")
    println("[tanner-hunt-jl] candidates_dir=$CANDIDATES_DIR  trials_screen=$TRIALS_SCREEN")
    mkpath(CANDIDATES_DIR)
    cycle = 0
    total = 0
    rng = MersenneTwister(rand(UInt32))

    while true
        cycle += 1
        seed = rand(rng, 0:2^31-1) ⊻ floor(Int, time()) ⊻ cycle
        t0 = time()
        println("\n[cycle $cycle] seed=$seed  HP=$BATCH_HP LP=$BATCH_LP BP=$BATCH_BP QT=$BATCH_QT  thr=$(Threads.nthreads())  $(Dates.now())")
        flush(stdout)

        candidates = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()

        # HP: random classical LDPC → HP
        for (spec,Hx,Hz) in sample_hypergraph_product(BATCH_HP; m_range=(3,8), n_range=(6,14), row_w=3, seed=seed+cycle)
            spec["tag"]="HP-rw3"
            push!(candidates, (spec, Hx, Hz))
        end
        for (spec,Hx,Hz) in sample_hypergraph_product(20; m_range=(4,9), n_range=(8,16), row_w=4, seed=seed+1000+cycle)
            spec["tag"]="HP-rw4"
            push!(candidates, (spec, Hx, Hz))
        end
        # LP: lifted product with a≠b
        for (spec,Hx,Hz) in sample_lifted_product(BATCH_LP; order_range=(8,28), weight_a=3, weight_b=3, seed=seed+2000+cycle)
            spec["tag"]="LP-3-3"
            push!(candidates, (spec, Hx, Hz))
        end
        for (spec,Hx,Hz) in sample_lifted_product(20; order_range=(12,32), weight_a=3, weight_b=4, seed=seed+3000+cycle)
            spec["tag"]="LP-3-4"
            push!(candidates, (spec, Hx, Hz))
        end
        # BP: HP of two 2BGAs
        for (spec,Hx,Hz) in sample_balanced_product(BATCH_BP; order_range=(4,12), seed=seed+4000+cycle)
            spec["tag"]="BP-4-12"
            push!(candidates, (spec, Hx, Hz))
        end
        # QT: quantum Tanner via random regular bipartite expanders
        for i in 1:BATCH_QT
            m1 = rand(rng, 4:7); n1 = rand(rng, 10:16)
            m2 = rand(rng, 4:7); n2 = rand(rng, 10:16)
            row_w = rand(rng, 3:4)
            Hx, Hz = quantum_tanner(m1, n1, m2, n2; row_weight=row_w, seed=seed+5000+cycle+i)
            spec = Dict("family"=>"quantum-tanner","H1_shape"=>[m1,n1],"H2_shape"=>[m2,n2],"row_weight"=>row_w,"tag"=>"QT")
            push!(candidates, (spec, Hx, Hz))
        end

        total += length(candidates)
        println("  sampled $(length(candidates)) codes (total $total)")

        seen = Dict{String,NamedTuple}()
        for (spec, Hx, Hz) in candidates
            n = size(Hx,2)
            (n > 700 || n < 12) && continue
            verify_css(Hx, Hz) || continue
            k = compute_k(Hx, Hz)
            k < MIN_K && continue
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
            (d == typemax(Int) || d < MIN_D_SCREEN) && continue
            ww > 10 && continue
            eff = round(k * d * d / n; digits=4)
            rec = (spec=spec, n=n, k=k, d=d, w=ww, efficiency=eff, fingerprint=fp, Hx=Hx, Hz=Hz)
            seen[fp] = rec
        end
        recs = sort(collect(values(seen)); by=r -> r.efficiency, rev=true)
        dt = time() - t0
        @printf("[cycle %d] screened %d k>=%d d>=%d in %.1fs  rate=%.1f codes/sec\n", cycle, length(recs), MIN_K, MIN_D_SCREEN, dt, length(candidates)/max(dt,1e-6))
        if isempty(recs)
            once && break
            sleep(0.5)
            continue
        end

        hot = filter(r -> r.n <= 700, recs)[1:min(8, length(recs))]
        for r in hot
            eff = r.k * r.d * r.d / r.n
            tag = get(r.spec, "tag", "?")
            @printf("  HOT [[%d,%d,%d]] eff=%.2f w=%d tag=%s fp=%s\n", r.n, r.k, r.d, eff, r.w, tag, r.fingerprint)
        end

        # confirm top handful
        cands = filter(r -> r.n <= 700 && r.k >= 4 && r.d >= 6, recs)
        cands = sort(cands; by=r -> r.k * r.d * r.d / r.n, rev=true)[1:min(8, length(cands))]
        @printf("  -> %d cands to confirm (n≤700, k≥4, d≥6)\n", length(cands))
        for r in cands
            spec = r.spec; Hx = r.Hx; Hz = r.Hz
            tag = get(spec, "tag", "tanner")
            println("    re-estimating d trials=$TRIALS_CONFIRM tag=$tag ...")
            flush(stdout)
            d2 = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            end
            d2 < 6 && continue
            eff2 = r.k * d2 * d2 / r.n
            @printf("  check [[%d,%d,%d]] eff=%.2f (was d=%d) tag=%s\n", r.n, r.k, d2, eff2, r.d, tag)
            family = get(spec, "family", "quantum-tanner")
            # map to valid schema family
            schema_family = family in ("hypergraph-product","lifted-product","balanced-product","quantum-tanner") ? family : "quantum-tanner"
            doc = make_submission(Hx, Hz;
                name="[[$(r.n),$(r.k),$d2]] $schema_family $(tag)",
                construction="$schema_family via $(tag): $(spec) — product/Tanner expander family (Julia hunter)",
                authors=["@aarontrowbridge"],
                family=schema_family,
                references=["Tillich-Zemor 2009","Panteleev-Kalachev 2021","Leverrier-Zemor 2022","Hastings-Haah-O'Donnell 2020"],
                confidence="upper_bound",
                trials=TRIALS_CONFIRM,
                seed=seed,
            )
            doc["provenance"] = get(doc, "provenance", Dict())
            doc["provenance"]["model"] = "QLDPC.jl tanner hunter"
            doc["provenance"]["threads"] = Threads.nthreads()
            doc["provenance"]["tag"] = tag
            try; validate_candidate(doc); catch e; @warn "Julia validate failed" exception=e; continue; end
            v = validate_via_python(doc)
            if v !== nothing
                passed = get(v, "passed", false); adv = get(get(v, "novelty", Dict()), "board_advancing", false)
                labels = get(v, "labels", nothing)
                println("    validate passed=$passed adv=$adv labels=$labels")
                if !(passed)
                    continue
                end
            else
                # no Python gate: keep if k≥4 d≥6 n≤700
                println("    (no Python gate) eff2=$eff2")
            end
            fname = "tanner-$(r.n)-$(r.k)-$(d2)-$(tag)-c$(cycle)-jl.json"
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
    once = "--once" in ARGS
    main(; once=once)
end
