#!/usr/bin/env julia --project=. --threads=24
# hunt_planar.jl — Open-boundary planar tile hunter (Julia, threaded)
#
# Samples the Liang-Eberhardt-Chen planar tile family on Lx × Ly grids
# with r≤7 local checks. Targets the `local-2d-bilayer` board cell via
# `grid_coordinates` + `layers=2` locality submission.
#
# The flagship [[72,8,4]] (6×6) and its siblings 8×8,10×10 are validated
# with k=8. We sample flagship + random f/g variants around the validated
# regime and screen with distance_rand_threaded.
#
# Run:
#   julia --project=. --threads=24 examples/hunt_planar.jl              # forever
#   julia --project=. --threads=24 examples/hunt_planar.jl --once       # one cycle
#   QLDPC_CHALLENGE=~/armonia/repos/qldpc-challenge julia --threads=24 --project=. examples/hunt_planar.jl --once

using QLDPC
using Random
using Printf
using JSON
using Dates
using SparseArrays

const TARGET_EFF = 6.0
const TRIALS_SCREEN = 400
const TRIALS_CONFIRM = 600
const MIN_K = 8
const MIN_D_SCREEN = 4

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

# flagship and nearby tile configs
const FLAGSHIP_F = [(1,0),(2,0),(0,2)]
const FLAGSHIP_G = [(0,0),(2,1),(2,2)]

function random_tile_terms(rng::AbstractRNG; radius::Int=3)::Tuple{Vector{Tuple{Int,Int}},Vector{Tuple{Int,Int}}}
    # random weight-3 f,g within radius
    function rand_terms()
        pts = [(di,dj) for di in -radius:radius for dj in -radius:radius if !(di==0 && dj==0)]
        shuffle!(rng, pts)
        return pts[1:3]
    end
    return rand_terms(), rand_terms()
end

function main(; once::Bool=false)
    println("[planar-hunt-jl] threads=$(Threads.nthreads()) julia=$(VERSION) target eff>$TARGET_EFF  L=6..12 flagship + random tiles")
    println("[planar-hunt-jl] challenge_root=$(CHALLENGE_ROOT === nothing ? "(none, saving to $CANDIDATES_DIR)" : CHALLENGE_ROOT)")
    println("[planar-hunt-jl] candidates_dir=$CANDIDATES_DIR")
    mkpath(CANDIDATES_DIR)
    cycle = 0
    total = 0
    rng = MersenneTwister(rand(UInt32))

    while true
        cycle += 1
        seed = rand(rng, 0:2^31-1) ⊻ floor(Int, time()) ⊻ cycle
        t0 = time()
        println("\n[cycle $cycle] seed=$seed  flagship + random tiles  thr=$(Threads.nthreads())  $(Dates.now())")
        flush(stdout)

        candidates = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()

        # flagship grids 6×6 .. 12×12 + rectangular
        for Lx in [6,7,8,9,10,12]
            for Ly in [Lx, Lx+1]
                Lx*Ly*2 > 700 && continue
                Hx, Hz = build_planar(Lx, Ly, FLAGSHIP_F, FLAGSHIP_G)
                spec = Dict("family"=>"tile","Lx"=>Lx,"Ly"=>Ly,"f"=>FLAGSHIP_F,"g"=>FLAGSHIP_G,"tag"=>"flagship")
                push!(candidates, (spec, Hx, Hz))
            end
        end
        # also rectangular variants 6×8 etc
        for (Lx,Ly) in [(6,8),(8,6),(8,10),(10,8),(6,10),(10,6)]
            Hx, Hz = build_planar(Lx, Ly, FLAGSHIP_F, FLAGSHIP_G)
            spec = Dict("family"=>"tile","Lx"=>Lx,"Ly"=>Ly,"f"=>FLAGSHIP_F,"g"=>FLAGSHIP_G,"tag"=>"flagship-rect")
            push!(candidates, (spec, Hx, Hz))
        end

        # random tile variants (screen k stability)
        for _ in 1:40
            Lx = rand(rng, 5:10); Ly = rand(rng, 5:10)
            Lx*Ly*2 > 700 && continue
            f_terms, g_terms = random_tile_terms(rng; radius=3)
            Hx, Hz = build_planar(Lx, Ly, f_terms, g_terms)
            verify_css(Hx, Hz) || continue
            k = compute_k(Hx, Hz)
            # k(L) screening: check nearby size doesn't wildly fluctuate
            # but for speed just filter k≥6 here and later
            spec = Dict("family"=>"tile","Lx"=>Lx,"Ly"=>Ly,"f"=>f_terms,"g"=>g_terms,"tag"=>"random-tile")
            push!(candidates, (spec, Hx, Hz))
        end

        total += length(candidates)
        println("  sampled $(length(candidates)) tiles (total $total)")

        seen = Dict{String,NamedTuple}()
        for (spec, Hx, Hz) in candidates
            n = size(Hx,2)
            n > 700 && continue
            verify_css(Hx, Hz) || continue
            k = compute_k(Hx, Hz)
            k < MIN_K && continue
            # locality radius check r≤7
            f_terms = get(spec, "f", FLAGSHIP_F)
            g_terms = get(spec, "g", FLAGSHIP_G)
            r = planar_radius(f_terms, g_terms)
            r > 7 && continue
            fp = fingerprint(Hx, Hz)
            haskey(seen, fp) && continue
            ww = max(row_weight(Hx), row_weight(Hz))
            ww > 10 && continue
            fp_seed = seed + parse(Int, fp[1:8]; base=16) % 100000
            d = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            end
            (d == typemax(Int) || d < MIN_D_SCREEN) && continue
            eff = round(k * d * d / n; digits=4)
            rec = (spec=spec, n=n, k=k, d=d, w=ww, r=r, efficiency=eff, fingerprint=fp, Hx=Hx, Hz=Hz)
            seen[fp] = rec
        end
        recs = sort(collect(values(seen)); by=r -> r.efficiency, rev=true)
        dt = time() - t0
        @printf("[cycle %d] screened %d k>=%d d>=%d in %.1fs  rate=%.1f tiles/sec\n", cycle, length(recs), MIN_K, MIN_D_SCREEN, dt, length(candidates)/max(dt,1e-6))
        if isempty(recs)
            once && break
            sleep(0.5)
            continue
        end

        for r in recs[1:min(6, length(recs))]
            @printf("  HOT [[%d,%d,%d]] eff=%.2f w=%d r=%d Lx=%d Ly=%d tag=%s fp=%s\n",
                r.n, r.k, r.d, r.k*r.d*r.d/r.n, r.w, r.r, r.spec["Lx"], r.spec["Ly"], r.spec["tag"], r.fingerprint)
        end

        cands = filter(r -> r.n <= 700 && r.k >= 8, recs)[1:min(6, length(recs))]
        @printf("  -> %d cands to confirm\n", length(cands))
        for r in cands
            spec = r.spec; Hx = r.Hx; Hz = r.Hz
            Lx = spec["Lx"]; Ly = spec["Ly"]
            println("    re-estimating d trials=$TRIALS_CONFIRM Lx=$Lx Ly=$(Ly) ...")
            flush(stdout)
            d2 = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            end
            d2 < MIN_D_SCREEN && continue
            eff2 = r.k * d2 * d2 / r.n
            @printf("  check [[%d,%d,%d]] eff=%.2f (was d=%d) Lx=%d Ly=%d\n", r.n, r.k, d2, eff2, r.d, Lx, Ly)
            coords = grid_coordinates(Lx, Ly)
            doc = make_submission(Hx, Hz;
                name="[[$(r.n),$(r.k),$d2]] planar-tile $(Lx)x$(Ly)",
                construction="Planar tile $(Lx)x$(Ly) f=$(spec["f"]) g=$(spec["g"]) — Liang-Eberhardt-Chen open-boundary, r=$(r.r), n=$(r.n) (Julia hunter)",
                authors=["@aarontrowbridge"],
                family="tile",
                references=["arXiv:2504.08887"],
                confidence="upper_bound",
                trials=TRIALS_CONFIRM,
                seed=seed,
                coordinates=coords,
                layers=2,
            )
            doc["provenance"] = get(doc, "provenance", Dict())
            doc["provenance"]["model"] = "QLDPC.jl planar hunter"
            doc["provenance"]["threads"] = Threads.nthreads()
            doc["provenance"]["Lx"] = Lx
            doc["provenance"]["Ly"] = Ly
            doc["provenance"]["radius"] = r.r
            try; validate_candidate(doc); catch e; @warn "Julia validate failed" exception=e; continue; end
            v = validate_via_python(doc)
            if v !== nothing
                passed = get(v, "passed", false); adv = get(get(v, "novelty", Dict()), "board_advancing", false)
                labels = get(v, "labels", nothing)
                wc = get(get(v, "candidate", Dict()), "weight_class", nothing)
                lc = get(get(v, "candidate", Dict()), "locality_class", nothing)
                println("    validate passed=$passed adv=$adv labels=$labels wc=$wc lc=$lc")
                if !passed
                    continue
                end
            else
                println("    (no Python gate) eff2=$eff2")
            end
            fname = "planar-$(r.n)-$(r.k)-$(d2)-$(Lx)x$(Ly)-c$(cycle)-jl.json"
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
