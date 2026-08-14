#!/usr/bin/env julia --project=. --threads=24
# hunt_2bga.jl — Non-abelian 2BGA + coset hunter (Julia, threaded)
#
# Samples dihedral / metacyclic groups of order 24..120, weight 3+3 (w6) and
# 4+4 (w8), targeting unrestricted × w6 vs [[360,12,24]]  kd²/n = 19.2.
#
# Targets the same board headroom as hunt_w6 but in the non-abelian 2BGA
# design space where odd k is reachable (abelian BB cannot) and coset codes
# generalize the torus.
#
# Mirrors hunt_w6.jl structure:
#   enumerate groups → sample (a,b) → screen via distance_rand_threaded → validate
#
# Run:
#   julia --project=. --threads=24 examples/hunt_2bga.jl              # forever
#   julia --project=. --threads=24 examples/hunt_2bga.jl --once       # one cycle
#   QLDPC_CHALLENGE=~/armonia/repos/qldpc-challenge julia --threads=24 --project=. examples/hunt_2bga.jl --w8
#
# Saves candidates to research/candidates/2bga-w<w>-<n>-<k>-<d>-<group>-c<cycle>.json

using QLDPC
using Random
using Printf
using JSON
using Dates
using SparseArrays

const TARGET_EFF = 19.2  # [[360,12,24]] to beat (unrestricted × w6)
const ORDERS = [24, 30, 36, 42, 48, 54, 60, 72, 84, 96, 108, 120]
const BATCH_PER_GROUP = 200
const TRIALS_SCREEN = 600
const TRIALS_CONFIRM = 800
const MIN_K = 4
const MIN_D_SCREEN = 8

# --- challenge root discovery (for saving + Python validation) ---
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

# ---------------------------------------------------------------------------
# Group enumeration for a given order N (24..120)
# ---------------------------------------------------------------------------

"""Return list of (name, mul) groups of order N from dihedral + metacyclic families."""
function groups_for_order(N::Int)::Vector{Tuple{String,Matrix{Int}}}
    out = Tuple{String,Matrix{Int}}[]

    # Dihedral D_{N/2} when N even
    if iseven(N)
        n = N ÷ 2
        try
            mul, _ = dihedral_group(n)
            push!(out, ("D$(n)_$(N)", mul))
        catch e
            @debug "dihedral $n failed" exception=e
        end
    end

    # Metacyclic C_n ⋊ C_k with N = n*k, r^k=1 mod n
    # enumerate factorizations N = n*k with k ≥2, small k
    for k in [2, 3, 4, 6]
        N % k != 0 && continue
        n = N ÷ k
        n < 3 && continue
        # find an r with r^k=1 mod n, r ≠1 (non-abelian)
        for r in [n - 1, 2, 3, 5, 7]
            r >= n && continue
            r <= 1 && continue
            if powermod(r, k, n) == 1
                try
                    mul, _ = metacyclic(n, k, r)
                    # dedup by fingerprint of mul (cheap: hash)
                    if !any(m -> m[2] == mul, out)
                        push!(out, ("M$(n)_$(k)_r$(r)_$(N)", mul))
                    end
                catch
                end
                break  # one r per (n,k)
            end
        end
    end

    # Also include abelian Z_N as baseline (cyclic) for comparison
    try
        mul, _ = cyclic_group(N)
        push!(out, ("C$(N)", mul))
    catch
    end

    return out
end

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

function sample_2bga_for_group(mul::Matrix{Int}, n_samples::Int; weight::Int=6, rng=Random.GLOBAL_RNG)
    N = size(mul, 1)
    w_a = weight ÷ 2
    w_b = weight - w_a
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    seen = Set{String}()
    for _ in 1:n_samples
        # sample distinct indices 1..N, identity bias low (include 1 sometimes)
        all_idx = collect(1:N)
        shuffle!(rng, all_idx)
        a = sort(all_idx[1:w_a])
        b = sort(all_idx[w_a+1:w_a+w_b])
        # avoid duplicate a,b sets that give same code under group automorphism — dedup by fp later
        Hx, Hz = build_2bga(mul, a, b)
        fp = fingerprint(Hx, Hz)
        fp in seen && continue
        push!(seen, fp)
        spec = Dict("family" => "2bga", "order" => N, "a" => a, "b" => b, "weight" => weight)
        push!(out, (spec, Hx, Hz))
    end
    return out
end

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

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

function main(; once::Bool=false, weight::Int=6)
    w_label = "w$weight"
    println("[2bga-hunt-jl] threads=$(Threads.nthreads()) julia=$(VERSION) target kd²/n > $TARGET_EFF  orders $(ORDERS)  $w_label (dihedral+metacyclic)")
    println("[2bga-hunt-jl] challenge_root=$(CHALLENGE_ROOT === nothing ? "(none, saving to $CANDIDATES_DIR)" : CHALLENGE_ROOT)")
    println("[2bga-hunt-jl] candidates_dir=$CANDIDATES_DIR  weight=$weight  batch_per_group=$BATCH_PER_GROUP")
    mkpath(CANDIDATES_DIR)
    cycle = 0
    total = 0
    rng = MersenneTwister(rand(UInt32))

    # pre-enumerate groups
    all_groups = Dict{Int,Vector{Tuple{String,Matrix{Int}}}}()
    for N in ORDERS
        gs = groups_for_order(N)
        all_groups[N] = gs
        println("  order $N: $(join([g[1] for g in gs], ", "))")
    end

    while true
        cycle += 1
        seed = rand(rng, 0:2^31-1) ⊻ floor(Int, time()) ⊻ cycle
        t0 = time()
        println("\n[cycle $cycle] seed=$seed  $(w_label)  orders $(ORDERS)  trials_screen=$TRIALS_SCREEN  thr=$(Threads.nthreads())  $(Dates.now())")
        flush(stdout)

        # collect candidates across all groups
        candidates = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
        for N in ORDERS
            for (gname, mul) in all_groups[N]
                cands = sample_2bga_for_group(mul, BATCH_PER_GROUP; weight=weight, rng=rng)
                for (spec, Hx, Hz) in cands
                    spec["group"] = gname
                    spec["N"] = N
                    spec["n"] = size(Hx, 2)
                    push!(candidates, (spec, Hx, Hz))
                end
            end
        end
        total += length(candidates)
        println("  sampled $(length(candidates)) codes (total $total)")

        # screen: k, d, efficiency (threaded with serial fallback — see surrogate.jl Polyester/StrideArrays dispatch)
        seen = Dict{String,NamedTuple}()
        for (spec, Hx, Hz) in candidates
            verify_css(Hx, Hz) || continue
            k = compute_k(Hx, Hz)
            k < MIN_K && continue
            fp = fingerprint(Hx, Hz)
            haskey(seen, fp) && continue
            n = size(Hx, 2)
            fp_seed = seed + parse(Int, fp[1:8]; base=16) % 100000
            d = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_SCREEN, seed=fp_seed)
            end
            (d == typemax(Int) || d < MIN_D_SCREEN) && continue
            ww = max(row_weight(Hx), row_weight(Hz))
            ww > weight && continue  # row weight should be weight by construction
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

        hot = filter(r -> r.k * r.d * r.d / r.n > TARGET_EFF - 4, recs)
        hot = sort(hot; by=r -> r.k * r.d * r.d / r.n, rev=true)[1:min(8, length(hot))]
        for r in hot
            eff = r.k * r.d * r.d / r.n
            @printf("  HOT [[%d,%d,%d]] eff=%.2f w=%d group=%s N=%d fp=%s\n", r.n, r.k, r.d, eff, r.w, r.spec["group"], r.spec["N"], r.fingerprint)
        end

        # confirm top handful with more trials and save if advancing
        cands = filter(r -> r.n <= 720 && r.k >= 4 && r.d >= 10, recs)
        cands = sort(cands; by=r -> r.k * r.d * r.d / r.n, rev=true)[1:min(10, length(cands))]
        @printf("  -> %d cands to confirm (n≤720, k≥4, d≥10)\n", length(cands))
        for r in cands
            spec = r.spec
            Hx = r.Hx; Hz = r.Hz
            println("    re-estimating d trials=$TRIALS_CONFIRM (thr=$(Threads.nthreads())) group=$(spec["group"]) a=$(spec["a"]) b=$(spec["b"]) ...")
            flush(stdout)
            d2 = try
                distance_rand_threaded(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            catch
                distance_rand(Hx, Hz; trials=TRIALS_CONFIRM, seed=seed)
            end
            eff2 = r.k * d2 * d2 / r.n
            @printf("  check [[%d,%d,%d]] eff=%.2f (was d=%d) group=%s\n", r.n, r.k, d2, eff2, r.d, spec["group"])
            d2 < 10 && continue
            # keep upper_bound unless already very high
            doc = make_submission(Hx, Hz;
                name="[[$(r.n),$(r.k),$d2]] 2bga-$w_label",
                construction="2BGA $(spec["group"]) a=$(spec["a"]) b=$(spec["b"]) w$weight (Julia threaded hunter)",
                authors=["@aarontrowbridge"],
                family="two_block_group_algebra",
                references=["arXiv:2306.16400", "arXiv:2606.17268"],
                confidence="upper_bound",
                trials=TRIALS_CONFIRM,
                seed=seed,
            )
            doc["provenance"] = get(doc, "provenance", Dict())
            doc["provenance"]["model"] = "QLDPC.jl 2BGA threaded hunter"
            doc["provenance"]["threads"] = Threads.nthreads()
            doc["provenance"]["group"] = spec["group"]
            doc["provenance"]["order"] = spec["N"]
            doc["provenance"]["weight"] = weight
            try; validate_candidate(doc); catch e; @warn "Julia validate failed" exception=e; continue; end
            v = validate_via_python(doc)
            if v !== nothing
                passed = get(v, "passed", false); adv = get(get(v, "novelty", Dict()), "board_advancing", false)
                labels = get(v, "labels", nothing)
                println("    validate passed=$passed adv=$adv labels=$labels")
                if !(passed && adv)
                    # still save interesting near-target even if not advancing, as fieldnote?
                    if eff2 < TARGET_EFF - 2
                        continue
                    end
                end
            else
                if eff2 < TARGET_EFF - 2
                    println("    (no Python gate) eff2=$eff2 < $(TARGET_EFF-2), skip save")
                    continue
                end
            end
            fname = "2bga-$(w_label)-$(r.n)-$(r.k)-$(d2)-$(spec["group"])-c$(cycle)-jl.json"
            out = joinpath(CANDIDATES_DIR, fname)
            save_submission(doc, out)
            println("    SAVED $out")
            flush(stdout)
        end
        flush(stdout)
        once && break
    end
end

function _parse_weight(args::Vector{String})::Int
    "--w8" in args && return 8
    for i in 1:max(0, length(args) - 1)
        if args[i] == "--weight"
            return parse(Int, args[i+1])
        end
    end
    return 6
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    main(; once=("--once" in ARGS), weight=_parse_weight(ARGS))
end
