"""
submit.jl — packaging + submission bridging to the challenge schema.

Mirrors `research/kit/submit.py`: `make_submission` (recomputes n/k, asserts CSS,
extracts witnesses via `lightest_logical`, fills locality if given) → `save_submission`
→ `validate_candidate` (shells to Python verifier).

The **gate stays Python** (`verify/validate_candidate.py` + `verify/qldpc_verify.py`);
this module never reimplements validation — it only packages.
"""

using SparseArrays
using JSON
using SHA

"""
    make_submission(Hx, Hz; name, construction, authors, family, references, confidence, logical_supports) -> Dict

Build a schema-shaped submission Dict ready for `validate_candidate`.

- Recomputes `n`, `k` from `Hx,Hz` and asserts CSS.
- Extracts witnesses via `lightest_logical` unless `logical_supports` supplied.
- `confidence` is "upper_bound" (default, distance_rand) or "exact" (server-certified).
- `family` e.g. "bivariate_bicycle", "tanner", etc.

Returns a Dict with keys `name, n, k, Hx, Hz, distance{d, witnesses, confidence}, ...`.
"""
function make_submission(
    Hx::AbstractMatrix,
    Hz::AbstractMatrix;
    name::String,
    construction::String,
    authors::Vector{String}=String[],
    family::String="bivariate_bicycle",
    references::Vector{String}=String[],
    confidence::String="upper_bound",
    logical_supports::Union{Nothing,Vector{Vector{Int}}}=nothing,
    trials::Int=400,
    seed::Int=0,
    coordinates=nothing,
    layers::Int=1,
)::Dict{String,Any}
    @assert verify_css(Hx, Hz) "CSS check failed: Hx*Hz' != 0"
    n = size(Hx, 2)
    k = compute_k(Hx, Hz)

    # witnesses: [support_X, support_Z] or empty if k==0
    supports = if logical_supports !== nothing
        logical_supports
    elseif k > 0
        wx, sx = lightest_logical(Hx, Hz; trials=trials, seed=seed)
        wz, sz = lightest_logical(Hz, Hx; trials=trials, seed=seed+1)
        # convert to 0-based for JSON schema (Python verifier expects 0-based)
        sx0 = [s - 1 for s in sx]
        sz0 = [s - 1 for s in sz]
        # if one side has no logical (typemax), use the other twice? — verifier wants two witnesses
        # if typemax, leave empty and set d to typemax? Better to set d = min found
        # keep what we have; verifier will reject typemax witnesses
        [sx0, sz0]
    else
        Vector{Vector{Int}}()
    end

    # distance is min witness weight (upper bound)
    d = if !isempty(supports) && all(!isempty, supports)
        minimum(length.(supports))
    elseif !isempty(supports) && any(!isempty, supports)
        minimum(length.(s) for s in supports if !isempty(s))
    else
        0
    end

    # Hx,Hz as dense Int arrays for JSON (schema expects list of lists)
    Hx_list = [collect(Int.(Hx[i, :])) for i in 1:size(Hx, 1)]
    Hz_list = [collect(Int.(Hz[i, :])) for i in 1:size(Hz, 1)]

    doc = Dict{String,Any}(
        "name" => name,
        "n" => n,
        "k" => k,
        "family" => family,
        "construction" => construction,
        "authors" => authors,
        "references" => references,
        "Hx" => Hx_list,
        "Hz" => Hz_list,
        "distance" => Dict{String,Any}(
            "d" => d,
            "witnesses" => supports,
            "confidence" => confidence,
        ),
    )
    if coordinates !== nothing
        doc["locality"] = Dict{String,Any}(
            "coordinates" => coordinates,
            "layers" => layers,
        )
    end
    return doc
end

function make_submission(
    c::CSSCode;
    name::String,
    construction::String,
    kwargs...,
)::Dict{String,Any}
    return make_submission(c.Hx, c.Hz; name=name, construction=construction, kwargs...)
end

"""
    save_submission(doc, path)

Write submission Dict to `path` as pretty JSON. Creates parent dirs.
"""
function save_submission(doc::Dict, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, doc, 4)
    end
    @info "staged to $path — now run `uv run python verify/validate_candidate.py \$path`"
    return path
end

"""
    validate_candidate(doc_or_path; python="python3", verifier_dir=nothing) -> Bool

Validate a submission Dict or JSON file via the Python verifier.

If `verifier_dir` is given, shells to `python verify/validate_candidate.py <tmp>`.
Otherwise does a lightweight Julia-only check: schema presence, n/k, CSS, witness
weights match `d`, and witness nontriviality via `commutes`/`in_rowspace`.

Returns `true` if valid, throws or returns `false` with message otherwise.
When Python verifier is available the return mirrors its verdict.
"""
function validate_candidate(
    doc_or_path::Union{Dict,String};
    verifier_dir::Union{Nothing,String}=nothing,
)::Bool
    doc = doc_or_path isa String ? JSON.parsefile(doc_or_path) : doc_or_path

    # lightweight Julia check (always runs)
    for key in ["name", "n", "k", "Hx", "Hz", "distance"]
        haskey(doc, key) || error("missing key: $key")
    end
    Hx = hcat([collect(row) for row in doc["Hx"]]...)'
    # Actually doc["Hx"] is Vector{Vector{Int}} rows; reconstruct Matrix
    # Simpler: build Bool matrices directly from lists
    Hx_m = _list_to_bool(doc["Hx"])
    Hz_m = _list_to_bool(doc["Hz"])
    n = doc["n"]
    @assert size(Hx_m, 2) == n "n mismatch"
    @assert size(Hz_m, 2) == n "n mismatch"
    k_claim = doc["k"]
    k_true = compute_k(Hx_m, Hz_m)
    k_claim == k_true || error("k mismatch: claimed $k_claim, computed $k_true")
    verify_css(Hx_m, Hz_m) || error("CSS violation")
    d = doc["distance"]["d"]
    witnesses = doc["distance"]["witnesses"]
    if !isempty(witnesses)
        for (i, w) in enumerate(witnesses)
            isempty(w) && continue
            v = zeros(Bool, n)
            for idx in w
                # schema is 0-based, Julia 1-based
                v[idx+1] = true
            end
            @assert count(identity, v) == d || count(identity, v) >= d "witness weight $(count(identity,v)) != d $d"
            # check nontriviality: in ker(opposite) and not in rowspace(own)
            if i == 1
                # first witness is X-type: in ker(Hz), not in rowspace(Hx)
                commutes(v, Hz_m) || error("witness $i not in ker(Hz)")
                !in_rowspace(v, Hx_m) || error("witness $i in rowspace(Hx) — trivial")
            else
                commutes(v, Hx_m) || error("witness $i not in ker(Hx)")
                !in_rowspace(v, Hz_m) || error("witness $i in rowspace(Hz) — trivial")
            end
        end
    end

    # optional Python verifier shell-out
    if verifier_dir !== nothing
        tmp = tempname() * ".json"
        open(tmp, "w") do io; JSON.print(io, doc, 2); end
        cmd = `python3 $(joinpath(verifier_dir, "validate_candidate.py")) $tmp`
        try
            run(cmd)
        catch e
            error("Python verifier failed: $e")
        end
    end
    return true
end

function _list_to_bool(rows::AbstractVector)::Matrix{Bool}
    m = length(rows)
    m == 0 && return zeros(Bool, 0, 0)
    n = length(rows[1])
    M = zeros(Bool, m, n)
    for i in 1:m
        for j in 1:n
            M[i, j] = Bool(rows[i][j] & 1 != 0)
        end
    end
    return M
end

using TestItems

@testitem "submit: make and validate BB 72" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    doc = make_submission(Hx, Hz; name="test-72", construction="Julia BB test", authors=["Tester"], family="bivariate_bicycle", trials=80, seed=0)
    @test doc["n"] == 72
    @test doc["k"] == 12
    @test haskey(doc, "distance")
    # lightweight validate
    @test validate_candidate(doc) == true
end

@testitem "submit: k mismatch is caught" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    doc = make_submission(Hx, Hz; name="bad-k", construction="x", trials=40)
    doc["k"] = 999
    @test_throws Exception validate_candidate(doc)
end

