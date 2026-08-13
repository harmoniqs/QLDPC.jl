# # Quickstart — QLDPC.jl

# Build the [[72,12,6]] BB code (Bravyi et al. gross code) and screen it.

using QLDPC

# 1. Build BB periodic code on Z_6 × Z_6 (weight-6, 3+3)
Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
@show size(Hx), size(Hz)

# 2. CSS + k
@assert verify_css(Hx, Hz)
k = compute_k(Hx, Hz)  # 12
@show k

# 3. Distance upper bound (RIS, 400 trials)
d = distance_rand(Hx, Hz; trials = 400, seed = 0)
@show d  # ≤ 6 expected

# 4. Search funnel
recs = screen([(("demo", Hx, Hz))]; min_k = 1, trials = 200)
@show recs[1]

# 5. Submission (witnesses extracted automatically)
doc = make_submission(
    Hx,
    Hz;
    name = "demo-72",
    construction = "Julia BB demo",
    authors = ["QLDPC.jl"],
    family = "bivariate_bicycle",
    trials = 100,
)
@show doc["n"], doc["k"], doc["distance"]["d"]
@assert validate_candidate(doc)

# Next: `save_submission(doc, "candidate.json")` then `uv run python verify/validate_candidate.py candidate.json`
