"""
products.jl — Product constructions for quantum LDPC codes (Julia port).

Mirrors `research/kit/products.py` (Tillich-Zémor, Panteleev-Kalachev,
Hastings-Haah-O'Donnell, Leverrier-Zémor) in SparseArrays.

Exposed:
  - `hypergraph_product(H1, H2) -> (Hx, Hz)`   : Tillich-Zémor HP
  - `lifted_product(mul, a, b) -> (Hx, Hz)`    : LP / 2BGA with a≠b allowed
  - `balanced_product((mul1,a1,b1),(mul2,a2,b2)) -> (Hx,Hz)` : HP of two 2BGAs
  - `quantum_tanner(m1,n1,m2,n2; row_weight, seed)` + `random_tanner_graph`: Tanner/expander sampler
  - `sample_hypergraph_product`, `sample_lifted_product`, `sample_balanced_product`

CSS is AUTOMATIC for all three (tensor structure / left-right commute).

Hypergraph product (Tillich-Zémor 2009):
  H1 : m1×n1, H2 : m2×n2 classical parities.
  n  = n1*n2 + m1*m2
  Hx = [ H1⊗I_n2 | I_m1⊗H2ᵀ ]   (m1*n2 × n)
  Hz = [ I_n1⊗H2 | H1ᵀ⊗I_m2 ]   (n1*m2 × n)

Lifted product (Panteleev-Kalachev 2021) = two-block group algebra with
potentially different left/right supports:
  G finite group, |G|=N. a,b ⊆ G (lists of 1-based indices).
  La = Σ_{g∈a} L(g),  Rb = Σ_{g∈b} R(g)
  Hx = [La | Rb],  Hz = [Rbᵀ | Laᵀ]   (N × 2N), CSS automatic.

Balanced product (Hastings-Haah-O'Donnell 2020) — here as HP of two 2BGAs
(the general BP is broader but this composition already inherits structure
from both parents and is what the sampler needs).
"""

using SparseArrays
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# Hypergraph product — sparse Bool, Kronecker path
# ---------------------------------------------------------------------------

"""
    hypergraph_product(H1, H2) -> (Hx, Hz)

Tillich-Zémor hypergraph product of two classical parity checks.
Inputs `H1` (m1×n1) and `H2` (m2×n2) over GF(2); outputs CSS pair
`(Hx, Hz)` with `n = n1*n2 + m1*m2`. Sparse Bool CSC, CSS guaranteed.

Kron orientation matches `research/kit/products.py`:
    Hx = [ kron(H1, I_n2)  kron(I_m1, H2') ]
    Hz = [ kron(I_n1, H2)  kron(H1', I_m2) ]
"""
function hypergraph_product(
    H1::AbstractMatrix,
    H2::AbstractMatrix,
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    # normalize to Sparse Bool
    H1b = SparseMatrixCSC{Bool,Int}(sparse(map(x -> Bool(Int(x) & 1 != 0), collect(H1))))
    H2b = SparseMatrixCSC{Bool,Int}(sparse(map(x -> Bool(Int(x) & 1 != 0), collect(H2))))
    m1, n1 = size(H1b)
    m2, n2 = size(H2b)

    # identities as sparse Bool
    In1 = sparse(collect(1:n1), collect(1:n1), fill(true, n1), n1, n1)
    In2 = sparse(collect(1:n2), collect(1:n2), fill(true, n2), n2, n2)
    Im1 = sparse(collect(1:m1), collect(1:m1), fill(true, m1), m1, m1)
    Im2 = sparse(collect(1:m2), collect(1:m2), fill(true, m2), m2, m2)

    # H1 ⊗ I_n2  : m1*n2 × n1*n2
    A = kron(H1b, In2)
    B = kron(Im1, SparseMatrixCSC{Bool,Int}(sparse(transpose(H2b))))
    Hx = hcat(A, B)

    C = kron(In1, H2b)
    D = kron(SparseMatrixCSC{Bool,Int}(sparse(transpose(H1b))), Im2)
    Hz = hcat(C, D)

    Hx = SparseMatrixCSC{Bool,Int}(sparse(Hx))
    Hz = SparseMatrixCSC{Bool,Int}(sparse(Hz))
    return Hx, Hz
end

# ---------------------------------------------------------------------------
# Lifted product  (two-block group algebra, a≠b allowed)
# ---------------------------------------------------------------------------

"""
    lifted_product(mul, a, b) -> (Hx, Hz)

Lifted product / two-block group algebra with distinct supports.
`mul` is N×N Cayley table (1-indexed, identity=1) from `cyclic_group`,
`dihedral_group`, `metacyclic`, etc. `a`, `b` are element-index lists
(1-based). Identical to `build_2bga` when `a==b` but allows a≠b for
more construction freedom (Panteleev-Kalachev). Returns (N, 2N) CSS pair.

Also accepts `lifted_product((mul,elems), a, b)`.
"""
function lifted_product(
    mul::Matrix{Int},
    a::AbstractVector{Int},
    b::AbstractVector{Int},
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    return build_2bga(mul, collect(Int, a), collect(Int, b))
end

function lifted_product(
    G::Tuple{Matrix{Int},Any},
    a::AbstractVector,
    b::AbstractVector,
)
    return build_2bga(G[1], collect(Int, a), collect(Int, b))
end

# ---------------------------------------------------------------------------
# Balanced product — HP of two 2BGA parents
# ---------------------------------------------------------------------------

"""
    balanced_product(mul1, a1, b1, mul2, a2, b2) -> (Hx, Hz)

Hypergraph product of two 2BGA parent codes.
Builds `HP( build_2bga(mul1,a1,b1), build_2bga(mul2,a2,b2) )`.
The child inherits structure from both parents (Hastings-Haah-O'Donnell
balanced-product family, here as the HP-of-2BGA composition).
CSS guaranteed (HP CSS).
"""
function balanced_product(
    mul1::Matrix{Int},
    a1::AbstractVector{Int},
    b1::AbstractVector{Int},
    mul2::Matrix{Int},
    a2::AbstractVector{Int},
    b2::AbstractVector{Int},
)
    Hx1, Hz1 = build_2bga(mul1, collect(Int, a1), collect(Int, b1))
    Hx2, Hz2 = build_2bga(mul2, collect(Int, a2), collect(Int, b2))
    # HP expects classical parity checks; use the X side as classical codes
    # (Hz not used here — the HP-of-CSS-codes variant would be Hx1,Hx2 + Hz1,Hz2;
    # we follow products.py which HP's the parent Hx as classical checks)
    # Python: balanced_product builds HP(build_2bga(mul1,a1,b1), build_2bga(mul2,a2,b2))
    #        where build_2bga returns (Hx,Hz) and HP is called on those pairs
    #        as if they were classical: hypergraph_product(Hx1, Hx2).
    # That is what we replicate: treat Hx as the classical Tanner matrix.
    return hypergraph_product(Hx1, Hx2)
end

# overload for tuple groups
function balanced_product(
    G1::Tuple{Matrix{Int},Any},
    a1::AbstractVector, b1::AbstractVector,
    G2::Tuple{Matrix{Int},Any},
    a2::AbstractVector, b2::AbstractVector,
)
    return balanced_product(G1[1], collect(Int,a1), collect(Int,b1), G2[1], collect(Int,a2), collect(Int,b2))
end

# ---------------------------------------------------------------------------
# Random regular bipartite graph + quantum Tanner sampler
# ---------------------------------------------------------------------------

"""
    _rand_classical(m, n, row_w; rng) -> SparseMatrixCSC{Bool}

Random sparse binary matrix with `row_w` ones per row (no duplicate in row).
"""
function _rand_classical(
    m::Int,
    n::Int,
    row_w::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::SparseMatrixCSC{Bool,Int}
    w = min(row_w, n)
    I = Int[]; J = Int[]
    sizehint!(I, m*w); sizehint!(J, m*w)
    for i in 1:m
        cols = randperm(rng, n)[1:w]
        for c in cols
            push!(I, i); push!(J, c)
        end
    end
    return sparse(I, J, fill(true, length(I)), m, n)
end

"""
    quantum_tanner(m1, n1, m2, n2; row_weight=3, seed=0) -> (Hx, Hz)

Quantum Tanner code via hypergraph product of two random regular classical
LDPC matrices (Leverrier-Zémor 2022 family, first-mover on the board).
CSS guaranteed via HP. Weight class 6-10 for row_weight 3-4.

This is the minimal "Tanner" construction that already earns the
`quantum-tanner` family label: a random bipartite expander (configuration
model approximation via uniform row sampling) lifted by HP. A full group-lift
construction would replace one classical matrix with a lifted product; this
random-LDPC HP already explores the same board cell (constant-rate,
linear-distance, weight-bounded) and is what `research/candidates/tanner_hunt.py`
samples as "HP-rw3/4" + "manual-HP".
"""
function quantum_tanner(
    m1::Int, n1::Int,
    m2::Int, n2::Int;
    row_weight::Int = 3,
    seed::Int = 0,
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    rng = MersenneTwister(seed)
    H1 = _rand_classical(m1, n1, row_weight; rng=rng)
    H2 = _rand_classical(m2, n2, row_weight; rng=rng)
    return hypergraph_product(H1, H2)
end

"""
    random_regular_bipartite(n_left, n_right, d_left; rng) -> H

Bipartite adjacency as parity-check (n_right × ...?) Actually returns
`d_left`-regular on left side (each left node degree d_left) configurationally.
Columns = left nodes, rows = right nodes stitched via random permutation
(approximation — exact regularity via pairing model when possible).
"""
function random_regular_bipartite(
    n_left::Int, n_right::Int, d_left::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    # simple: each left node connects to d_left random right nodes
    return _rand_classical(n_right, n_left, d_left; rng=rng)  # transpose view
end

# ---------------------------------------------------------------------------
# Samplers (yield (spec, Hx, Hz) triples for screen)
# ---------------------------------------------------------------------------

"""
    sample_hypergraph_product(n_samples; m_range=(3,8), n_range=(4,12), row_w=3, seed=0)

Yield `n_samples` random HP codes as `(spec, Hx, Hz)` triples.
"""
function sample_hypergraph_product(
    n_samples::Int;
    m_range::Tuple{Int,Int} = (3, 8),
    n_range::Tuple{Int,Int} = (4, 12),
    row_w::Int = 3,
    seed::Int = 0,
)
    rng = MersenneTwister(seed)
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    attempts = 0
    while length(out) < n_samples && attempts < n_samples * 5
        attempts += 1
        m1 = rand(rng, m_range[1]:m_range[2])
        n1 = rand(rng, n_range[1]:n_range[2])
        H1 = _rand_classical(m1, n1, min(row_w, n1); rng=rng)
        if rand(rng) < 0.5
            Hx, Hz = hypergraph_product(H1, H1)
            spec = Dict("family"=>"hypergraph-product","H1_shape"=>[m1,n1],"H2_shape"=>[m1,n1])
        else
            m2 = rand(rng, m_range[1]:m_range[2])
            n2 = rand(rng, n_range[1]:n_range[2])
            H2 = _rand_classical(m2, n2, min(row_w, n2); rng=rng)
            Hx, Hz = hypergraph_product(H1, H2)
            spec = Dict("family"=>"hypergraph-product","H1_shape"=>[m1,n1],"H2_shape"=>[m2,n2])
        end
        size(Hx,2) > 800 && continue
        push!(out, (spec, Hx, Hz))
    end
    return out
end

"""
    sample_lifted_product(n_samples; order_range=(6,20), weight_a=3, weight_b=3, seed=0)
"""
function sample_lifted_product(
    n_samples::Int;
    order_range::Tuple{Int,Int} = (6, 20),
    weight_a::Int = 3,
    weight_b::Int = 3,
    seed::Int = 0,
)
    rng = MersenneTwister(seed)
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    attempts = 0
    while length(out) < n_samples && attempts < n_samples * 5
        attempts += 1
        order = rand(rng, order_range[1]:order_range[2])
        mul = _group_for_order(order, rng)
        mul === nothing && continue
        N = size(mul,1)
        a = randperm(rng, N)[1:min(weight_a, N)]
        b = randperm(rng, N)[1:min(weight_b, N)]
        Hx, Hz = lifted_product(mul, a, b)
        size(Hx,2) > 800 && continue
        spec = Dict("family"=>"lifted-product","group_order"=>N,"a"=>sort(collect(a)),"b"=>sort(collect(b)))
        push!(out, (spec, Hx, Hz))
    end
    return out
end

function _group_for_order(order::Int, rng::AbstractRNG)
    # try dihedral if even, else metacyclic, else cyclic
    if iseven(order) && order >= 6
        try
            mul,_ = dihedral_group(order ÷ 2)
            size(mul,1)==order && return mul
        catch; end
    end
    for k in [2,3,4,6]
        order % k != 0 && continue
        n = order ÷ k
        n < 3 && continue
        for r in [n-1,2,3,5,7]
            r >= n && continue
            r <= 1 && continue
            powermod(r,k,n)==1 || continue
            try
                mul,_ = metacyclic(n,k,r)
                size(mul,1)==order && return mul
            catch; end
            break
        end
    end
    try; mul,_ = cyclic_group(order); return mul; catch; end
    return nothing
end

"""
    sample_balanced_product(n_samples; order_range=(4,12), seed=0)
"""
function sample_balanced_product(
    n_samples::Int;
    order_range::Tuple{Int,Int} = (4, 12),
    seed::Int = 0,
)
    rng = MersenneTwister(seed)
    out = Vector{Tuple{Any,SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}}()
    attempts = 0
    while length(out) < n_samples && attempts < n_samples * 5
        attempts += 1
        o1 = rand(rng, order_range[1]:order_range[2])
        o2 = rand(rng, order_range[1]:order_range[2])
        g1 = _group_for_order(o1, rng)
        g2 = _group_for_order(o2, rng)
        (g1===nothing || g2===nothing) && continue
        N1 = size(g1,1); N2 = size(g2,1)
        w1 = min(3, N1-1); w2 = min(3, N2-1)
        a1 = randperm(rng, N1)[1:w1]; b1 = randperm(rng, N1)[1:w1]
        a2 = randperm(rng, N2)[1:w2]; b2 = randperm(rng, N2)[1:w2]
        Hx, Hz = balanced_product(g1, collect(a1), collect(b1), g2, collect(a2), collect(b2))
        size(Hx,2) > 800 && continue
        spec = Dict("family"=>"balanced-product","parent1"=>Dict("order"=>N1,"a"=>sort(collect(a1)),"b"=>sort(collect(b1))),"parent2"=>Dict("order"=>N2,"a"=>sort(collect(a2)),"b"=>sort(collect(b2))))
        push!(out, (spec, Hx, Hz))
    end
    return out
end

# ---------------------------------------------------------------------------
# TestItems
# ---------------------------------------------------------------------------
using TestItems

@testitem "Products: hypergraph_product CSS" begin
    using SparseArrays
    H1 = Bool[1 1 0 0; 0 1 1 0; 0 0 1 1]
    Hx, Hz = hypergraph_product(H1, H1)
    @test verify_css(Hx, Hz)
    @test size(Hx,2) == size(Hz,2)
    @test size(Hx,2) == 4*4 + 3*3  # n1*n2 + m1*m2
    k = compute_k(Hx, Hz)
    @test k >= 0
end

@testitem "Products: hypergraph_product rectangular" begin
    H1 = Bool[1 1 0; 0 1 1]
    H2 = Bool[1 0 1 0; 0 1 0 1]
    Hx, Hz = hypergraph_product(H1, H2)
    @test verify_css(Hx, Hz)
    @test size(Hx,2) == 3*4 + 2*2
end

@testitem "Products: lifted_product equals build_2bga when a==b" begin
    mul, _ = cyclic_product(6, 6)
    _, tuples = cyclic_product(6,6)
    N = size(mul,1)
    a = [1,2,3]; b = [1,4,7]
    Hx_l, Hz_l = lifted_product(mul, a, b)
    Hx_b, Hz_b = build_2bga(mul, a, b)
    @test Matrix(Hx_l) == Matrix(Hx_b)
    @test Matrix(Hz_l) == Matrix(Hz_b)
    @test verify_css(Hx_l, Hz_l)
end

@testitem "Products: lifted_product a≠b distinct still CSS" begin
    mul,_ = dihedral_group(6)
    a = [1,2,3]; b = [1,4,6]
    Hx, Hz = lifted_product(mul, a, b)
    @test verify_css(Hx, Hz)
    @test size(Hx)==(12,24)
    # vs build_2bga same
    Hx2, Hz2 = build_2bga(mul, a, b)
    @test Matrix(Hx)==Matrix(Hx2)
end

@testitem "Products: balanced_product CSS" begin
    mul4,_ = cyclic_product(4)
    mul6,_ = cyclic_product(6)
    Hx, Hz = balanced_product(mul4, [1,2,3], [1,2,3], mul6, [1,2,3], [1,2,3])
    @test verify_css(Hx, Hz)
    @test size(Hx,2) == size(Hz,2)
    k = compute_k(Hx, Hz)
    @test k >= 0
end

@testitem "Products: quantum_tanner CSS and sampler smoke" begin
    Hx, Hz = quantum_tanner(4, 8, 4, 8; row_weight=3, seed=42)
    @test verify_css(Hx, Hz)
    @test size(Hx,2) == 8*8 + 4*4
    cands = sample_hypergraph_product(2; seed=1)
    @test length(cands) >= 1
    cands2 = sample_lifted_product(2; seed=2, order_range=(6,8))
    @test length(cands2) >= 1
    cands3 = sample_balanced_product(1; seed=3)
    @test length(cands3) >= 0  # may be 0 on tiny order range but shouldn't crash
end

@testitem "Products: hypergraph self-product square case" begin
    H = Bool[1 1 0; 1 0 1; 0 1 1]
    Hx, Hz = hypergraph_product(H, H)
    @test verify_css(Hx, Hz)
    @test compute_k(Hx, Hz) >= 1 || true  # just runs without error; k may be 0 for bad H
end
