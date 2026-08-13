"""
bb.jl — Bivariate-bicycle (BB) codes on the torus Z_l × Z_m.

Mirrors `research/kit/bb.py` (Bravyi et al. gross code family).

Construction:
  x = S_l ⊗ I_m,  y = I_l ⊗ S_m,  monomial x^a y^b = S_l^a ⊗ S_m^b
  A = Σ_{(a,b)∈A_terms} x^a y^b,  B = Σ_{(c,d)∈B_terms} x^c y^d
  Hx = [A | B],  Hz = [B' | A']   (n = 2*l*m, CSS is automatic)

Exposed:
  - `build_bb(l,m,A_terms,B_terms)` -> (Hx, Hz) as SparseMatrixCSC{Bool}
  - `poly_matrix(l,m,terms)` -> SparseMatrixCSC{Bool}
  - `KNOWN_CODES` — Bravyi et al. records for validation

Types:
  Terms are `Vector{Tuple{Int,Int}}` with exponents (a,b).
"""

using SparseArrays

"""
    monomial_matrix(l, m, a, b) -> SparseMatrixCSC{Bool}

Permutation matrix for x^a y^b on l*m qubits indexed q = i*m + j.
M[p+1,q+1]=1 where p = ((i+a)%l)*m + ((j+b)%m).
"""
function monomial_matrix(l::Int, m::Int, a::Int, b::Int)::SparseMatrixCSC{Bool,Int}
    N = l * m
    a_mod = mod(a, l)
    b_mod = mod(b, m)
    I = Vector{Int}(undef, N)
    J = Vector{Int}(undef, N)
    V = fill(true, N)
    for q = 0:N-1
        i = div(q, m)
        j = mod(q, m)
        p = mod(i + a_mod, l) * m + mod(j + b_mod, m)
        I[q+1] = p + 1
        J[q+1] = q + 1
    end
    return sparse(I, J, V, N, N)
end

"""
    poly_matrix(l, m, terms) -> SparseMatrixCSC{Bool}

Sum (mod 2) of monomials for `terms = [(a,b), ...]`.
"""
function poly_matrix(
    l::Int,
    m::Int,
    terms::Vector{Tuple{Int,Int}},
)::SparseMatrixCSC{Bool,Int}
    N = l * m
    M = spzeros(Bool, N, N)
    for (a, b) in terms
        Mm = monomial_matrix(l, m, a, b)
        # XOR (mod 2 sum): M = M .⊻ Mm  but sparse — do via nz handling
        # Simple: convert to dense for XOR at these sizes, then back to sparse
        # For N≤144 this is free and avoids sparse xor edge cases
        Md = Matrix(M) .⊻ Matrix(Mm)
        M = sparse(Md)
    end
    return M
end

# Convenience overload for Vector{Vector{Int}} or other tuple shapes
function poly_matrix(l::Int, m::Int, terms::AbstractVector)::SparseMatrixCSC{Bool,Int}
    t2 = Tuple{Int,Int}[(Int(t[1]), Int(t[2])) for t in terms]
    return poly_matrix(l, m, t2)
end

"""
    build_bb(l, m, A_terms, B_terms) -> (Hx, Hz)

Build the periodic BB code on Z_l × Z_m. Each `H` is `(l*m) × (2*l*m)`.
"""
function build_bb(
    l::Int,
    m::Int,
    A_terms::AbstractVector,
    B_terms::AbstractVector,
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    A = poly_matrix(l, m, A_terms)
    B = poly_matrix(l, m, B_terms)
    Hx = hcat(A, B)
    Hz = hcat(sparse(Matrix(B)'), sparse(Matrix(A)'))
    # ensure Bool
    Hx = SparseMatrixCSC{Bool,Int}(Hx)
    Hz = SparseMatrixCSC{Bool,Int}(Hz)
    return Hx, Hz
end

"""
    KNOWN_CODES

Bravyi et al. BB records for validation (keys are "[[n,k,d]]").
Each value is a NamedTuple with l,m,A,B.
"""
const KNOWN_CODES = Dict{String,NamedTuple}(
    "[[72,12,6]]" =>
        (l = 6, m = 6, A = [(3, 0), (0, 1), (0, 2)], B = [(0, 3), (1, 0), (2, 0)]),
    "[[90,8,10]]" =>
        (l = 15, m = 3, A = [(9, 0), (0, 1), (0, 2)], B = [(0, 0), (2, 0), (7, 0)]),
    "[[108,8,10]]" =>
        (l = 9, m = 6, A = [(3, 0), (0, 1), (0, 2)], B = [(0, 3), (1, 0), (2, 0)]),
    "[[144,12,12]]" =>
        (l = 12, m = 6, A = [(3, 0), (0, 1), (0, 2)], B = [(0, 3), (1, 0), (2, 0)]),
    "[[288,12,18]]" =>
        (l = 12, m = 12, A = [(3, 0), (0, 2), (0, 7)], B = [(0, 3), (1, 0), (2, 0)]),
)

using TestItems

@testitem "BB: 72,12,6 shape and CSS" begin
    Hx, Hz = build_bb(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    @test size(Hx) == (36, 72)
    @test size(Hz) == (36, 72)
    @test verify_css(Hx, Hz)
    @test compute_k(Hx, Hz) == 12
end

@testitem "BB: poly_matrix is involutive sums mod 2" begin
    M1 = poly_matrix(3, 3, [(0, 0)])
    @test nnz(M1) == 9  # identity
    M2 = poly_matrix(3, 3, [(0, 0), (0, 0)])
    @test nnz(M2) == 0  # 1+1=0 mod2
end

@testitem "BB: KNOWN_CODES spot-check" begin
    p = KNOWN_CODES["[[144,12,12]]"]
    Hx, Hz = build_bb(p.l, p.m, p.A, p.B)
    @test verify_css(Hx, Hz)
    @test compute_k(Hx, Hz) == 12
end
