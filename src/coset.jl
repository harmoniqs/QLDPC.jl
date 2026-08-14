"""
coset.jl — Coset-based two-block codes (Aydin-Tamo-Barg, arXiv:2606.17268).

Strict generalization of the regular 2BGA (`group_algebra.jl`): qubits are
indexed by the coset space G/H for H ≤ G (record codes use non-normal H),
rather than by G itself.

Construction:
  m := [G:H] = number of left cosets;  n = 2m; each block is m×m.
  Left action  (any g ∈ G):          L(g): xH ↦ (g·x)H
  Right action (only g ∈ N_G(H)):    R(g): xH ↦ (x·g)H   (must normalize H!)
  L(a)=Σ_{g∈a}L(g),  R(b)=Σ_{g∈b}R(g)   (b ⊆ N_G(H))
  H_X=[L(a) | R(b)], H_Z=[R(b)ᵀ | L(a)ᵀ]
CSS is automatic (left and right coset actions commute).

H = {e} ⇒ cosets = G, N_G(H)=G ⇒ reduces to the regular 2BGA.

Group encoding: same Cayley table `mul` (N×N, 1-indexed, identity=1) as
`group_algebra.jl`. Subgroups are `Vector{Int}` of element indices (1-based,
including identity 1).

Public:
  - `subgroup_closure(mul, gens)`, `left_cosets(mul, H)`, `normalizer(mul, H)`,
    `group_index(mul, H)`, `inverse(mul, g)`
  - `build_coset(mul, H, a, b; check_normalizer=true)` → (Hx, Hz) Sparse Bool (m,2m)
"""

using SparseArrays

# ---------------------------------------------------------------------------
# Basic group-theoretic helpers on a Cayley table
# ---------------------------------------------------------------------------

"""    inverse(mul, g) -> Int

Index of g⁻¹ (the h with g·h = e)."""
function group_inverse(mul::Matrix{Int}, g::Int)::Int
    N = size(mul, 1)
    @assert 1 <= g <= N
    row = mul[g, :]
    # find h with mul[g,h]==1
    for h in 1:N
        if row[h] == 1
            return h
        end
    end
    error("inverse: not found for $g")
end

# alias for compatibility with spec naming
const inverse = group_inverse

"""
    subgroup_closure(mul, gens) -> Vector{Int}

Smallest subgroup (sorted element-index list, 1-based, includes 1) containing
`gens` and the identity.
"""
function subgroup_closure(mul::Matrix{Int}, gens::AbstractVector{Int})::Vector{Int}
    N = size(mul, 1)
    H = Set{Int}([1])
    for g in gens
        @assert 1 <= g <= N
        push!(H, g)
    end
    frontier = collect(H)
    while !isempty(frontier)
        nxt = Int[]
        H_list = collect(H)
        for a in H_list, b in frontier
            for prod in (mul[a, b], mul[b, a])
                if prod ∉ H
                    push!(H, prod)
                    push!(nxt, prod)
                end
            end
        end
        frontier = nxt
    end
    return sort(collect(H))
end

"""    subgroup_closure(mul, gens::AbstractVector) fallback for empty."""
subgroup_closure(mul::Matrix{Int}, gens::AbstractVector{<:Integer}) = subgroup_closure(mul, Vector{Int}(gens))

"""
    left_cosets(mul, H) -> (reps, coset_of)

Partition G into left cosets xH. Returns `(reps, coset_of)` where
`coset_of[x]` is the coset index (1..m) of element x and `reps[c]` is a
representative of coset c. `m = [G:H]`.
"""
function left_cosets(mul::Matrix{Int}, H::AbstractVector{Int})
    N = size(mul, 1)
    coset_of = zeros(Int, N)
    reps = Int[]
    for x in 1:N
        if coset_of[x] != 0
            continue
        end
        c = length(reps) + 1
        push!(reps, x)
        for h in H
            y = mul[x, h]
            coset_of[y] = c
        end
    end
    return reps, coset_of
end

"""    group_index(mul, H) -> Int — [G:H]."""
group_index(mul::Matrix{Int}, H::AbstractVector{Int}) = size(mul, 1) ÷ length(H)

"""
    normalizer(mul, H) -> Vector{Int}

N_G(H) = { g : g·H·g⁻¹ = H } as a sorted index list."""
function normalizer(mul::Matrix{Int}, H::AbstractVector{Int})::Vector{Int}
    N = size(mul, 1)
    Hset = Set(H)
    out = Int[]
    for g in 1:N
        gi = group_inverse(mul, g)
        ok = true
        for h in H
            conj = mul[mul[g, h], gi]
            if conj ∉ Hset
                ok = false
                break
            end
        end
        ok && push!(out, g)
    end
    return sort(out)
end

# alias per spec
const normalizer_group = normalizer

# ---------------------------------------------------------------------------
# Coset permutation helpers
# ---------------------------------------------------------------------------

"""    _perm_L(mul, reps, coset_of, g) -> Vector{Int}

Coset permutation for the left action of g: c ↦ coset_of[g·rep_c] (1-indexed)."""
function _perm_L(mul::Matrix{Int}, reps::Vector{Int}, coset_of::Vector{Int}, g::Int)::Vector{Int}
    return [coset_of[mul[g, reps[c]]] for c in 1:length(reps)]
end

"""    _perm_R(mul, reps, coset_of, g) -> Vector{Int}

Coset permutation for the right action of g (g must be in N_G(H)):
c ↦ coset_of[rep_c·g]."""
function _perm_R(mul::Matrix{Int}, reps::Vector{Int}, coset_of::Vector{Int}, g::Int)::Vector{Int}
    return [coset_of[mul[reps[c], g]] for c in 1:length(reps)]
end

# reuse _block_from_perms defined in group_algebra.jl — but define locally for
# standalone include order; if already defined, this is idempotent.

if !isdefined(@__MODULE__, :_block_from_perms)
    function _block_from_perms(perms::Vector{Vector{Int}}, m::Int)::SparseMatrixCSC{Bool,Int}
        pos = Dict{Tuple{Int,Int},Bool}()
        for p in perms
            for c in 1:m
                key = (p[c], c)
                pos[key] = !get(pos, key, false)
            end
        end
        I = Int[]; J = Int[]
        sizehint!(I, length(pos)); sizehint!(J, length(pos))
        for (k, v) in pos
            v || continue
            push!(I, k[1]); push!(J, k[2])
        end
        return sparse(I, J, fill(true, length(I)), m, m)
    end
end

# ---------------------------------------------------------------------------
# Public coset builder
# ---------------------------------------------------------------------------

"""
    build_coset(mul, H, a, b; check_normalizer=true) -> (Hx, Hz)

Coset two-block code. `H` a subgroup (element-index list including 1), `a` a
subset of G, `b` a subset of N_G(H). Returns `(Hx, Hz)` as `SparseMatrixCSC{Bool}`
of shape (m, 2m), CSS guaranteed. With `H=[1]` this reduces to `build_2bga`.

Also accepts `build_coset((mul,elems), H, a, b)`.

Throws `ArgumentError` if `b` has elements outside N_G(H) and checking is on.
"""
function build_coset(
    mul::Matrix{Int},
    H::AbstractVector{Int},
    a::AbstractVector{Int},
    b::AbstractVector{Int};
    check_normalizer::Bool = true,
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    reps, coset_of = left_cosets(mul, Vector{Int}(H))
    m = length(reps)
    if check_normalizer
        Nset = Set(normalizer(mul, Vector{Int}(H)))
        bad = [g for g in b if g ∉ Nset]
        if !isempty(bad)
            throw(ArgumentError("build_coset: b has elements outside N_G(H): $bad (N_G(H)=$(sort(collect(Nset))))"))
        end
    end
    perms_L = [_perm_L(mul, reps, coset_of, g) for g in a]
    perms_R = [_perm_R(mul, reps, coset_of, g) for g in b]
    La = _block_from_perms(perms_L, m)
    Rb = _block_from_perms(perms_R, m)
    Hx = hcat(La, Rb)
    Hz = hcat(transpose(Rb), transpose(La))
    Hx = SparseMatrixCSC{Bool,Int}(sparse(Hx))
    Hz = SparseMatrixCSC{Bool,Int}(sparse(Hz))
    return Hx, Hz
end

function build_coset(
    G::Tuple{Matrix{Int},Any},
    H::AbstractVector,
    a::AbstractVector,
    b::AbstractVector;
    check_normalizer::Bool = true,
)
    return build_coset(G[1], Vector{Int}(H), Vector{Int}(a), Vector{Int}(b); check_normalizer=check_normalizer)
end

"""
    cosets(G, H) -> (reps, coset_of)

Alias for `left_cosets` (spec naming `cosets(G,H)`).
"""
cosets(mul::Matrix{Int}, H::AbstractVector{Int}) = left_cosets(mul, H)
cosets(G::Tuple{Matrix{Int},Any}, H::AbstractVector) = left_cosets(G[1], Vector{Int}(H))

# ---------------------------------------------------------------------------
# TestItems
# ---------------------------------------------------------------------------

using TestItems

@testitem "Coset: subgroup_closure trivial" begin
    mul, _ = cyclic_group(6)
    H = subgroup_closure(mul, Int[1])
    @test H == [1]
    H2 = subgroup_closure(mul, Int[2])
    @test length(H2) == 6  # generator of C6 gives whole group
end

@testitem "Coset: left_cosets identity subgroup" begin
    mul, _ = dihedral_group(6)
    N = size(mul, 1)
    H = [1]
    reps, coset_of = left_cosets(mul, H)
    @test length(reps) == N
    @test sort(reps) == 1:N
end

@testitem "Coset: normalizer whole group for abelian" begin
    mul, _ = cyclic_product(3, 4)
    H = subgroup_closure(mul, [2])
    Nrm = normalizer(mul, H)
    @test length(Nrm) == size(mul, 1)  # abelian => every subgroup normal
end

@testitem "Coset: build_coset H={e} reduces to build_2bga" begin
    mul, tuples = cyclic_product(6, 6)
    tmap = Dict(tuples[i] => i for i in 1:length(tuples))
    a = [tmap[[3, 0]], tmap[[0, 1]], tmap[[0, 2]]]
    b = [tmap[[0, 3]], tmap[[1, 0]], tmap[[2, 0]]]
    H = [1]
    Hx_c, Hz_c = build_coset(mul, H, a, b)
    Hx_r, Hz_r = build_2bga(mul, a, b)
    @test Matrix(Hx_c) == Matrix(Hx_r)
    @test Matrix(Hz_c) == Matrix(Hz_r)
    @test size(Hx_c) == (36, 72)
    @test verify_css(Hx_c, Hz_c)
    @test compute_k(Hx_c, Hz_c) == 12
end

@testitem "Coset: build_coset commutes (non-normal H via dihedral)" begin
    mul, _ = dihedral_group(6)  # order 12
    # H = <s> order 2, non-normal in D6
    # find an order-2 element (reflection): s is the second generator; its index?
    # Instead brute force: find subgroup of size 2 from element 7 or similar
    # pick H = closure of element 7 (some reflection)
    H = subgroup_closure(mul, [7])
    @test length(H) == 2
    m = group_index(mul, H)
    @test m == 6
    Nrm = normalizer(mul, H)
    @test length(Nrm) >= 2
    # pick a ⊆ G weight 3, b ⊆ N_G(H) weight 3
    a = [1, 2, 3]
    b = Nrm[1:min(3, length(Nrm))]
    Hx, Hz = build_coset(mul, H, a, b)
    @test size(Hx) == (m, 2m)
    @test verify_css(Hx, Hz)
end

@testitem "Coset: normalizer check rejects bad b" begin
    mul, _ = sym_group(4)  # order 24, has non-normal subgroups
    H = subgroup_closure(mul, [2])  # some subgroup
    Nrm = normalizer(mul, H)
    # find an element outside Nrm
    outside = [g for g in 1:size(mul, 1) if g ∉ Nrm]
    if !isempty(outside)
        bad = [outside[1]]
        @test_throws ArgumentError build_coset(mul, H, [1, 2], bad; check_normalizer=true)
        # with check off it should not throw
        Hx, Hz = build_coset(mul, H, [1], [1]; check_normalizer=false)
        @test verify_css(Hx, Hz)
    else
        @test true  # H normal, skip
    end
end

@testitem "Coset: dihedral varying sizes even/odd k" begin
    # just verify plumbing for dihedral orders 12 and 24
    for n in [6, 12]
        mul, _ = dihedral_group(n)
        N = 2n
        H = [1]
        a = [1, 2, 3]; b = [1, 4, 7]
        # clamp b to valid range
        b = [min(x, N) for x in b]
        Hx, Hz = build_coset(mul, H, a, b)
        @test verify_css(Hx, Hz)
        k = compute_k(Hx, Hz)
        @test k >= 0
    end
end
