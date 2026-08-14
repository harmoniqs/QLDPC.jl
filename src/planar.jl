"""
planar.jl — Open-boundary planar tile codes (Liang-Eberhardt-Chen 2025).

Mirrors `research/local2d/planar.py` (arXiv:2504.08887, Sec II / Table V):
a torus bivariate-bicycle code with polynomials f, g truncated to an open
Lx × Ly grid with directional anyon condensation.

Qubit indexing: A(i,j) idx = i*Ly + j;  B(i,j) idx = Lx*Ly + i*Ly + j.
Hx rows = [f on A | g on B];  Hz rows = [gbar on A | fbar on B].
Boundary condensation:
  X boundary terms may hang off the top/bottom edges (row-out, column-in),
  Z boundary terms may hang off the left/right edges (column-out, row-in).
Corner anticommutation resolved by greedy drop of conflicting X rows
(paper's footnote-6 convention). Validated flagship [[288,8,12]] family:
  f = x + x² + y², g = 1 + x²y + x²y²  gives k=8 for L=6..12, r ≤ 7.

This is the tile family that earns the `local-2d-bilayer` locality class:
qubits live on a bilayer grid, every check has bounded radius. Use with
`make_submission(...; coordinates=grid_coordinates(Lx,Ly), layers=2)` to
measure locality.

Public:
  - `build_planar(Lx, Ly, f_terms, g_terms) -> (Hx, Hz)`
  - `grid_coordinates(Lx, Ly; kept=nothing) -> Vector{Vector{Int}}`
  - `graft(codes...)` — stitch small planar tiles into larger surface
"""

using SparseArrays

# Default flagship polynomials of the [[288,8,12]] code (Eq.9 in paper)
# Supports as (di, dj) offsets
const DEFAULT_F = [(1,0),(2,0),(0,2)]
const DEFAULT_G = [(0,0),(2,1),(2,2)]
const DEFAULT_FBAR = [(-1,0),(-2,0),(0,-2)]
const DEFAULT_GBAR = [(0,0),(-2,-1),(-2,-2)]

"""
    build_planar(Lx, Ly; f_terms=DEFAULT_F, g_terms=DEFAULT_G) -> (Hx, Hz)

Open-boundary CSS code on Lx × Ly grid, n = 2*Lx*Ly.
`f_terms`, `g_terms` are vectors of (di, dj) integer offsets.

For the flagship family `f=[(1,0),(2,0),(0,2)]`, `g=[(0,0),(2,1),(2,2)]`
the construction is validated to give k=8 and distance scaling 4,6,9 for
6×6, 8×8, 10×10 (Table V). Other (f,g) are supported as "a CSS code" but
may need k(L)-stability screening (see planar.py docstring).

Radius is bounded by max |di|,|dj|; for default polynomials r ≤ 3.
"""
function build_planar(
    Lx::Int,
    Ly::Int;
    f_terms::AbstractVector = DEFAULT_F,
    g_terms::AbstractVector = DEFAULT_G,
    resolve_corners::Bool = true,
)::Tuple{SparseMatrixCSC{Bool,Int},SparseMatrixCSC{Bool,Int}}
    # derive fbar/gbar as negatives (over the torus, fbar = f(x⁻¹,y⁻¹))
    fbar = [(-d[1], -d[2]) for d in f_terms]
    gbar = [(-d[1], -d[2]) for d in g_terms]
    return build_open_directional(Lx, Ly, f_terms, g_terms, fbar, gbar; resolve_corners=resolve_corners)
end

"""
    build_planar(Lx, Ly, f_terms, g_terms) -> (Hx, Hz)

Positional overload for `f_terms`, `g_terms`.
"""
function build_planar(
    Lx::Int,
    Ly::Int,
    f_terms::AbstractVector,
    g_terms::AbstractVector;
    resolve_corners::Bool = true,
)
    fbar = [(-d[1], -d[2]) for d in f_terms]
    gbar = [(-d[1], -d[2]) for d in g_terms]
    return build_open_directional(Lx, Ly, f_terms, g_terms, fbar, gbar; resolve_corners=resolve_corners)
end

# core builder with explicit f,g,fbar,gbar (mirrors planar.py build_open_directional)
function build_open_directional(
    Lx::Int,
    Ly::Int,
    S_f::AbstractVector,
    S_g::AbstractVector,
    S_fbar::AbstractVector,
    S_gbar::AbstractVector;
    resolve_corners::Bool = true,
)
    n2 = Lx * Ly
    n  = 2 * n2

    # helper index
    inb(i,j) = (0 <= i < Lx) && (0 <= j < Ly)
    row_in(i) = 0 <= i < Lx
    col_in(j) = 0 <= j < Ly

    all_offs = vcat(collect(S_f), collect(S_g), collect(S_fbar), collect(S_gbar))
    di_hi = maximum(d->d[1], all_offs)
    di_lo = minimum(d->d[1], all_offs)
    dj_hi = maximum(d->d[2], all_offs)
    dj_lo = minimum(d->d[2], all_offs)

    # keep predicates for out-of-bounds qubits:
    # X may hang off row-out/col-in, Z off row-in/col-out
    x_keep(i,j) = col_in(j) && !row_in(i)
    z_keep(i,j) = row_in(i) && !col_in(j)

    function make_row(ci, cj, suppA, suppB, keep_oob)
        row = zeros(Bool, n)
        any_in = false
        for (di, dj) in suppA
            i = ci + di; j = cj + dj
            if inb(i,j)
                idx = i*Ly + j + 1  # A block 1..n2
                row[idx] = !row[idx]
                any_in = true
            elseif !keep_oob(i,j)
                return nothing
            end
        end
        for (di, dj) in suppB
            i = ci + di; j = cj + dj
            if inb(i,j)
                idx = n2 + i*Ly + j + 1  # B block n2+1..2n2
                row[idx] = !row[idx]
                any_in = true
            elseif !keep_oob(i,j)
                return nothing
            end
        end
        if !any_in || !any(row)
            return nothing
        end
        return row
    end

    xrows = Vector{Vector{Bool}}()
    zrows = Vector{Vector{Bool}}()
    for ci in (-di_hi):(Lx-1 - di_lo)
        for cj in (-dj_hi):(Ly-1 - dj_lo)
            rx = make_row(ci, cj, S_f, S_g, x_keep)
            if rx !== nothing; push!(xrows, rx); end
            rz = make_row(ci, cj, S_gbar, S_fbar, z_keep)
            if rz !== nothing; push!(zrows, rz); end
        end
    end

    # assemble sparse Bool matrices
    function rows_to_sparse(rows::Vector{Vector{Bool}}, cols::Int)
        m = length(rows)
        m == 0 && return spzeros(Bool, 0, cols)
        I = Int[]; J = Int[]
        sizehint!(I, m*6); sizehint!(J, m*6)
        for i in 1:m
            r = rows[i]
            for j in 1:cols
                if r[j]
                    push!(I, i); push!(J, j)
                end
            end
        end
        return sparse(I, J, fill(true, length(I)), m, cols)
    end

    Hx = rows_to_sparse(xrows, n)
    Hz = rows_to_sparse(zrows, n)

    if resolve_corners && size(Hx,1) > 0 && size(Hz,1) > 0
        # greedy corner resolution: drop X rows that anticommute with any Z row
        # (Hx * Hz' mod 2) row conflicts
        Hxd = Matrix{Bool}(Hx)
        Hzd = Matrix{Bool}(Hz)
        # compute overlap: (Hx * Hz') mod2
        # dense since m small (few hundred)
        conflict = falses(size(Hx,1))
        for i in 1:size(Hx,1)
            for j in 1:size(Hz,1)
                s = false
                @inbounds for k in 1:n
                    s = s ⊻ (Hxd[i,k] & Hzd[j,k])
                end
                if s
                    conflict[i] = true
                    break
                end
            end
        end
        if any(conflict)
            keep = findall(!, conflict)
            Hx = Hx[keep, :]
        end
    end

    Hx = SparseMatrixCSC{Bool,Int}(Hx)
    Hz = SparseMatrixCSC{Bool,Int}(Hz)
    return Hx, Hz
end

"""
    grid_coordinates(Lx, Ly; kept=nothing) -> Vector{Vector{Int}}

Per-qubit [x, y] layout for the bilayer grid, for
`make_submission(coordinates=..., layers=2)`.

A(i,j) and B(i,j) share site (i, j) on two layers. `kept` restricts to
surviving qubits after gauge fixing, in post-cleanup order.
"""
function grid_coordinates(
    Lx::Int,
    Ly::Int;
    kept::Union{Nothing,AbstractVector{Int}} = nothing,
)::Vector{Vector{Int}}
    n2 = Lx * Ly
    coords = Vector{Vector{Int}}(undef, 2*n2)
    for i in 0:Lx-1, j in 0:Ly-1
        idx_a = i*Ly + j + 1
        idx_b = n2 + i*Ly + j + 1
        coords[idx_a] = [i, j]
        coords[idx_b] = [i, j]
    end
    if kept !== nothing
        coords = [coords[k] for k in kept]
    end
    return coords
end

"""
    graft(codes::Tuple...) -> (Hx, Hz, coords)

Stitch planar tiles side-by-side into a larger surface (experimental).

Given a list of `(Hx, Hz, Lx, Ly)` tile tuples, concatenate them horizontally
by offsetting coordinates and block-diagonal stacking the checks.
Minimal implementation: just returns the first tile's code (the "tile" family
boards are earned by `build_planar` alone; graft is for future multi-tile
surfaces).
"""
function graft(tiles::Tuple...)
    @assert !isempty(tiles) "graft: need at least one tile"
    # minimal: return first tile, extend later for true block-diagonal graft
    Hx, Hz = tiles[1][1], tiles[1][2]
    return Hx, Hz
end

# ---------------------------------------------------------------------------
# Helpers for locality / radius reporting
# ---------------------------------------------------------------------------

"""
    planar_radius(f_terms, g_terms) -> Int

Maximum Chebyshev radius of checks (max |di|,|dj|).
For the local-2d track with r ≤ 7 (paper uses r=3 for flagship).
"""
function planar_radius(
    f_terms::AbstractVector = DEFAULT_F,
    g_terms::AbstractVector = DEFAULT_G,
)::Int
    all = vcat(collect(f_terms), collect(g_terms))
    return maximum(max(abs(d[1]), abs(d[2])) for d in all)
end

# ---------------------------------------------------------------------------
# TestItems
# ---------------------------------------------------------------------------
using TestItems

@testitem "Planar: flagship 6x6 flagship CSS and k=8" begin
    Hx, Hz = build_planar(6, 6)
    @test verify_css(Hx, Hz)
    @test size(Hx,2) == 72
    @test compute_k(Hx, Hz) == 8
end

@testitem "Planar: flagship 8x8 CSS" begin
    Hx, Hz = build_planar(8, 8)
    @test verify_css(Hx, Hz)
    @test compute_k(Hx, Hz) == 8
end

@testitem "Planar: radius check flagship r<=3" begin
    @test planar_radius() <= 3
    @test planar_radius([(3,0),(0,1)], [(0,0),(1,2)]) <= 3
end

@testitem "Planar: grid_coordinates shape and bilayer" begin
    coords = grid_coordinates(4, 3)
    @test length(coords) == 24
    @test coords[1] == [0,0]
    @test coords[13] == [0,0]  # first B repeats site of first A
    # kept filtering
    kept = [1,2,13]
    c2 = grid_coordinates(4,3; kept=kept)
    @test length(c2)==3
    @test c2[1]==[0,0]
end

@testitem "Planar: custom f,g stays CSS" begin
    f = [(1,0),(0,1),(1,1)]
    g = [(0,0),(2,0),(0,2)]
    Hx, Hz = build_planar(5, 5, f, g)
    @test verify_css(Hx, Hz)
    @test size(Hx,2)==50
end

@testitem "Planar: weight class check max weight <=9" begin
    Hx, Hz = build_planar(6,6)
    w = max(row_weight(Hx), row_weight(Hz))
    @test w <= 9  # flagship family weight 6
end
