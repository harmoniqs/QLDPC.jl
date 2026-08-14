module QLDPC

using Reexport
@reexport using SparseArrays
@reexport using LinearAlgebra
@reexport using Random

include("css.jl")
include("bb.jl")
include("group_algebra.jl")
include("coset.jl")
include("surrogate.jl")
include("search.jl")
include("submit.jl")
include("precompile.jl")

export verify_css, compute_k, CSSCode, ncode, kcode, weight, row_weight
export rank_gf2, rank_gf2_fast, rref_gf2, kernel_basis, logical_basis, in_rowspace, commutes
export build_bb, poly_matrix, monomial_matrix, KNOWN_CODES
export distance_rand, distance_rand_threaded, lightest_logical, lightest_logical_threaded
export _search_lightest, _search_lightest_threaded, _search_lightest_distributed
export efficiency, fingerprint, screen, pareto_frontier, sample_bb
export make_submission, save_submission, validate_candidate
export cyclic_group, cyclic_product, dihedral_group, metacyclic, sym_group, alt_group, perm_group, direct_product
export perm_matrix, left_regular_rep, right_regular_rep, build_2bga
export subgroup_closure, left_cosets, cosets, normalizer, group_index, group_inverse, inverse, build_coset

end
