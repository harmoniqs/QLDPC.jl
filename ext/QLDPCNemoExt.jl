module QLDPCNemoExt

using QLDPC
using Nemo

"""
    rank_nemo(M) -> Int

Internal helper for `rank_gf2_fast`: rank over GF(2) via Nemo.
Called by `QLDPC.rank_gf2_fast` when this extension is loaded.
"""
function rank_nemo(M::AbstractMatrix)::Int
    # Normalize to Int mod 2, dense
    A = Matrix{Int}(map(x -> Int(x) & 1, collect(M)))
    # Nemo.matrix(GF(2), ...) expects matrix over GF(2)
    F = Nemo.GF(2)
    Nm = Nemo.matrix(F, A)
    return Int(Nemo.rank(Nm))
end

# Overload fast path to use Nemo directly (world-age safe via extension)
function QLDPC.rank_gf2_fast(M::AbstractMatrix)::Int
    try
        return rank_nemo(M)
    catch
        return QLDPC._rank_gf2_fallback(M)
    end
end

# Transparently accelerate plain rank_gf2 when Nemo is present
function QLDPC.rank_gf2(M::AbstractMatrix)::Int
    try
        return rank_nemo(M)
    catch
        return QLDPC._rank_gf2_fallback(M)
    end
end

end # module
