@testitem "JET correctness analysis" tags = [:jet] begin
    if VERSION >= v"1.12"
        using JET, QLDPC
        JET.test_package(QLDPC; target_modules = (QLDPC,))
    else
        @info "Skipping JET on Julia $VERSION (requires >= 1.12)"
        @test true
    end
end
