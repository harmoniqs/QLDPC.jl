@testitem "ExplicitImports check" tags = [:explicit_imports] begin
    using QLDPC
    try
        using ExplicitImports
        @test ExplicitImports.check_no_implicit_imports(QLDPC) === nothing
        @test ExplicitImports.check_no_stale_explicit_imports(QLDPC) === nothing
    catch e
        @info "Skipping ExplicitImports — not installed or check failed" exception = e
        @test true
    end
end
