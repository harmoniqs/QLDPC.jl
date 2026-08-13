@testitem "Aqua quality assurance" tags = [:aqua] begin
    using Aqua, QLDPC
    Aqua.test_all(
        QLDPC;
        stale_deps = (ignore = [:JLD2, :TestItemRunner],),
        deps_compat = (ignore = [:Test, :Aqua, :JET],),
    )
end
