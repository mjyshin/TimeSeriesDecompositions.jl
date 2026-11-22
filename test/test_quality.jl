using Aqua
using ExplicitImports
using TimeSeriesDecompositions
using Test

@testset "Aqua" begin
    Aqua.test_all(TimeSeriesDecompositions; persistent_tasks = false)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(TimeSeriesDecompositions) === nothing
    @test check_no_stale_explicit_imports(TimeSeriesDecompositions) === nothing
end
