using Aqua
using ExplicitImports
using GPDecompositions
using Test

@testset "Aqua" begin
    Aqua.test_all(GPDecompositions; persistent_tasks = false)
end

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(GPDecompositions) === nothing
    @test check_no_stale_explicit_imports(GPDecompositions) === nothing
end
