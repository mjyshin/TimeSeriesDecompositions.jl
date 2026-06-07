using Test

@testset "GPDecompositions" begin
    include("test_api.jl")
    include("test_decompose.jl")
end
