using Test

@testset "GPEstimators" begin
    include("test_api.jl")
    include("test_decompose.jl")
end
