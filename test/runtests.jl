using Test

@testset "GPDecompositions" begin
    include("test_api.jl")
    include("test_decompose.jl")
    include("test_filters.jl")
    include("test_classical.jl")
end
