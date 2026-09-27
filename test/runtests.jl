using Logging
using Test

# Hyperparameters chosen from the data are reported with @info; tests that check those
# messages capture them with @test_logs
with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    @testset "GPDecompositions" begin
        include("test_api.jl")
        include("test_decompose.jl")
        include("test_smoothers.jl")
        include("test_classical.jl")
    end
end
