using Logging
using Test

# Hyperparameters chosen from the data are reported with @info; tests that check those
# messages capture them with @test_logs
include("setup.jl")

with_logger(ConsoleLogger(stderr, Logging.Warn)) do
    @testset "TimeSeriesDecompositions" begin
        @testset "quality" include("test_quality.jl")
        @testset "logging" include("test_logging.jl")
        @testset "API" include("test_api.jl")
        @testset "Gaussian process" include("test_decompose.jl")
        @testset "smoothers" include("test_smoothers.jl")
        @testset "X11 and STL" include("test_classical.jl")
    end
end
