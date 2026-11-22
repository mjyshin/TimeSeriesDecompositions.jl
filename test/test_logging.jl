using GPDecompositions
using Logging
using Test

# Top level, where a closure cannot assign the caller's variables
module QuietlyScope
using GPDecompositions, Logging
x = 0
for i in 1:3
    @quietly global x = i
end
@quietly u = 4
@quietly Logging.Warn y, z = 1, 2
w = [0]
@quietly w .= 5
end

@testset "quietly filters the current logger" begin
    @test_logs (:warn, "kept") quietly(level = Logging.Warn) do
        @info "dropped"
        @warn "kept"
    end
    @test_logs (:warn, "kept") @quietly Logging.Warn begin
        @info "dropped"
        @warn "kept"
    end
    @test_logs @quietly @warn "dropped"
    @test (@quietly 1 + 1) == 2
    @test_throws ErrorException @quietly error("still thrown")
end

@testset "@quietly assigns in the caller's scope" begin
    @test QuietlyScope.x == 3
    @test (QuietlyScope.y, QuietlyScope.z) == (1, 2)
    @test QuietlyScope.w == [5]
    @test QuietlyScope.u == 4
    function f()
        a = 0
        @quietly a = 7
        local b = @quietly 1 + 1
        @quietly a += b
        return a
    end
    @test f() == 9
    @test_logs (:warn, "kept") begin
        @quietly Logging.Warn v = (@info("dropped"); @warn("kept"); 1)
        @test v == 1
    end
end

@testset "quietly silences fit" begin
    t = collect(1.0:30.0)
    y = sin.(2π .* t ./ 6) .+ 0.05 .* randn(30)
    @test_logs min_level = Logging.Info (@quietly fit(LOESS, y, t)) isa FittedSmoother
end
