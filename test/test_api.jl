using GPDecompositions
using Test

const _t = collect(1.0:50.0)
const _y = sin.(2π .* _t ./ 12) .+ 0.01 .* _t .+ 0.1 .* randn(50)

@testset "decompose returns Decomposition" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    @test d isa Decomposition
end

@testset "Component fields" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    @test !isnothing(d.seasonal)
    for comp in (d.trend, d.seasonal, d.noise)
        @test comp isa Component
        @test length(comp.mean) == 50
        @test length(comp.std)  == 50
        @test length(comp.lower) == length(comp.sig)
        @test length(comp.upper) == length(comp.sig)
        @test all(comp.std .≥ 0)
    end
    @test d.rate isa Component
end

@testset "no seasonal component (default)" begin
    d = fit(Decomposition, _y, _t)
    @test isnothing(d.seasonal)
    @test d.trend isa Component
    @test d.noise isa Component
end

@testset "no seasonal component (explicit Inf)" begin
    d = fit(Decomposition, _y, _t; period=Inf)
    @test isnothing(d.seasonal)
end

@testset "rate component (approx, NaN std)" begin
    d = fit(Decomposition, _y, _t; period=12.0, approx_rate=true)
    @test d.rate isa Component
    @test length(d.rate.mean) == 50
    @test all(isnan, d.rate.std)
    @test all(isnan, d.rate.lower[1])
    @test all(isnan, d.rate.upper[1])
end

@testset "rate component (full, finite std)" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    @test d.rate isa Component
    @test length(d.rate.mean) == 50
    @test all(isfinite, d.rate.std)
    @test all(d.rate.std .≥ 0)
end

@testset "sig levels" begin
    d = fit(Decomposition, _y, _t; period=12.0, sig=[0.68, 0.95])
    @test d.trend.sig == [0.68, 0.95]
    @test length(d.trend.lower) == 2
end

@testset "predict at new times" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    t_new = collect(51.0:60.0)
    d2 = predict(d, t_new)
    @test d2 isa Decomposition
    @test length(d2.t) == 10
    @test length(d2.trend.mean) == 10
end

@testset "predict preserves sig" begin
    d = fit(Decomposition, _y, _t; period=12.0, sig=[0.9])
    d2 = predict(d, collect(51.0:55.0))
    @test d2.trend.sig == [0.9]
end

@testset "t_out kwarg" begin
    t_out = collect(1.0:2.0:49.0)
    d = fit(Decomposition, _y, _t; period=12.0, t_out=t_out)
    @test length(d.trend.mean) == length(t_out)
end

@testset "params NamedTuple has required keys" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    @test haskey(d.params, :σ_trend)
    @test haskey(d.params, :l_trend)
    @test haskey(d.params, :σ_seasonal)
    @test haskey(d.params, :l_seasonal)
    @test haskey(d.params, :period)
    @test haskey(d.params, :bandwidth)
    @test haskey(d.params, :σ_noise)
    @test d.params.period ≈ 12.0
end

@testset "quasi-periodic bandwidth" begin
    d = fit(Decomposition, _y, _t; period=12.0, bandwidth=30.0)
    @test isfinite(d.params.bandwidth)
    @test d.params.bandwidth ≈ 30.0
end

@testset "show method does not error" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    io = IOBuffer()
    @test (show(io, d); true)
end

@testset "show method no seasonal" begin
    d = fit(Decomposition, _y, _t)
    io = IOBuffer()
    @test (show(io, d); true)
end
