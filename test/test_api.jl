using GPEstimators
using Test

const _t = collect(1.0:50.0)
const _y = sin.(2π .* _t ./ 12) .+ 0.01 .* _t .+ 0.1 .* randn(50)

@testset "decompose returns Decomposition" begin
    d = decompose(_y, _t; period=12.0)
    @test d isa Decomposition
end

@testset "ComponentEstimate fields" begin
    d = decompose(_y, _t; period=12.0)
    for comp in (d.trend, d.seasonal, d.noise)
        @test comp isa ComponentEstimate
        @test length(comp.mean) == 50
        @test length(comp.std)  == 50
        @test length(comp.lower) == length(comp.sig)
        @test length(comp.upper) == length(comp.sig)
        @test all(comp.std .≥ 0)
    end
    @test isnothing(d.rate)
end

@testset "rate component" begin
    d = decompose(_y, _t; period=12.0, rate=true)
    @test d.rate isa ComponentEstimate
    @test length(d.rate.mean) == 50
end

@testset "sig levels" begin
    d = decompose(_y, _t; period=12.0, sig=[0.68, 0.95])
    @test d.trend.sig == [0.68, 0.95]
    @test length(d.trend.lower) == 2
end

@testset "predict at new times" begin
    d = decompose(_y, _t; period=12.0)
    t_new = collect(51.0:60.0)
    d2 = predict(d, t_new)
    @test d2 isa Decomposition
    @test length(d2.t) == 10
    @test length(d2.trend.mean) == 10
end

@testset "predict preserves sig" begin
    d = decompose(_y, _t; period=12.0, sig=[0.9])
    d2 = predict(d, collect(51.0:55.0))
    @test d2.trend.sig == [0.9]
end

@testset "t_out kwarg" begin
    t_out = collect(1.0:2.0:49.0)
    d = decompose(_y, _t; period=12.0, t_out=t_out)
    @test length(d.trend.mean) == length(t_out)
end

@testset "params NamedTuple has required keys" begin
    d = decompose(_y, _t; period=12.0)
    @test haskey(d.params, :σ_trend)
    @test haskey(d.params, :l_trend)
    @test haskey(d.params, :σ_seasonal)
    @test haskey(d.params, :l_seasonal)
    @test haskey(d.params, :period)
    @test haskey(d.params, :bandwidth)
    @test haskey(d.params, :σ_noise)
    @test d.params.period > 0
end

@testset "quasi-periodic bandwidth" begin
    d = decompose(_y, _t; period=12.0, bandwidth=30.0)
    @test isfinite(d.params.bandwidth)
    @test d.params.bandwidth ≈ 30.0
end

@testset "logscale decomposition" begin
    y_pos = exp.(_y .- minimum(_y) .+ 0.1)
    d = decompose(y_pos, _t; period=12.0, logscale=true)
    @test d isa Decomposition
    @test all(d.trend.mean .> 0)
end

@testset "show method does not error" begin
    d = decompose(_y, _t; period=12.0)
    io = IOBuffer()
    @test (show(io, d); true)
end
