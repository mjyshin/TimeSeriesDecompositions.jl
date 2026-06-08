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
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
    @test d.θ.bandwidth  == Inf
end

@testset "no seasonal component (explicit Inf)" begin
    d = fit(Decomposition, _y, _t; period=Inf)
    @test isnothing(d.seasonal)
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
end

@testset "bandwidth ignored when no seasonal" begin
    d = fit(Decomposition, _y, _t; bandwidth=30.0)
    @test isnothing(d.seasonal)
    @test d.θ.bandwidth == Inf
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
    @test d.sig == [0.68, 0.95]
    @test d.trend.sig == [0.68, 0.95]
    @test length(d.trend.lower) == 2
end

@testset "sig deduplication and sorting" begin
    d = fit(Decomposition, _y, _t; period=12.0, sig=[0.95, 0.5, 0.5])
    @test d.sig == [0.5, 0.95]
    @test length(d.trend.lower) == 2
end

@testset "predict with no t_new uses decomp.t" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    d2 = predict(d)
    @test d2.t == d.t
    @test length(d2.trend.mean) == length(_t)
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
    @test d2.sig == [0.9]
    @test d2.trend.sig == [0.9]
end

@testset "t_out kwarg" begin
    t_out = collect(1.0:2.0:49.0)
    d = fit(Decomposition, _y, _t; period=12.0, t_out=t_out)
    @test length(d.trend.mean) == length(t_out)
end

@testset "θ NamedTuple has required keys" begin
    d = fit(Decomposition, _y, _t; period=12.0)
    @test haskey(d.θ, :σ_trend)
    @test haskey(d.θ, :l_trend)
    @test haskey(d.θ, :σ_seasonal)
    @test haskey(d.θ, :l_seasonal)
    @test haskey(d.θ, :period)
    @test haskey(d.θ, :bandwidth)
    @test haskey(d.θ, :σ_noise)
    @test d.θ.period ≈ 12.0
end

@testset "quasi-periodic bandwidth" begin
    d = fit(Decomposition, _y, _t; period=12.0, bandwidth=30.0)
    @test isfinite(d.θ.bandwidth)
    @test d.θ.bandwidth ≈ 30.0
end



@testset "σ_seasonal=0 disables seasonal" begin
    d = fit(Decomposition, _y, _t; period=12.0, θ_init=(σ_seasonal=0.0,))
    @test isnothing(d.seasonal)
    @test d.θ.period     == Inf
    @test d.θ.σ_seasonal == 0.0
    @test d.θ.l_seasonal == Inf
    @test d.θ.bandwidth  == Inf
end

@testset "σ_noise=0 disables noise" begin
    d = fit(Decomposition, _y, _t; period=12.0, θ_init=(σ_noise=0.0,))
    @test isnothing(d.noise)
    @test d.θ.σ_noise == 0.0
    @test !isnothing(d.seasonal)
end

@testset "custom kernels" begin
    d = fit(Decomposition, _y, _t; period=12.0,
        kernel=(trend=Matern52Kernel(), seasonal=Matern32Kernel()))
    @test d isa Decomposition
    @test d.trend isa Component
    @test !isnothing(d.seasonal)
end

@testset "θ from fit can be passed as θ_init" begin
    d1 = fit(Decomposition, _y, _t; period=12.0, bandwidth=30.0)
    d2 = fit(Decomposition, _y, _t; θ_init=d1.θ)
    @test d2.θ.period    ≈ 12.0
    @test d2.θ.bandwidth ≈ 30.0
    @test !isnothing(d2.seasonal)
end

@testset "explicit kwarg overrides θ_init period" begin
    d1 = fit(Decomposition, _y, _t; period=12.0)
    d2 = fit(Decomposition, _y, _t; period=Inf, θ_init=d1.θ)
    @test isnothing(d2.seasonal)
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
