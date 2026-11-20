# Resolve the `smoother` keyword against the defaults, NamedTuples of components of stages
function _resolve_smoothers(smoother::NamedTuple, defaults::NamedTuple)
    _check_keys(smoother, keys(defaults), "smoother")
    resolved = map(keys(defaults)) do c
        return _resolve_stages(get(smoother, c, nothing), defaults[c], c)
    end
    return NamedTuple{keys(defaults)}(resolved)
end

_resolve_stages(::Nothing, default::NamedTuple, ::Symbol) = default

function _resolve_stages(s::AbstractSmoother, default::NamedTuple, c::Symbol)
    _check_stage(s, default, c, first(keys(default)))
    return map(_ -> s, default)
end

function _resolve_stages(s::NamedTuple, default::NamedTuple, c::Symbol)
    _check_keys(s, keys(default), "smoother.$c")
    for (stage, v) in pairs(s)
        _check_stage(v, default, c, stage)
    end
    return merge(default, s)
end

function _resolve_stages(s, _, c::Symbol)
    throw(ArgumentError("smoother.$c must be a smoother or a NamedTuple, got $(typeof(s))"))
end

_smoother_type(c::Symbol) = c == :trend ? AbstractTrendSmoother : AbstractSeasonalSmoother

function _check_stage(v, default::NamedTuple, c::Symbol, stage::Symbol)
    isnothing(v) && isnothing(default[stage]) && return nothing
    v isa _smoother_type(c) || throw(
        ArgumentError(
        "smoother.$c.$stage must be an $(_smoother_type(c)), got $(typeof(v))"
    ),
    )
    return nothing
end

# Numeric hyperparameters of a fitted smoother
function _hyperparameters(m::LocalPolynomial, _)
    (; bandwidth = m.bandwidth, _ratio(m.endpoints)...)
end
_hyperparameters(m::MovingAverage, _) = (; bandwidth = m.bandwidth, _ratio(m.endpoints)...)

function _hyperparameters(m::LOESS, N)
    span = isnothing(m.span) ? (;) :
           (; span = m.span, neighbors = _span_neighbors(m.span, N))
    bandwidth = isnothing(m.bandwidth) ? (;) : (; m.bandwidth)
    return (; span..., bandwidth...)
end

_hyperparameters(m::SeasonalMovingAverage, _) = (; m.n, m.m)
_hyperparameters(::StableSeasonal, _) = (;)

_ratio(e::MinimumRevision) = (; e.ratio)
_ratio(_) = (;)

const _CHOSEN_HYPERPARAMETERS = (:bandwidth, :span, :ratio)

# Hyperparameters keyed <quantity>_<component>_<stage> in fitting order, those estimated
# and derived, and the criteria and notes
struct _Report
    θ::Vector{Pair{Symbol, Any}}
    estimated::Vector{Symbol}
    derived::Vector{Symbol}
    criteria::Vector{Pair{Symbol, Any}}
end

function _Report(period::Real)
    return _Report(
        Pair{Symbol, Any}[:period => Float64(period)], Symbol[], Symbol[], Pair{
            Symbol, Any}[]
    )
end

# Record the fitted smoother of a stage (suffix e.g. :_trend_final)
function _record!(r::_Report, f::FittedSmoother, suffix::Symbol; derived = (), chosen = ())
    data = [k for k in keys(f.estimated) if k ∈ _CHOSEN_HYPERPARAMETERS]
    :span ∈ data && push!(data, :neighbors)
    for (k, v) in pairs(_hyperparameters(f.model, length(f.t)))
        key = Symbol(k, suffix)
        push!(r.θ, key => v)
        (k ∈ data || k ∈ chosen) && push!(r.estimated, key)
        k ∈ derived && push!(r.derived, key)
    end
    for (k, v) in pairs(f.estimated)
        k ∈ _CHOSEN_HYPERPARAMETERS || push!(r.criteria, Symbol(k, suffix) => v)
    end
    return r
end

function _finish(r::_Report, name, Δt, grid)
    θ = (; r.θ...)
    criteria = (; r.criteria...)
    chosen = (; (k => θ[k] for k in keys(θ) if k ∈ r.estimated)...)
    _log_estimated(name, (; chosen..., criteria...))
    return θ, (; Δt, grid, estimated = r.estimated, derived = r.derived, criteria)
end
