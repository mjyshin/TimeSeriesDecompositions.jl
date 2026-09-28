# Stages of a filter-based decomposition: the `smoother` keyword, and the record of the
# hyperparameters and criteria of the fitted smoothers.

# Resolve the `smoother` keyword against the defaults, a NamedTuple of components (trend,
# seasonal), each a NamedTuple of stages. A component may be given as one smoother, which
# applies to every stage of it, or as a NamedTuple of some of its stages; the others keep
# the default. `nothing` (where the default is `nothing`) leaves a stage to be chosen.
function _resolve_smoothers(smoother::NamedTuple, defaults::NamedTuple)
    _check_smoother_keys(smoother, keys(defaults), "smoother")
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
    _check_smoother_keys(s, keys(default), "smoother.$c")
    for (stage, v) in pairs(s)
        _check_stage(v, default, c, stage)
    end
    return merge(default, s)
end

function _resolve_stages(s, _, c::Symbol)
    throw(ArgumentError("smoother.$c must be a smoother or a NamedTuple, got $(typeof(s))"))
end

function _check_smoother_keys(s::NamedTuple, valid, name)
    for key in keys(s)
        key ∈ valid || throw(ArgumentError("Unknown $name key :$key. Valid keys: $valid"))
    end
    return nothing
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

# Numeric hyperparameters of a fitted smoother: its smoothing width, minimum-revision
# ratio and seasonal filter lengths. Structural choices (degree, kernel, end-weight
# scheme) are left to the smoother itself.
_hyperparameters(m::LocalPolynomial, _) = (; bandwidth=m.bandwidth, _ratio(m.endpoints)...)
_hyperparameters(m::MovingAverage, _) = (; bandwidth=m.bandwidth, _ratio(m.endpoints)...)

function _hyperparameters(m::LOESS, N)
    span = isnothing(m.span) ? (;) : (; span=m.span, neighbors=_span_neighbors(m.span, N))
    bandwidth = isnothing(m.bandwidth) ? (;) : (; m.bandwidth)
    return (; span..., bandwidth...)
end

_hyperparameters(m::SeasonalMovingAverage, _) = (; m.n, m.m)
_hyperparameters(::StableSeasonal, _) = (;)

_ratio(e::MinimumRevision) = (; e.ratio)
_ratio(_) = (;)

# Keys of a smoother's `estimated` that are hyperparameters; the others are criteria
const _CHOSEN_HYPERPARAMETERS = (:bandwidth, :span, :ratio)

# Hyperparameters θ, keyed <quantity>_<component>_<stage> in the order they were set,
# with those chosen from the data (estimated) and those derived from the period or the
# length of the series (derived); the rest were given. Criteria are the selection
# criteria and notes.
struct _Report
    θ::Vector{Pair{Symbol,Any}}
    estimated::Vector{Symbol}
    derived::Vector{Symbol}
    criteria::Vector{Pair{Symbol,Any}}
end

function _Report(period::Real)
    return _Report(
        Pair{Symbol,Any}[:period => Float64(period)], Symbol[], Symbol[], Pair{Symbol,Any}[]
    )
end

# Record the fitted smoother f of a stage (suffix e.g. :_trend_final): `derived` and
# `chosen` name the hyperparameters derived by rule or chosen from the data by the
# decomposition; those the smoother chose itself are in f.estimated
function _record!(r::_Report, f::FittedSmoother, suffix::Symbol; derived=(), chosen=())
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

# θ and the fit information; logs what was chosen from the data
function _finish(r::_Report, name, Δt, grid)
    θ = (; r.θ...)
    criteria = (; r.criteria...)
    chosen = (; (k => θ[k] for k in keys(θ) if k ∈ r.estimated)...)
    _log_estimated(name, (; chosen..., criteria...))
    return θ, (; Δt, grid, estimated=r.estimated, derived=r.derived, criteria)
end
