# Stage smoothers of a filter model M from its keywords: the stages (e.g. `trend_final`) and
# the components (e.g. `trend`, every trend stage), over the defaults
function _stage_settings(::Type{M}, defaults::NamedTuple, kwargs::NamedTuple) where {M}
    stages = keys(defaults)
    component(stage) = Symbol(first(split(string(stage), '_')))
    components = unique(component.(stages))
    _check_keys(kwargs, (stages..., components...), "$(nameof(M)) keyword")
    settings = map(stages) do stage
        c = component(stage)
        haskey(kwargs, c) && haskey(kwargs, stage) &&
            throw(ArgumentError("give $c or $stage, not both"))
        return haskey(kwargs, stage) ? kwargs[stage] :
               haskey(kwargs, c) ? kwargs[c] : defaults[stage]
    end
    return NamedTuple{stages}(settings)
end

function _check_stages(::Type{M}, settings::NamedTuple) where {M}
    for (stage, v) in pairs(settings)
        T = fieldtype(M, stage)
        v isa T || throw(ArgumentError("$stage must be of type $T, got $(typeof(v))"))
    end
    return settings
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
_hyperparameters(m::SeasonalLOESS, _) = (; m.bandwidth)

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

_Report() = _Report(Pair{Symbol, Any}[], Symbol[], Symbol[], Pair{Symbol, Any}[])

# Record the fitted smoother of a stage (e.g. :trend_final)
function _record!(r::_Report, f::FittedSmoother, stage::Symbol; derived = (), chosen = ())
    suffix = Symbol(:_, stage)
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
