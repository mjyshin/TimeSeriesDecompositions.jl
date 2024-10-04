# Preprocessing defaults
const logscale::Bool = false
const norm::Bool = true
const ȳ::Real = 0
const s_y::Real = 1
const slope::Bool = false

# Decomposition defaults
const δ::Real = 1e-12
const filter_outliers::Bool = false
const β_outliers::Real = 0.95
const dynamic::Bool = false
const n_dynamic::Int=1

# Rate computation defaults
const rate::Bool = false
const fast_rate::Bool = false

# Output defaults
const β_ci::Vector{<:Real} = [0.5, 0.9]
const save::Bool = false
