######################################
###                                  #
##    TREND FILTERING ALGORITHMS    ##
#                                #####
######################################

#== Simple moving average (SMA) filter ==#

"""
    sma(y::Vector{<:Real}, bw::Int; kwargs...)

Apply the simple moving average (SMA) filter on the time series (yₜ) using a uniform kernel of bandwidth bw.
"""
function sma(y::Vector{<:Real}, bw::Int; kwargs...)
    @assert bw > 0
    # No filtering case
    if bw == 1 return y end

    w = isodd(bw) ? ones(bw) : [0.5; ones(bw-1); 0.5]
    h = Int((length(w) - 1)//2)
    ȳ = fill(NaN, length(y))
    y = [fill(NaN, h); y; fill(NaN, h)]

    for (t, n) ∈ enumerate(h+1:length(y)-h)
        y_t = y[n-h:n+h]
        keep = .!isnan.(y_t)
        ȳ[t] = mean(y_t[keep], Weights(w[keep]))
    end

    return ȳ
end


#== Henderson filter ==#

compute_X(h::Int, d::Int) = hcat([(-h:h).^n for n ∈ 0:d]...)    # Eq. (1.2)
compute_K(h::Int) = Diagonal(15/16*(1 .- (-h/(h+1):1/(h+1):h/(h+1)).^2).^2 ./ (h+1))    # Eq. (32) in Dagum & Bianconcini
compute_w(X::Matrix{<:Int}, K::Diagonal{<:Real}) = (K*X/(X'*K*X))[:, 1]    # Eq. (3)

"""
    henderson(y::Vector{<:Real}, h::Int; d::Int=3, kwargs...)

Apply the symmetric Henderson filter on the time series (yₜ) for t ∈ [h+1, N-h] using a kernel of bandwidth 2h+1 and degree d.
"""
function henderson(y::Vector{<:Real}, h::Int; d::Int=3, kwargs...)
    @assert h ≥ 0
    # No filtering case
    if h == 0 return y end

    # Symmetric weights
    if :w ∈ keys(kwargs)
        w = kwargs[:w]
    else
        X = :X ∈ keys(kwargs) ? kwargs[:X] : compute_X(h, d)
        K = :K ∈ keys(kwargs) ? kwargs[:K] : compute_K(h)
        w = compute_w(X, K)
    end

    # Initialize regression vector
    N = length(y)
    h = Int((length(w) - 1)//2)
    x̄ = fill(NaN, N)

    # Points from h+1 to n-h
    x̄[h+1:end-h] = [w'*y[t-h:t+h] for t ∈ h+1:N-h]    # Eq. (2.2) x̄ₜ := m̂ₜ = wᵀy

    return x̄
end


#== Direct asymmetric filter (DAF) ==#

function daf_t(idxp::UnitRange{<:Int}, yp::Vector{<:Real}, X::Matrix{<:Int}, K::Diagonal{<:Real})
    Xp = X[idxp, :]    # first h+1+q rows of X
    Kp = Diagonal(K[idxp, idxp])    # first h+1+q square of K
    wa = compute_w(Xp, Kp)    # DAF weights
    x̄ = wa'*yp

    return x̄
end

"""
    daf(y::Vector{<:Real}, h::Int; d::Int=3, kwargs...)

Apply the direct asymmetric filter (DAF) on the time series (yₜ) for t ∈ [1,h] ∪ [N-h+1,N] using a kernel of bandwidth 2h+1 and degree d. Apply the Henderson filter for the rest.
"""
function daf(y::Vector{<:Real}, h::Int; d::Int=3, kwargs...)
    @assert h ≥ 0
    # No filtering case
    if h == 0 return y end

    # Symmetric weights
    X = :X ∈ keys(kwargs) ? kwargs[:X] : compute_X(h, d)
    K = :K ∈ keys(kwargs) ? kwargs[:K] : compute_K(h)
    w = :w ∈ keys(kwargs) ? kwargs[:w] : compute_w(X, K)

    # Points from h+1 to n-h
    x̄ = henderson(y, h, d=d, w=w)
    N = length(y)

    # First h points
    for t ∈ 1:h
        q = t-1    # q ∈ [0,h-1]
        idxf = h+1-q:2h+1    # present and future indices
        yf = y[1:t+h]    # all available data

        x̄[t] = daf_t(idxf, yf, X, K)
    end

    # Last h points
    for t ∈ N-h+1:N
        q = N-t    # q ∈ [0,h-1]
        idxp = 1:h+1+q    # past and present indices
        yp = y[t-h:end]    # all available data

        x̄[t] = daf_t(idxp, yp, X, K)
    end

    return x̄
end


#== Minimum revision filter (MRF) ==#

compute_Q(Zp::Matrix{<:Int}, Dp::Diagonal{<:Real}, δ::Real) = Dp + Zp*δ*δ'*Zp'    # Eq. (8.1)
compute_M(Up::Matrix{<:Int}, Q::Matrix{<:Real}) = Q\(I - Up*((Up'*(Q\Up))\(Up'/Q)))    # Eq. (9.1)
compute_L(Up::Matrix{<:Int}, Q::Matrix{<:Real}) = Q\Up/(Up'*(Q\Up))    # Eq. (9.2)
compute_v(wp::Vector{<:Real}, wf::Vector{<:Real}, Uf::Matrix{<:Int}, Zp::Matrix{<:Int}, Zf::Matrix{<:Int}, M::Matrix{<:Real}, L::Matrix{<:Real}, δ::Real) = wp + (L*Uf' + M*Zp*δ*δ'*Zf')*wf    # Eq. (9)
function mrf_t(idxp::UnitRange{<:Int}, yp::Vector{<:Real}, h::Int, U::Matrix{<:Int}, Z::Matrix{<:Int}, w::Vector{<:Real}, D::Diagonal{<:Real}, δ::Real)
    Up = U[idxp, :]    # first/last h+1+q rows of U
    Zp = Z[idxp, :]    # first/last h+1+q rows of Z
    wp = w[idxp]    # first/last h+1+q elements of w
    Dp = Diagonal(D[idxp, idxp])    # first/last h+1+q square of D

    idxf = setdiff(1:2h+1, idxp)    # other indices
    Uf = U[idxf, :]    # rest of U
    Zf = Z[idxf, :]    # rest of Z
    wf = w[idxf]    # rest of w

    Q = compute_Q(Zp, Dp, δ)
    M = compute_M(Up, Q)
    L = compute_L(Up, Q)
    v = compute_v(wp, wf, Uf, Zp, Zf, M, L, δ)
    x̄ = v'*yp

    return x̄
end

"""
    mrf(y::Vector{<:Real}, h::Int; d::Int=3, dU::Int=d-1, δ::Real=0, σ::Real=1, kwargs...)

Apply the minimum revision filter (MRF) on the time series (yₜ) for t ∈ [1,h] ∪ [N-h+1,N] using a kernel of bandwidth 2h+1 and degree d. Apply the Henderson filter for the rest.
"""
function mrf(y::Vector{<:Real}, h::Int; d::Int=3, dU::Int=d-1, δ::Real=0, σ::Real=1, kwargs...)
    @assert h ≥ 0
    # No filtering case
    if h == 0 return y end

    # Symmetric weights
    X = :X ∈ keys(kwargs) ? kwargs[:X] : compute_X(h, d)
    K = :K ∈ keys(kwargs) ? kwargs[:K] : compute_K(h)
    w = :w ∈ keys(kwargs) ? kwargs[:w] : compute_w(X, K)
    D = σ^2*I(2h+1)

    # Data partitions in margins
    idxU = 1:dU
    idxZ = dU+1:dU+1
    U = X[:, idxU]
    Z = X[:, idxZ]

    # Points from h+1 to n-h
    x̄ = henderson(y, h, d=d, w=w)
    N = length(y)

    # First h points
    for t ∈ 1:h
        q = t-1    # q ∈ [0,h-1]
        idxf = h+1-q:2h+1    # present and future indices
        yf = y[1:t+h]    # all available data

        x̄[t] = mrf_t(idxf, yf, h, U, Z, w, D, δ)
    end

    # Last h points
    for t ∈ N-h+1:N
        q = N-t    # q ∈ [0,h-1]
        idxp = 1:h+1+q    # past and present indices
        yp = y[t-h:end]    # all available data

        x̄[t] = mrf_t(idxp, yp, h, U, Z, w, D, δ)
    end

    return x̄
end

function loess(y::Vector{<:Real}, h::Real; kwargs...)
    @assert h ≥ 0
    # No filtering case
    if h == 0 return y end

    N = length(y)
    n = 2h + 1
    span = n/N

    x̄ = predict(Loess.loess(1:N, y, span=span), 1:N)

    return x̄
end
