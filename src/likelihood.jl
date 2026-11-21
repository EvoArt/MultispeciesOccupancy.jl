# The multivariate Bernoulli occupancy model of Rota et al. (2016) and its
# likelihoods. At site i the latent presence vector z ∈ {0,1}^S has
#
#     log P(z) = Σ_s f_s z_s + Σ_{s<t} f_st z_s z_t - log Σ_{z'} exp(…)
#
# with natural parameters f_s = X[i,:]'B[:,s] (first order) and
# f_st = W[i,:]'G[:,pair] (species interactions). Detections are independent
# Bernoulli(p_ijs) draws where a species is present. With S species the
# normalising sum has 2^S terms, so these exact computations suit S ≲ 10.

"""
    MultivariateBernoulli(f1, f2, pairs)

The distribution of a presence vector `z ∈ {0,1}^S` with first-order natural
parameters `f1` (one per species) and second-order (interaction) parameters `f2`
(one per pair in `pairs`, a vector of `(s, t)` tuples):

    log P(z) = Σₛ f1[s] zₛ + Σₖ f2[k] z[pairs[k][1]] z[pairs[k][2]] - log normaliser

`f2[k] > 0` means the two species occur together more often than their first-order
terms alone imply. With no interactions the species are independent with
occupancy probabilities `logistic.(f1)`.
"""
struct MultivariateBernoulli{T<:Real} <: DiscreteMultivariateDistribution
    f1::Vector{T}
    f2::Vector{T}
    pairs::Vector{Tuple{Int,Int}}
    states::BitMatrix
end
function MultivariateBernoulli(f1::AbstractVector, f2::AbstractVector, pairs=all_pairs(length(f1)))
    T = promote_type(eltype(f1), eltype(f2))
    length(f2) == length(pairs) || throw(DimensionMismatch("$(length(f2)) interaction parameters for $(length(pairs)) pairs"))
    return MultivariateBernoulli(f1, f2, pairs, occupancy_states(length(f1)))
end
# with the state table precomputed (it depends only on the number of species)
function MultivariateBernoulli(f1::AbstractVector, f2::AbstractVector, pairs, states::BitMatrix)
    T = promote_type(eltype(f1), eltype(f2))
    length(f2) == length(pairs) || throw(DimensionMismatch("$(length(f2)) interaction parameters for $(length(pairs)) pairs"))
    size(states) == (2^length(f1), length(f1)) || throw(DimensionMismatch("state table does not match $(length(f1)) species"))
    return MultivariateBernoulli{T}(collect(T, f1), collect(T, f2), collect(pairs), states)
end

Base.length(d::MultivariateBernoulli) = length(d.f1)
Base.eltype(::Type{<:MultivariateBernoulli}) = Bool

"All `(s, t)` species pairs with `s < t`, in the order used for interaction terms."
all_pairs(S::Integer) = [(s, t) for s in 1:S for t in s+1:S]

"All `2^S` presence vectors as the rows of a `BitMatrix`."
function occupancy_states(S::Integer)
    S <= 16 || throw(ArgumentError("$S species give 2^$S joint states; the exact likelihood is limited to 16"))
    return BitMatrix([((k - 1) >> (s - 1)) & 1 == 1 for k in 1:2^S, s in 1:S])
end

# Unnormalised log-probability of row k of `states`
@inline function _state_logit(f1, f2, pairs, states, k)
    θ = zero(promote_type(eltype(f1), eltype(f2)))
    for s in eachindex(f1)
        states[k, s] && (θ += f1[s])
    end
    for (q, (s, t)) in enumerate(pairs)
        (states[k, s] && states[k, t]) && (θ += f2[q])
    end
    return θ
end

_state_logits(d::MultivariateBernoulli) = [_state_logit(d.f1, d.f2, d.pairs, d.states, k) for k in axes(d.states, 1)]

function Distributions._logpdf(d::MultivariateBernoulli, z::AbstractVector)
    θ = _state_logits(d)
    k = 1 + sum(Int(Bool(z[s])) << (s - 1) for s in eachindex(z))
    return θ[k] - logsumexp(θ)
end

function Distributions._rand!(rng::AbstractRNG, d::MultivariateBernoulli, z::AbstractVector)
    θ = _state_logits(d)
    z .= view(d.states, _draw_state(rng, θ), :)
    return z
end

"Probability of each joint state (the rows of `occupancy_states`)."
state_probabilities(d::MultivariateBernoulli) = softmax(_state_logits(d))

function _draw_state(rng, lp)
    w = exp.(lp .- maximum(lp))
    u = rand(rng) * sum(w)
    acc = zero(eltype(w))
    for (k, wk) in enumerate(w)
        acc += wk
        u <= acc && return k
    end
    return length(w)
end

# ----------------------------------------------------------- likelihoods

# Natural parameters and detection log-odds from coefficient vectors
function _unpack(d, b, g, a)
    S, P = length(d.species), length(d.pairs)
    B = reshape(b, size(d.X, 2), S)
    G = reshape(g, size(d.W, 2), P)
    A = reshape(a, size(d.V, 3), S)
    return d.X * B, d.W * G, A
end

# log P(y_i,s | z_s = 1): Bernoulli detections at the site's visits
function _detection_loglik(d, A, i, s)
    T = eltype(A)
    ℓ = zero(T)
    for j in axes(d.y, 2)
        yij = d.y[i, j, s]
        ismissing(yij) && continue
        η = zero(T)
        for c in axes(d.V, 3)
            η += d.V[i, j, c] * A[c, s]
        end
        ℓ += yij > 0 ? -log1pexp(-η) : -log1pexp(η)
    end
    return ℓ
end

"""
    marginal_loglik(data, b, g, a)

The log-likelihood of the detection data with the latent occupancy states summed
out (Rota et al. 2016, eq. 4), for coefficient vectors `b` (first-order occupancy,
`p_occ × S`, by species), `g` (interactions, `p_int × P`, by pair) and `a`
(detection, `p_det × S`, by species). Differentiable in all three; use it in any
PracticalBayes model with `@addlogprob!`, or through [`rota_model`](@ref).
"""
function marginal_loglik(d, b, g, a)
    F1, F2, A = _unpack(d, b, g, a)
    T = promote_type(eltype(F1), eltype(F2), eltype(A))
    S, K = length(d.species), size(d.states, 1)
    θ = Vector{T}(undef, K)
    w = Vector{T}(undef, K)
    ℓ1 = Vector{T}(undef, S)
    total = zero(T)
    for i in axes(d.y, 1)
        for s in 1:S
            ℓ1[s] = _detection_loglik(d, A, i, s)
        end
        f1, f2 = view(F1, i, :), view(F2, i, :)
        for k in 1:K
            θ[k] = _state_logit(f1, f2, d.pairs, d.states, k)
            obs = zero(T)
            for s in 1:S
                if d.states[k, s]
                    obs += ℓ1[s]
                elseif d.detected[i, s]
                    obs = T(-Inf)
                    break
                end
            end
            w[k] = θ[k] + obs
        end
        total += logsumexp(w) - logsumexp(θ)
    end
    return total
end

"""
    occupancy_loglik(data, b, g, z)

The complete-data log-likelihood of latent states `z` (one presence vector per
site) under the occupancy part of the model: `Σᵢ log P(zᵢ | b, g)`. With `z` given,
this is all that `b` and `g` depend on, so it is the target of their Gibbs block.
"""
function occupancy_loglik(d, b, g, z)
    S, P = length(d.species), length(d.pairs)
    F1 = d.X * reshape(b, size(d.X, 2), S)
    F2 = d.W * reshape(g, size(d.W, 2), P)
    T = promote_type(eltype(F1), eltype(F2))
    θ = Vector{T}(undef, size(d.states, 1))
    total = zero(T)
    for i in axes(d.y, 1)
        f1, f2 = view(F1, i, :), view(F2, i, :)
        for k in axes(d.states, 1)
            θ[k] = _state_logit(f1, f2, d.pairs, d.states, k)
        end
        kz = 1 + sum(Int(Bool(z[i][s])) << (s - 1) for s in 1:S)
        total += θ[kz] - logsumexp(θ)
    end
    return total
end

"""
    detection_loglik(data, a, z)

The log-likelihood of the detections given latent states `z`: Bernoulli detections
at the visits to sites where each species is present. Zero likelihood (`-Inf`) if a
species was detected where `z` says it is absent.
"""
function detection_loglik(d, a, z)
    A = reshape(a, size(d.V, 3), length(d.species))
    total = zero(eltype(A))
    for i in axes(d.y, 1), s in eachindex(d.species)
        if z[i][s]
            total += _detection_loglik(d, A, i, s)
        elseif d.detected[i, s]
            return oftype(total, -Inf)
        end
    end
    return total
end
