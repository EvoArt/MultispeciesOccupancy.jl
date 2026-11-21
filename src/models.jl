# The model as PracticalBayes models: marginal (latent states summed out, for
# NUTS) and latent (states as parameters, for Gibbs with exact kernels).

_sizes(d) = (size(d.X, 2) * nspecies(d), size(d.W, 2) * length(d.pairs), size(d.V, 3) * nspecies(d))

_priors(prior_sd::Real) = (occupancy=prior_sd, interaction=prior_sd, detection=prior_sd)
_priors(p::NamedTuple) = merge((occupancy=2.0, interaction=2.0, detection=2.0), p)

@model function _rota_marginal(d, nb, ng, na, so, si, sp)
    b ~ filldist(Normal(0, so), nb)
    if ng > 0
        g ~ filldist(Normal(0, si), ng)
    else
        g = zeros(0)
    end
    a ~ filldist(Normal(0, sp), na)
    @addlogprob! marginal_loglik(d, b, g, a)
end

"""
    rota_model(data::OccupancyData; prior_sd=2)

The multispecies occupancy model with the latent states summed out, as a
PracticalBayes model with parameters `b` (first-order occupancy), `g`
(interactions) and `a` (detection), each with independent `Normal(0, prior_sd)`
priors (`prior_sd` may be a `NamedTuple` with `occupancy`, `interaction` and
`detection` entries). Sample it with NUTS, or use [`marginal_loglik`](@ref) in a
model of your own.

```julia
chn = sample(rota_model(data), NUTS(), 1000)
```
"""
function rota_model(d::OccupancyData; prior_sd=2.0)
    p = _priors(prior_sd)
    nb, ng, na = _sizes(d)
    return _rota_marginal(d, nb, ng, na, p.occupancy, p.interaction, p.detection)
end

@model function _rota_latent(d, nb, ng, na, so, si, sp)
    b ~ filldist(Normal(0, so), nb)
    if ng > 0
        g ~ filldist(Normal(0, si), ng)
    else
        g = zeros(0)
    end
    a ~ filldist(Normal(0, sp), na)
    S, P = nspecies(d), length(d.pairs)
    F1 = d.X * reshape(b, size(d.X, 2), S)
    F2 = ng > 0 ? d.W * reshape(g, size(d.W, 2), P) : zeros(eltype(F1), nsites(d), P)
    z = Vector{Vector{Bool}}(undef, nsites(d))
    for i in 1:nsites(d)
        z[i] ~ MultivariateBernoulli(F1[i, :], F2[i, :], d.pairs, d.states)
    end
    @addlogprob! detection_loglik(d, a, z)
end

"""
    LatentStates(data::OccupancyData)

A Gibbs kernel for the latent presence vectors `z` (one per site): each site's
`z` is drawn exactly from its full conditional, over all `2^S` joint states,
given the coefficients and the detections.
"""
struct LatentStates <: AbstractGibbsKernel
    data::OccupancyData
end

function PracticalBayes.gibbs_step(rng, k::LatentStates, names, c::ModelConditional)
    names == (:z,) || error("LatentStates owns block :z, got $names")
    d = k.data
    F1, F2, A = _unpack(d, c.values.b, _g(c, d), c.values.a)
    S, K = nspecies(d), size(d.states, 1)
    w = zeros(K)
    ℓ1 = zeros(S)
    z = map(1:nsites(d)) do i
        for s in 1:S
            ℓ1[s] = _detection_loglik(d, A, i, s)
        end
        f1, f2 = view(F1, i, :), view(F2, i, :)
        for kk in 1:K
            w[kk] = _state_logit(f1, f2, d.pairs, d.states, kk)
            for s in 1:S
                if d.states[kk, s]
                    w[kk] += ℓ1[s]
                elseif d.detected[i, s]
                    w[kk] = -Inf
                    break
                end
            end
        end
        Vector{Bool}(d.states[_draw_state(rng, w), :])
    end
    return (; z)
end

_g(c, d) = hasproperty(c.values, :g) ? c.values.g : zeros(size(d.W, 2) * length(d.pairs))

"""
    DetectionPG(data::OccupancyData; prior_sd=2)

A Gibbs kernel for the detection coefficients `a`: given the latent states, each
species' detections at the sites where it is present form a logistic regression,
drawn exactly by Pólya–Gamma augmentation.
"""
struct DetectionPG <: AbstractGibbsKernel
    data::OccupancyData
    prior_sd::Float64
    rows::Vector{Vector{Tuple{Int,Int}}}     # surveyed (site, visit) per species
end
function DetectionPG(d::OccupancyData; prior_sd=2.0)
    rows = [[(i, j) for j in 1:nvisits(d) for i in 1:nsites(d) if !ismissing(d.y[i, j, s])] for s in 1:nspecies(d)]
    return DetectionPG(d, Float64(prior_sd), rows)
end

function PracticalBayes.gibbs_step(rng, k::DetectionPG, names, c::ModelConditional)
    names == (:a,) || error("DetectionPG owns block :a, got $names")
    d, z = k.data, c.values.z
    p = size(d.V, 3)
    A = reshape(collect(Float64, c.values.a), p, nspecies(d))
    for s in 1:nspecies(d)
        r = [(i, j) for (i, j) in k.rows[s] if z[i][s]]
        isempty(r) && (A[:, s] .= randn(rng, p) .* k.prior_sd; continue)   # no data: the prior
        X = [d.V[i, j, q] for (i, j) in r, q in 1:p]
        y = [d.y[i, j, s] for (i, j) in r]
        A[:, s] .= polyagamma_logit_draw(rng, X, y, A[:, s]; prior_sd=k.prior_sd)
    end
    return (; a=vec(A))
end

"""
    rota_gibbs(data::OccupancyData; prior_sd=2) -> (model, sampler)

The multispecies occupancy model with the latent presence vectors `z` as
parameters, and a PracticalBayes `Gibbs` sampler built from exact kernels: the
states by [`LatentStates`](@ref), detection by [`DetectionPG`](@ref)
(Pólya–Gamma), and the occupancy and interaction coefficients by NUTS. Replace or
add blocks to extend the model.

```julia
model, spl = rota_gibbs(data)
chn = sample(model, spl, 2000; n_adapts=500, discard_initial=500)
```
"""
function rota_gibbs(d::OccupancyData; prior_sd=2.0)
    p = _priors(prior_sd)
    nb, ng, na = _sizes(d)
    model = _rota_latent(d, nb, ng, na, p.occupancy, p.interaction, p.detection)
    occ = ng > 0 ? ((:b, :g) => NUTS(0.8)) : (:b => NUTS(0.8))
    spl = Gibbs(:z => LatentStates(d), occ, :a => DetectionPG(d; prior_sd=p.detection))
    return model, spl
end
