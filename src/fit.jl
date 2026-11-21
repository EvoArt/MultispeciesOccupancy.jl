# One-call fitting and the summaries people want from a multispecies
# occupancy model.

"""
    OccupancyFit

The result of [`occupancy`](@ref): the data, the posterior draws (`chain`, a
PracticalBayes `SymChain`) and the sampler used. Summarise it with
[`coeftable`](@ref), [`marginal_occupancy`](@ref),
[`conditional_occupancy`](@ref) and [`occupancy_probabilities`](@ref).
"""
struct OccupancyFit
    data::OccupancyData
    chain::Any
    method::Symbol
    b::Matrix{Float64}      # parameters × draws
    g::Matrix{Float64}
    a::Matrix{Float64}
end

"""
    occupancy(detections; sites, visits=(;), occupancy=@formula(ψ ~ 1),
              interaction=@formula(ψψ ~ 1), detection=@formula(p ~ 1),
              method=:nuts, draws=1000, warmup=1000, chains=4, prior_sd=2,
              rng=Random.default_rng(), progress=false)

Fit a multispecies occupancy model with species interactions (Rota et al. 2016)
and return an [`OccupancyFit`](@ref). The data and formula arguments are those of
[`OccupancyData`](@ref), which can also be passed directly. With one species this
is the single-species occupancy model of MacKenzie et al. (2002).

- `method = :nuts` samples the coefficients by NUTS with the latent states summed
  out ([`rota_model`](@ref)); `method = :gibbs` samples the states too, with exact
  kernels ([`rota_gibbs`](@ref)), which keeps every species' latent presence.
- `chains` run in parallel when Julia has several threads (`julia -t auto`).

```julia
fit = occupancy(Dict("bobcat" => yb, "coyote" => yc, "redfox" => yr); sites,
                occupancy = @formula(ψ ~ dist_city + housing),
                interaction = @formula(ψψ ~ 1))
conditional_occupancy(fit, "redfox", "coyote" => true)
```
"""
function occupancy(detections; method::Symbol=:nuts, draws::Integer=1000, warmup::Integer=1000,
                   chains::Integer=4, prior_sd=2.0, rng::AbstractRNG=Random.default_rng(),
                   progress::Bool=false, kw...)
    d = detections isa OccupancyData ? detections : OccupancyData(detections; kw...)
    if method === :nuts
        model, spl, skw = rota_model(d; prior_sd), NUTS(0.8), (; n_adapts=warmup)
    elseif method === :gibbs
        (model, spl), skw = rota_gibbs(d; prior_sd), (; n_adapts=warmup, discard_initial=warmup)
    else
        throw(ArgumentError("method must be :nuts or :gibbs, got :$method"))
    end
    chn = chains == 1 ? sample(rng, model, spl, draws; progress, skw...) :
          sample(rng, model, spl, Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial(), draws, chains; progress, skw...)
    nb, ng, na = _sizes(d)
    draws_of(name, n) = n == 0 ? zeros(0, length(vec(chn[:b]))) : reduce(hcat, (Float64.(v) for v in vec(chn[name])))
    return OccupancyFit(d, chn, method, draws_of(:b, nb), draws_of(:g, ng), draws_of(:a, na))
end

ndraws(f::OccupancyFit) = size(f.b, 2)

_q(x, p) = quantile(x, p)
_summary(x) = (mean=mean(x), sd=std(x), lower=_q(x, 0.025), upper=_q(x, 0.975))

"""
    coeftable(fit) -> DataFrame

Posterior mean, standard deviation and 95 % interval of every coefficient, on the
logit scale: `occupancy` (first-order, one row per species and term),
`interaction` (one per species pair and term; positive means the pair occurs
together more often than its first-order terms imply) and `detection`.
"""
function coeftable(f::OccupancyFit)
    d = f.data
    rows = NamedTuple[]
    pairname((s, t)) = d.species[s] * " × " * d.species[t]
    for (block, M, terms, groups) in (("occupancy", f.b, d.occupancy_terms, d.species),
                                       ("interaction", f.g, d.interaction_terms, pairname.(d.pairs)),
                                       ("detection", f.a, d.detection_terms, d.species))
        k = 0
        for gname in groups, term in terms
            k += 1
            push!(rows, (; submodel=block, species=gname, term, _summary(view(M, k, :))...))
        end
    end
    return DataFrame(rows)
end

# Natural parameters of every site (rows) for draw r, from design matrices X, W
function _natural(f::OccupancyFit, X, W, r)
    d = f.data
    S, P = nspecies(d), length(d.pairs)
    F1 = X * reshape(view(f.b, :, r), size(X, 2), S)
    F2 = P == 0 || size(W, 2) == 0 ? zeros(size(X, 1), P) : W * reshape(view(f.g, :, r), size(W, 2), P)
    return F1, F2
end

# Design matrices for new site covariates, with the schema of the fitted data
function _newdesign(f::FormulaTerm, original::DataFrame, new)
    t = copy(original); t[!, :__response__] = zeros(nrow(t))
    f2 = FormulaTerm(Term(:__response__), f.rhs)
    sf = apply_schema(f2, schema(f2, t), StatisticalModel)
    nt = DataFrame(Tables.columntable(new)); nt[!, :__response__] = zeros(nrow(nt))   # (already scaled)
    M = modelcols(sf.rhs, nt)
    return Matrix{Float64}(M isa AbstractVector ? reshape(M, :, 1) : M)
end

function _designs(f::OccupancyFit, newdata)
    d = f.data
    newdata === nothing && return d.X, d.W
    newdata = _apply_scaling(DataFrame(Tables.columntable(newdata)), d.scaling)
    X = _newdesign(d.formulas.occupancy, d.sites, newdata)
    W = isempty(d.pairs) ? zeros(size(X, 1), 0) : _newdesign(d.formulas.interaction, d.sites, newdata)
    return X, W
end

# Joint state probabilities at every site for draw r: sites × 2^S
function _state_probs(f::OccupancyFit, X, W, r)
    d = f.data
    F1, F2 = _natural(f, X, W, r)
    return reduce(vcat, permutedims(state_probabilities(MultivariateBernoulli(F1[i, :], F2[i, :], d.pairs, d.states)))
                  for i in axes(X, 1))
end

_species_index(d, s::Integer) = s
function _species_index(d, s)
    i = findfirst(==(string(s)), d.species)
    i === nothing && throw(ArgumentError("no species $s; the species are " * join(d.species, ", ")))
    return i
end

"""
    occupancy_probabilities(fit; newdata=nothing, conditional=false) -> Matrix

Posterior mean probability that each species occupies each site (sites ×
species). By default this is the model's occupancy probability at the site's
covariates (or at `newdata`, a table of site covariates). With
`conditional=true` it is the probability given the detections there ("realised"
occupancy): 1 where a species was detected.
"""
function occupancy_probabilities(f::OccupancyFit; newdata=nothing, conditional::Bool=false)
    d = f.data
    conditional && newdata !== nothing && throw(ArgumentError("conditional occupancy needs the fitted sites, not newdata"))
    X, W = _designs(f, newdata)
    out = zeros(size(X, 1), nspecies(d))
    for r in 1:ndraws(f)
        P = _state_probs(f, X, W, r)
        if conditional
            A = reshape(view(f.a, :, r), size(d.V, 3), nspecies(d))
            for i in axes(P, 1)
                lik = [prod(d.states[k, s] ? exp(_detection_loglik(d, A, i, s)) : (d.detected[i, s] ? 0.0 : 1.0)
                            for s in 1:nspecies(d)) for k in axes(d.states, 1)]
                P[i, :] .*= lik
                P[i, :] ./= sum(P[i, :])
            end
        end
        out .+= P * d.states
    end
    return out ./ ndraws(f)
end

"""
    marginal_occupancy(fit; newdata=nothing) -> DataFrame

Each species' occupancy probability averaged over the sites (or the rows of
`newdata`), with its posterior mean and 95 % interval.
"""
function marginal_occupancy(f::OccupancyFit; newdata=nothing)
    d = f.data
    X, W = _designs(f, newdata)
    vals = [vec(mean(_state_probs(f, X, W, r) * d.states; dims=1)) for r in 1:ndraws(f)]
    return DataFrame([(; species=d.species[s], _summary(getindex.(vals, s))...) for s in 1:nspecies(d)])
end

"""
    conditional_occupancy(fit, species, other => present; newdata=nothing) -> NamedTuple

The probability that `species` occupies a site given that `other` is present
(`true`) or absent (`false`), averaged over the sites (or `newdata`), with its
posterior mean and 95 % interval. Comparing the two conditions shows the
interaction on the probability scale:

```julia
conditional_occupancy(fit, "redfox", "coyote" => true)
conditional_occupancy(fit, "redfox", "coyote" => false)
```
"""
function conditional_occupancy(f::OccupancyFit, species, given::Pair; newdata=nothing)
    d = f.data
    s, t = _species_index(d, species), _species_index(d, first(given))
    s == t && throw(ArgumentError("condition on a different species"))
    present = Bool(last(given))
    X, W = _designs(f, newdata)
    keep = d.states[:, t] .== present
    vals = map(1:ndraws(f)) do r
        P = _state_probs(f, X, W, r)
        mean(sum(P[:, keep .& d.states[:, s]]; dims=2) ./ sum(P[:, keep]; dims=2))
    end
    return (; species=d.species[s], given=d.species[t], present, _summary(vals)...)
end

function Base.show(io::IO, ::MIME"text/plain", f::OccupancyFit)
    d = f.data
    println(io, "OccupancyFit: $(nspecies(d)) species at $(nsites(d)) sites, $(ndraws(f)) posterior draws ($(f.method))")
    tab = coeftable(f)
    w = maximum(length, tab.species; init=8)
    for block in ("occupancy", "interaction", "detection")
        sub = tab[tab.submodel .== block, :]
        nrow(sub) == 0 && continue
        println(io, "  ", block)
        for r in eachrow(sub)
            @printf(io, "    %-*s  %-14s %7.3f  [%7.3f, %7.3f]\n", w, r.species, r.term, r.mean, r.lower, r.upper)
        end
    end
end
