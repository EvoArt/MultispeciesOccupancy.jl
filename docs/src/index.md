```@meta
CurrentModule = MultispeciesOccupancy
```

# MultispeciesOccupancy.jl

Occupancy models for several species at once, with species interactions
([Rota et al. 2016](https://doi.org/10.1111/2041-210X.12587)). Each site's joint presence of
the species follows a [`MultivariateBernoulli`](@ref) distribution: first-order terms give
each species' occupancy, second-order terms say whether pairs of species occur together
more (positive) or less (negative) often than that, and both can depend on covariates.
Each species is detected imperfectly on repeat visits.

## Put the data in, get answers out

```julia
using MultispeciesOccupancy

data = mesocarnivores()          # bobcat, coyote and red fox at 1,437 camera sites
fit = occupancy(data.detections; sites=data.sites,
                occupancy   = @formula(ψ ~ Dist_5km + HDens_5km),
                interaction = @formula(ψψ ~ 1),
                detection   = @formula(p ~ 1))

coeftable(fit)
conditional_occupancy(fit, "redfox", "coyote" => true)
conditional_occupancy(fit, "redfox", "coyote" => false)
```

Scale continuous covariates first. `interaction = @formula(ψψ ~ 0)` removes the
interactions; one species gives the single-species model of MacKenzie et al. (2002).

## In PracticalBayes

- [`marginal_loglik`](@ref): the likelihood with the latent states summed out,
  differentiable for NUTS; add it to any model with `@addlogprob!`, or use
  [`rota_model`](@ref).
- [`rota_gibbs`](@ref): latent states as parameters, with a `Gibbs` sampler of exact
  kernels ([`LatentStates`](@ref), [`DetectionPG`](@ref)) and NUTS for the coefficients.
- [`occupancy_loglik`](@ref) and [`detection_loglik`](@ref): the two complete-data parts,
  for building other samplers.

The likelihood agrees with unmarked's `occuMulti` to rounding error on the
MesoCarnivores data (tested). Exact sums over the `2^S` joint states suit up to about
ten species.

## API reference

```@autodocs
Modules = [MultispeciesOccupancy]
```
