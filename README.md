# MultispeciesOccupancy.jl

[![CI](https://github.com/EvoArt/MultispeciesOccupancy.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/EvoArt/MultispeciesOccupancy.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/EvoArt/MultispeciesOccupancy.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EvoArt/MultispeciesOccupancy.jl)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://EvoArt.github.io/MultispeciesOccupancy.jl/dev/)

Occupancy models for several species at once, with species interactions
([Rota et al. 2016](https://doi.org/10.1111/2041-210X.12587)). Each site's joint
presence of the species follows a multivariate Bernoulli distribution: first-order
terms give each species' occupancy, second-order terms say whether pairs of species
occur together more or less often than that, and both can depend on covariates.
Each species is detected imperfectly on repeat visits.

## Put the data in, get answers out

```julia
using MultispeciesOccupancy

data = mesocarnivores()          # bobcat, coyote and red fox at 1,437 camera sites
fit = occupancy(data.detections; sites=data.sites,
                occupancy   = @formula(ψ ~ Dist_5km + HDens_5km),
                interaction = @formula(ψψ ~ 1),
                detection   = @formula(p ~ 1))

coeftable(fit)                                     # every coefficient, logit scale
conditional_occupancy(fit, "redfox", "coyote" => true)
conditional_occupancy(fit, "redfox", "coyote" => false)
marginal_occupancy(fit)
```

Detections are one `sites × visits` matrix per species (0, 1 or `missing` for no
visit); site covariates are a table; visit covariates (`visits = (effort = E,)`)
can enter the detection formula. `interaction = @formula(ψψ ~ 0)` fits independent
single-species models, and one species gives MacKenzie et al.'s (2002) model.

## Plug into PracticalBayes

- `marginal_loglik(data, b, g, a)` is the likelihood with the latent states summed
  out, differentiable for NUTS; use it in your own model with `@addlogprob!`, or
  take the ready-made `rota_model(data)`.
- `rota_gibbs(data)` returns a model with the latent states as parameters and a
  `Gibbs` sampler built from exact kernels: `LatentStates` (each site's joint state
  from its full conditional), `DetectionPG` (Pólya–Gamma, from
  [PracticalEcoBayes](https://github.com/EvoArt/PracticalEcoBayes.jl)) and NUTS for the
  occupancy and interaction coefficients. Swap or add blocks to extend the model.

```julia
model, spl = rota_gibbs(OccupancyData(data.detections; sites=data.sites))
chn = sample(model, spl, 2000; n_adapts=500, discard_initial=500)
```

The likelihood agrees with unmarked's `occuMulti` to rounding error (see the tests).
The exact sums over joint states cost `2^S` per site, which suits up to about ten
species.
