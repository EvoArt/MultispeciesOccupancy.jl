"""
    MultispeciesOccupancy

Occupancy models for several species at once, with species interactions (Rota et
al. 2016, Methods Ecol. Evol. 7:1164): each site's joint presence of the species
follows a multivariate Bernoulli distribution whose first-order terms
(occupancy) and second-order terms (co-occurrence) depend on covariates, and
each species is detected imperfectly on repeat visits.

Three ways in:

- [`occupancy`](@ref): give the detection histories and formulas, get a fit with
  coefficient tables and occupancy probabilities.
- [`rota_model`](@ref) and [`marginal_loglik`](@ref): the model, or its
  likelihood with the latent states summed out, for PracticalBayes (NUTS, or a
  model of your own via `@addlogprob!`).
- [`rota_gibbs`](@ref): a Gibbs sampler from exact kernels ([`LatentStates`](@ref),
  [`DetectionPG`](@ref)) that can be extended block by block.
"""
module MultispeciesOccupancy

using PracticalBayes
using PracticalBayes: AbstractGibbsKernel, ModelConditional
using PracticalEcoBayes: polyagamma_logit_draw
using Distributions
using Distributions: DiscreteMultivariateDistribution
using DataFrames, Tables
import CSV
using StatsModels: @formula, FormulaTerm, Term, apply_schema, schema, modelcols, coefnames, StatisticalModel
import StatsModels: coeftable
using LogExpFunctions: logsumexp, softmax, log1pexp
using Random: Random, AbstractRNG
using Statistics: mean, std, quantile
using Printf: @printf

export OccupancyData, nsites, nvisits, nspecies
export MultivariateBernoulli, state_probabilities, occupancy_states, all_pairs
export marginal_loglik, occupancy_loglik, detection_loglik
export rota_model, rota_gibbs, LatentStates, DetectionPG
export occupancy, OccupancyFit, coeftable, occupancy_probabilities, marginal_occupancy, conditional_occupancy
export @formula
export mesocarnivores

include("likelihood.jl")
include("data.jl")
include("models.jl")
include("fit.jl")
include("datasets.jl")

end
