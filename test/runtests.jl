using Test, Statistics, StableRNGs
using MultispeciesOccupancy, DataFrames, CSV
using Distributions: logpdf, Bernoulli
using LogExpFunctions: logistic, logsumexp

# Simulate S species with given natural parameters at n sites, J visits
function simulate(rng; n=400, J=4, f1=[0.3, -0.2], f2=[1.5], p=[0.5, 0.4])
    S = length(f1)
    d = MultivariateBernoulli(f1, f2)
    z = [rand(rng, d) for _ in 1:n]
    y = Array{Union{Missing,Int}}(undef, n, J, S)
    for i in 1:n, j in 1:J, s in 1:S
        y[i, j, s] = z[i][s] && rand(rng) < p[s] ? 1 : 0
    end
    return y, z
end

@testset "MultispeciesOccupancy" begin

@testset "multivariate Bernoulli" begin
    d = MultivariateBernoulli([0.2, -0.4, 1.0], [0.8, 0.0, -0.5])
    P = state_probabilities(d)
    @test sum(P) ≈ 1
    @test all(exp(logpdf(d, Vector{Bool}(d.states[k, :]))) ≈ P[k] for k in 1:8)
    ind = MultivariateBernoulli([0.2, -0.4], [0.0])             # no interaction: independent
    @test vec(sum(state_probabilities(ind) .* ind.states; dims=1)) ≈ logistic.([0.2, -0.4])
    rng = StableRNG(1)
    draws = [rand(rng, d) for _ in 1:40_000]
    freq = [count(z -> z == d.states[k, :], draws) / length(draws) for k in 1:8]
    @test maximum(abs.(freq .- P)) < 0.01
    @test all_pairs(3) == [(1, 2), (1, 3), (2, 3)]
end

@testset "OccupancyData from different inputs" begin
    rng = StableRNG(2)
    y, _ = simulate(rng; n=30, J=3)
    y[1, 3, :] .= missing
    a = OccupancyData(y; species=["fox", "badger"])
    b = OccupancyData(Dict("fox" => y[:, :, 1], "badger" => y[:, :, 2]); species=["fox", "badger"])
    c = OccupancyData((fox=y[:, :, 1], badger=y[:, :, 2]))
    @test isequal(a.y, b.y) && isequal(a.y, c.y) && c.species == ["fox", "badger"]
    @test (nsites(a), nvisits(a), nspecies(a)) == (30, 3, 2)
    sites = DataFrame(forest=randn(rng, 30), habitat=rand(rng, ["wet", "dry"], 30))
    d = OccupancyData(y; species=["fox", "badger"], sites, visits=(effort=rand(rng, 30, 3),),
                      occupancy=@formula(ψ ~ forest + habitat), interaction=@formula(ψψ ~ forest),
                      detection=@formula(p ~ effort))
    @test d.occupancy_terms == ["(Intercept)", "forest", "habitat: wet"]
    @test d.interaction_terms == ["(Intercept)", "forest"]
    @test d.detection_terms == ["(Intercept)", "effort"]
    @test size(d.V) == (30, 3, 2)
    @test occursin("detected at", sprint(show, MIME"text/plain"(), d))
    @test_throws ArgumentError OccupancyData(fill(2, 5, 2, 2))
end

@testset "marginal likelihood against brute force" begin
    rng = StableRNG(3)
    y, _ = simulate(rng; n=12, J=3, f1=[0.1, 0.4, -0.3], f2=[0.5, -0.7, 0.2], p=[0.5, 0.6, 0.3])
    y[2, 1, 2] = missing
    d = OccupancyData(y)
    b, g, a = [0.1, 0.4, -0.3], [0.5, -0.7, 0.2], [0.2, 0.0, -0.8]
    brute = sum(1:12) do i
        mvb = MultivariateBernoulli(b, g)
        logsumexp(map(1:8) do k
            z = mvb.states[k, :]
            lp = logpdf(mvb, Vector{Bool}(z))
            for s in 1:3, j in 1:3
                ismissing(y[i, j, s]) && continue
                lp += z[s] ? logpdf(Bernoulli(logistic(a[s])), y[i, j, s]) : (y[i, j, s] == 1 ? -Inf : 0.0)
            end
            lp
        end)
    end
    @test marginal_loglik(d, b, g, a) ≈ brute
    # with the latent states known, the two complete-data parts add up
    z = [Vector{Bool}(vec(any(x -> !ismissing(x) && x == 1, y[i, :, :]; dims=1))) for i in 1:12]
    @test isfinite(occupancy_loglik(d, b, g, z) + detection_loglik(d, a, z))
    # one species: MacKenzie's single-season likelihood
    d1 = OccupancyData(y[:, :, 1:1])
    ψ, p = logistic(0.3), logistic(-0.2)
    mk = sum(1:12) do i
        obs = [y[i, j, 1] for j in 1:3 if !ismissing(y[i, j, 1])]
        det = prod(p^v * (1 - p)^(1 - v) for v in obs)
        log(ψ * det + (1 - ψ) * (sum(obs) == 0))
    end
    @test marginal_loglik(d1, [0.3], Float64[], [-0.2]) ≈ mk
end

@testset "matches unmarked::occuMulti on the MesoCarnivores data" begin
    meso = mesocarnivores()
    sites = copy(meso.sites)
    for v in (:Dist_5km, :HDens_5km)                     # as R's scale()
        sites[!, v] = (sites[!, v] .- mean(sites[!, v])) ./ std(sites[!, v])
    end
    d = OccupancyData(meso.detections; sites, occupancy=@formula(ψ ~ Dist_5km + HDens_5km))
    @test (nsites(d), nvisits(d), d.species) == (1437, 3, ["bobcat", "coyote", "redfox"])
    ref = CSV.read(joinpath(@__DIR__, "reference_estimates.csv"), DataFrame)
    est = ref.estimate
    b, g, a = est[1:9], est[10:12], est[13:15]          # unmarked's order matches ours
    nll = parse(Float64, readline(joinpath(@__DIR__, "reference_negloglik.txt")))
    @test marginal_loglik(d, b, g, a) ≈ -nll rtol = 1e-9
end

@testset "fitting recovers a simulated interaction" begin
    rng = StableRNG(4)
    y, z = simulate(rng; n=500, J=4, f1=[0.2, -0.3], f2=[1.6], p=[0.5, 0.45])
    d = OccupancyData(y; species=["a", "b"])
    for method in (:nuts, :gibbs)
        fit = occupancy(d; method, draws=400, warmup=400, chains=1, rng=StableRNG(5))
        tab = coeftable(fit)
        inter = only(eachrow(tab[tab.submodel .== "interaction", :]))
        @test inter.lower < 1.6 < inter.upper
        @test inter.lower > 0                          # the positive association is found
        det = tab[tab.submodel .== "detection", :]
        @test det.mean ≈ log.([0.5, 0.45] ./ (1 .- [0.5, 0.45])) atol = 0.3
        with = conditional_occupancy(fit, "a", "b" => true)
        without = conditional_occupancy(fit, "a", "b" => false)
        @test with.mean > without.mean
        P = occupancy_probabilities(fit; conditional=true)
        detected = vec(any(x -> !ismissing(x) && x == 1, y[:, :, 1]; dims=2))
        @test all(P[detected, 1] .≈ 1)
        @test nrow(marginal_occupancy(fit)) == 2
        @test occursin("interaction", sprint(show, MIME"text/plain"(), fit))
    end
end

end
