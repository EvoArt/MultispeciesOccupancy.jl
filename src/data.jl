# Detection histories and covariates, with the three model formulas turned into
# design matrices.

"""
    OccupancyData(detections; species, sites, visits=(;),
                  occupancy=@formula(ψ ~ 1), interaction=@formula(ψψ ~ 1),
                  detection=@formula(p ~ 1))

Detection histories of several species at the same sites, ready to fit.

- `detections`: one `sites × visits` matrix (or table) per species, as a
  `Dict`/`NamedTuple` of `species => matrix`, or a vector of matrices with
  `species` names, or a `sites × visits × species` array. Entries are 0/1, or
  `missing` for a visit that did not happen.
- `sites`: a table with one row per site, holding the covariates the `occupancy`,
  `interaction` and `detection` formulas use.
- `visits`: covariates that vary by visit, each a `sites × visits` matrix, as a
  `NamedTuple` or `Dict` (`visits = (effort = E,)`); the `detection` formula can
  use them alongside site covariates.
- `occupancy`: first-order occupancy, one set of coefficients per species.
- `interaction`: pairwise co-occurrence (Rota et al. 2016's second-order natural
  parameters), one set of coefficients per pair of species. `@formula(ψψ ~ 0)`
  removes interactions, giving independent single-species models.
- `detection`: detection probability, one set of coefficients per species.

The left-hand sides (`ψ`, `ψψ`, `p`) only label the formulas. Scale continuous
covariates first; the default priors assume coefficients of order one.
"""
struct OccupancyData
    y::Array{Union{Missing,Int},3}          # sites × visits × species
    species::Vector{String}
    sites::DataFrame
    X::Matrix{Float64}                      # occupancy design, sites × p_occ
    W::Matrix{Float64}                      # interaction design, sites × p_int
    V::Array{Float64,3}                     # detection design, sites × visits × p_det
    occupancy_terms::Vector{String}
    interaction_terms::Vector{String}
    detection_terms::Vector{String}
    formulas::NamedTuple
    pairs::Vector{Tuple{Int,Int}}
    states::BitMatrix                       # the 2^S joint presence states
    detected::BitMatrix                     # sites × species: detected at least once
end

function OccupancyData(detections; species=nothing, sites=nothing, visits=(;),
                       occupancy=@formula(ψ ~ 1), interaction=@formula(ψψ ~ 1),
                       detection=@formula(p ~ 1))
    y, names_ = _detection_array(detections, species)
    n, J, S = size(y)
    S >= 1 || throw(ArgumentError("no species"))
    site_tbl = sites === nothing ? DataFrame() : DataFrame(Tables.columntable(sites))
    if ncol(site_tbl) == 0
        site_tbl = DataFrame(__site__ = 1:n)
    end
    nrow(site_tbl) == n || throw(DimensionMismatch("$(nrow(site_tbl)) rows of site covariates for $n sites"))
    X, xn = _design(occupancy, site_tbl)
    pairs = all_pairs(S)
    W, wn = isempty(pairs) ? (zeros(n, 0), String[]) : _design(interaction, site_tbl)
    long = _visit_table(site_tbl, visits, y)
    Vl, vn = _design(detection, long)
    V = reshape(Vl, n, J, size(Vl, 2))
    detected = BitMatrix([any(v -> !ismissing(v) && v > 0, view(y, i, :, s)) for i in 1:n, s in 1:S])
    return OccupancyData(y, names_, site_tbl, X, W, V, xn, wn, vn,
                         (; occupancy, interaction, detection), pairs, occupancy_states(S), detected)
end

"""
    nsites(data), nvisits(data), nspecies(data)

The numbers of sites, visits per site (the most at any site) and species.
"""
nsites(d::OccupancyData) = size(d.y, 1)
nvisits(d::OccupancyData) = size(d.y, 2)
nspecies(d::OccupancyData) = size(d.y, 3)

_cell(v) = ismissing(v) || (v isa AbstractFloat && isnan(v)) ? missing : Int(v)

function _matrix(m)
    M = Tables.istable(m) && !(m isa AbstractMatrix) ? Tables.matrix(m) : m
    return [_cell(v) for v in M]
end

function _detection_array(y::AbstractArray{<:Any,3}, species)
    S = size(y, 3)
    names_ = species === nothing ? ["species$s" for s in 1:S] : string.(collect(species))
    length(names_) == S || throw(DimensionMismatch("$(length(names_)) species names for $S species"))
    return _stack([_cell.(y[:, :, s]) for s in 1:S], names_)
end
function _detection_array(y::Union{AbstractDict,NamedTuple}, species)
    ks = species === nothing ? collect(keys(y)) : [_key(y, k) for k in species]
    return _stack([_matrix(y[k]) for k in ks], string.(ks))
end
_key(y::NamedTuple, k) = Symbol(k)
_key(y::AbstractDict, k) = haskey(y, k) ? k : haskey(y, string(k)) ? string(k) : Symbol(k)
function _detection_array(y::AbstractVector, species)
    species === nothing && throw(ArgumentError("give `species` names for a vector of detection matrices"))
    return _stack([_matrix(m) for m in y], string.(collect(species)))
end

function _stack(mats, names_)
    allequal(size.(mats)) || throw(DimensionMismatch("every species needs a detection matrix of the same size (sites × visits)"))
    all(m -> all(v -> ismissing(v) || v in (0, 1), m), mats) ||
        throw(ArgumentError("detections must be 0, 1 or missing"))
    n, J = size(first(mats))
    y = Array{Union{Missing,Int},3}(undef, n, J, length(mats))
    for (s, m) in enumerate(mats)
        y[:, :, s] .= m
    end
    return y, names_
end

# Rows (site i, visit j), i fastest: the site table repeated by visit plus the
# visit covariates. A visit covariate may be missing where no species was surveyed.
function _visit_table(site_tbl, visits, y)
    n, J, _ = size(y)
    long = reduce(vcat, (site_tbl for _ in 1:J))
    surveyed = [any(s -> !ismissing(y[i, j, s]), axes(y, 3)) for i in 1:n, j in 1:J]
    for (name, m) in pairs(visits)
        M = m isa AbstractMatrix ? m : Tables.matrix(m)
        size(M) == (n, J) || throw(DimensionMismatch("visit covariate $name is $(size(M)), not sites × visits = ($n, $J)"))
        bad = [ismissing(M[i, j]) && surveyed[i, j] for i in 1:n, j in 1:J]
        any(bad) && throw(ArgumentError("visit covariate $name is missing at $(count(bad)) surveyed visit(s)"))
        long[!, Symbol(name)] = vec([ismissing(M[i, j]) ? 0.0 : Float64(M[i, j]) for i in 1:n, j in 1:J])
    end
    return long
end

# Design matrix and coefficient names for the right-hand side of `f`
function _design(f::FormulaTerm, tbl::DataFrame)
    t = copy(tbl)
    t[!, :__response__] = zeros(nrow(t))
    f2 = FormulaTerm(Term(:__response__), f.rhs)
    sf = apply_schema(f2, schema(f2, t), StatisticalModel)
    M = modelcols(sf.rhs, t)
    names_ = coefnames(sf.rhs)
    M = M isa AbstractVector ? reshape(M, :, 1) : M
    return Matrix{Float64}(M), names_ isa AbstractString ? [names_] : String.(collect(names_))
end

function Base.show(io::IO, ::MIME"text/plain", d::OccupancyData)
    n, J, S = size(d.y)
    surveyed = count(!ismissing, d.y)
    println(io, "OccupancyData: $n sites × $J visits × $S species ($(surveyed) species-visits surveyed)")
    for (s, name) in enumerate(d.species)
        println(io, "  ", rpad(name, 14), " detected at ", count(view(d.detected, :, s)), " sites")
    end
    println(io, "  occupancy   ψ  ~ ", join(d.occupancy_terms, " + "))
    isempty(d.pairs) || println(io, "  interaction ψψ ~ ", isempty(d.interaction_terms) ? "0" : join(d.interaction_terms, " + "))
    print(io, "  detection   p  ~ ", join(d.detection_terms, " + "))
end
