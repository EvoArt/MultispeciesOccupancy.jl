const _DATA = joinpath(@__DIR__, "..", "data")

"""
    mesocarnivores() -> (; detections, sites)

Camera-trap detections of bobcat, coyote and red fox at 1,437 sites, each
surveyed on three occasions: the data of Rota et al. (2016) as distributed with
the R package unmarked (`MesoCarnivores`).

`detections` is a `NamedTuple` of `1437 × 3` matrices (`bobcat`, `coyote`,
`redfox`; 1 = detected, `missing` = no survey). `sites` holds the site
covariates: `Dist_5km` (proportion of disturbed land within 5 km), `HDens_5km`
(housing density within 5 km), `Latitude`, `Longitude`, `People_site` (people
photographed per day) and `Trail` (camera on a trail).

```julia
data = mesocarnivores()
fit = occupancy(data.detections; sites=data.sites,
                occupancy=@formula(ψ ~ Dist_5km + HDens_5km))
```
"""
function mesocarnivores()
    read(name) = CSV.read(joinpath(_DATA, name), DataFrame; missingstring="NA")
    det(name) = [ismissing(v) ? missing : Int(v) for v in Matrix(read("y_$name.csv"))]
    detections = (bobcat=det("bobcat"), coyote=det("coyote"), redfox=det("redfox"))
    return (; detections, sites=read("sitecovs.csv"))
end
