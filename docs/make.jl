using Documenter
using MultispeciesOccupancy

DocMeta.setdocmeta!(MultispeciesOccupancy, :DocTestSetup,
                   :(using MultispeciesOccupancy); recursive=true)

makedocs(;
    modules=[MultispeciesOccupancy],
    checkdocs=:none,
    sitename="MultispeciesOccupancy.jl",
    repo="https://github.com/EvoArt/MultispeciesOccupancy.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/MultispeciesOccupancy.jl",
        repolink="https://github.com/EvoArt/MultispeciesOccupancy.jl",
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/EvoArt/MultispeciesOccupancy.jl")
