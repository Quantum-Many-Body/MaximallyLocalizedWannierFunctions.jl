using MaximallyLocalizedWannierFunctions
using Documenter

DocMeta.setdocmeta!(MaximallyLocalizedWannierFunctions, :DocTestSetup, :(using MaximallyLocalizedWannierFunctions); recursive=true)

makedocs(;
    modules=[MaximallyLocalizedWannierFunctions],
    authors="waltergu <waltergu1989@gmail.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/MaximallyLocalizedWannierFunctions.jl/blob/{commit}{path}#{line}",
    sitename="MaximallyLocalizedWannierFunctions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/MaximallyLocalizedWannierFunctions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/MaximallyLocalizedWannierFunctions.jl",
    devbranch="main",
)
