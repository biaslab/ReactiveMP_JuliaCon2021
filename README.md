# ReactiveMP notebooks for JuliaCon 2021

This repository contains an experiments for [ReactiveMP.jl](https://github.com/biaslab/ReactiveMP.jl) package [presented](https://pretalx.com/juliacon2021/talk/J7Z9PL/) on [JuliaCon 2021](https://juliacon.org/2021/).

## Prerequisites

To run this notebook you need [Julia](https://julialang.org) 1.6.x installed on your system.

To start open a Julia REPL in project directory with the following command:

```bash
julia --project
```

To instantiate project dependencies run:

```julia
julia> import Pkg; Pkg.instantiate()
```

This command will install all needed dependencies in a local project environment.

## Notebooks

Experiments in this repository use [Pluto notebooks](https://github.com/fonsp/Pluto.jl), to start Pluto server run the following command:

```julia
julia> import Pluto; Pluto.run()
```

In Pluto.jl UI open the `notebooks/presentation.jl` notebook. It takes some time to initialise notebook since Pluto.jl executes all cells automatically upon opening.

**Note**: Interactive plotting has been tested properly in Chrome web-browser only. We cannot guarantee proper interactive visualisation in other web browsers.
