module GANs

using NNlib
using Flux
using Zygote
using Base:Fix2

include.(filter(contains(r".jl$"), readdir("../models"; join=true)))
include("training.jl")
include("output.jl")

end
