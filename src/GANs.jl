module GANs

using NNlib
using Flux
using Zygote
using Base: Fix2

include("models/gan.jl")
include("models/dcgan.jl")
include("training.jl")
include("output.jl")

end
