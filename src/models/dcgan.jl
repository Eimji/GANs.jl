"""
Deep Convolutional GAN
https://fluxml.ai/tutorialposts/2021-10-08-dcgan-mnist/
"""

function get_cdcgan_discriminator(args)
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
    
    return Chain(Conv((3, 3), 11 => 32, act),
                 Conv((5, 5), 32 => 64, act),
                 MaxPool((2, 2)),
                 x -> Flux.flatten(x),
                 Dense(11 * 11 * 64, 256, relu),
                 Dropout(0.3),
                 Dense(256, 1, sigmoid)) |> gpu;
end


function get_cdcgan_generator(args)
    # This is just the generator proposed in the Keras tutorial
    # https://github.com/malzantot/Pytorch-conditional-GANs
    latent_dim = args["latent_dim"]
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
    return Chain(Dense(latent_dim + 10, 20 * 20 * (latent_dim + 10), act),
                        x -> reshape(x, (20, 20, latent_dim + 10, :)),
                        ConvTranspose((5, 5), latent_dim + 10 => 32, bias=false), 
                        BatchNorm(32, act),
                        ConvTranspose((5, 5), 32 => 1, tanh, bias=false)) |> gpu;
end
