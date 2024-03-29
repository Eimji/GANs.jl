using GANs: get_vanilla_discriminator, get_vanilla_generator, train_dscr!, train_gen!
using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux
using CUDA
using Zygote
using Logging
using TensorBoardLogger
using ArgParse
using Plots

"""
Trains a vanilla GAN on the MNIST dataset

This is adapated from the pytorch tutorial presented here
    https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

Generative Adversarial Networks by I. Goodfellow et al. 2014 https://arxiv.org/abs/1406.2661
"""



s = ArgParseSettings()
@add_arg_table s begin
    "--lr_dscr"
    help = "Learning rate for the discriminator. Default = 0.0002"
    arg_type = Float64
    default = 0.0002
    "--lr_gen"
    help = "Learning rate for the generator. Default = 0.0002"
    arg_type = Float64
    default = 0.0002
    "--batch_size"
    help = "Batch size. Default = 1024"
    arg_type = Int
    default = 128
    "--num_iterations"
    help = "Number of training iterations. Not epochs. Default = 1000"
    arg_type = Int
    default = 1000
    "--latent_dim"
    help = "Size of the latent dimension. Default = 100"
    arg_type = Int
    default = 100
    "--optimizer"
    help = "Optimizer for both, generator and discriminator. Defaults to ADAM"
    arg_type = String
    default = "ADAM"
    "--activation"
    help = "Activation function. Defaults to leakyrelu with 0.2"
    arg_type = String
    default = "relu"
    "--activation_alpha"
    help = "Optional parameter for activation function, α in leakyrelu, celu, elu, etc."
    arg_type = Float64
    default = 0.1
    "--train_k"
    help = "Number of steps that the discriminator is trained while holding the generator fixed."
    arg_type = Int
    default = 4
    "--prob_dropout"
    help = "Probability for Dropout"
    arg_type = Float64
    default = 0.3
    "--output_period"
    help = "Period between graphical output of the generator, in units of training iterations."
    arg_type = Int
    default = 100
end

args = parse_args(s)


for (arg, val) in args
    println("   $arg => $val")
end

# Instantiate TensorBoardLogger
tb_logger = TBLogger("logs/testlog")
with_logger(tb_logger) do
    @info "hyperparameters" args
end
# Number of features per MNIST sample
n_features = 28 * 28

# Load MNIST train and test data
train_x, train_y = MNIST(Float32, :train)[:]
test_x, test_y = MNIST(Float32, :test)[:]

# This dataset has pixel values ∈ [0:1]. Map these to [-1:1]
# See GAN hacks: https://github.com/soumith/ganhacks
train_x = 2.0f0 * reshape(train_x, 28, 28, 1, :) .- 1.0f0 |> gpu
test_x = 2.0f0 * reshape(test_x, 28, 28, 1, :) .- 1.0f0 |> gpu

train_y = Flux.onehotbatch(train_y, 0:9) |> gpu
test_y = Flux.onehotbatch(test_y, 0:9) |> gpu

# Insert the train images and labels into a DataLoader
train_loader = DataLoader((data=train_x, label=train_y), batchsize=args["batch_size"], shuffle=true)

# Define the discriminator network.
# The networks takes a flattened 28x28=784 image as input and outputs the
# probability of the image belonging to the real dataset.
discriminator = get_vanilla_discriminator(args)

# The generator will generate images which come from the learned
# distribution. The output layer has a tanh activation function
# which maps the output to [-1:1], the same range as in the
# pre-processed MNIST images
generator = get_vanilla_generator(args)

# Optimizer for the discriminator
opt_dscr = getfield(Flux, Symbol(args["optimizer"]))(args["lr_dscr"])
opt_gen = getfield(Flux, Symbol(args["optimizer"]))(args["lr_gen"])
# Extract the parameters of the discriminator and generator
ps_dscr = Flux.params(discriminator)
ps_gen = Flux.params(generator)

println("Entering training loop")
lossvec_gen = zeros(args["num_iterations"])
lossvec_dscr = zeros(args["num_iterations"])

# This loop follows the algorithm described in Goodfellow et al. 2014
# for number of training iterations
with_logger(tb_logger) do
    for n in 1:args["num_iterations"]
        println("Training $n...")
        loss_sum_gen = 0.0f0
        loss_sum_dscr = 0.0f0

        # for k steps do
        for (x, y) in Base.Iterators.take(train_loader, args["train_k"])
            # Samples in the current batch, handles edge case
            this_batch = size(x)[end]
            # Train the discriminator
            # - Flatten the images, which squashes all dimensions and keeps the last dimension, which is the batch dimension
            real_data = Flux.flatten(x)

            # Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
            noise = randn(args["latent_dim"], this_batch) |> gpu
            fake_data = generator(noise)

            # Update the discriminator by ascending its stochastic gradient
            # ∇_theta_d 1\m Σ_{i=1}^{m} [ log D(xⁱ) + log(1 - D(G(zⁱ))]
            loss_dscr = train_dscr!(discriminator, real_data, fake_data, ps_dscr, opt_dscr)
            loss_sum_dscr += loss_dscr
        end

        # Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
        # Update the generator by descending its stochastic gradient:
        # ∇_theta_g 1/m Σ_{i=1}^{m} log(1 - D(G(zⁱ))
        loss_gen = train_gen!(discriminator, generator, args["latent_dim"], ps_gen, opt_gen, args["batch_size"])
        loss_sum_gen += loss_gen


        # Add the per-sample loss of the generator and discriminator
        lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
        lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]

        if n % args["output_period"] == 0
            noise = randn(args["latent_dim"], 4) |> gpu
            fake_img = reshape(generator(noise), 28, 4 * 28) |> cpu
            fake_img[fake_img.>1.0] .= 1.0
            fake_img[fake_img.<-1.0] .= -1.0
            fake_img = (fake_img .+ 1.0) .* 0.5
            log_image(tb_logger, "generatedimage", fake_img, ImageFormat(202))
        end
        @info "test" loss_generator = lossvec_gen[n] loss_discriminator = lossvec_dscr[n]
    end # Training loop
end # Logger
