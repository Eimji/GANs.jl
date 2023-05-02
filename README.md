# Implementation of GAN models in Julia

This repository implements the following GAN architectures using [Flux](https://fluxml.ai).
We use the MNIST dataset to train the models.

* Vanilla [GAN](https://arxiv.org/abs/1406.2661)
* [Conditional](https://arxiv.org/abs/1411.1784) Deep Convolutional GANs

Each GAN model is trained with a separate script to have a control on the model
hyperparameters. (Hyperparameters must be optimized within the context of each
machine learning project).

## Get Started

Clone the repo and go to the gans directory. Then enter the Julia Pkg REPL to
activate the GANs package:

```shell
(v1.8) pkg> activate .
```

Then install the GANs package dependencies:

```shell
(GANs) pkg>instantiate
```

You can now run a script to train a GAN:

```shell
# Vanilla GAN
julia --project=. -i src/scripts/run_gan.jl
```

To list the parameters run this command

```shell
julia --project=. src/scripts/run_gan.jl --help
```

To train a CDCGAN run

```shell
julia --project=. -i src/scripts/run_cdcgan.jl
```
