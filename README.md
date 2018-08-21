# Constant Memory Invertible Flow Implementation
This repository has the code to run a O(1) memory utilization implementation of an invertible flow generative model. Invertible flow models such as [RealNVP](https://arxiv.org/abs/1605.08803) and [Glow](https://arxiv.org/abs/1807.03039) have become popular choices for generative modeling because they allow for efficient sampling and construct a bijection between latents and observations. However, these models require that each layer be invertible, which severely limits the transformation power of each individual layer. Therefore, SOTA performance models are very large and difficult to run on a single GPU without gradient checkpointing.

The goal of this repository is to allow the construction of arbitrarily deep flows in this class of networks, with the trade-off of time of course. Since the entire model is invertible, we can force TensorFlow not to save any activations on the forward pass and recompute them dynamically when computing the gradients. Thus, no matter how many flow steps there are, the network will use constant memory.

## Requirements
Tested on Tensorflow 1.9.0

## Running the Model
`python rev_glow.py` runs the main script for the Glow model. Some common options are:
- `--dataset`: imagenet, cifar, or mnist. To add a new dataset, modify `utils.py` accordingly.
- `--width`: width of the layers in the coupling parts
- `--depth`: number of flow steps per scale block
- `--n_levels`: number of scale blocks
- `--n_bits_x`: preprocessing hyperparameter (more bits for more "complex" images)

The default parameters in addition to `--precision float64` (running the network in float64 precision) achieve reported performance of the Glow model on MNIST and CIFAR10.

## Code Structure
The layers, contained in `rev_layer.py` are all subclasses of `NVPLayer`. `NVPLayer` requires the log determinant of the layer transformation, the forward pass, and the inverse pass to be defined. The layers are concatenated together in the `Network` object, which condenses forward and backward propagation. 

Each forward pass must return the output (which flows into the forward pass of the next layer), a tuple of latents (which is usually empty and represents the discarded latents from the layer), and the log determinant of the transformation (a single number).

## Debugging and More
To profile speed, use `--profile` to run the network for a few SGD steps. Then, open the `timeline.json` file in Chrome by navigating to `chrome://tracing` and loading the file in. 

To share parameters across coupling layers within the same scale block, use `--shared_coupling`.

To test how numerical precision errors propagate and cause the computed gradients to diverge from the true ones, use `--test_grad_divergence`. Keep in mind that enabling this flag will force the gradients to be computed as they normally are, which makes the implementation no longer constant memory.
