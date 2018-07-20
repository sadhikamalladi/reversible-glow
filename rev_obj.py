import tensorflow as tf
from rev_layer import sum_batch

def log_likelihood(layer, inputs, init=False):
    """
    Compute the log likelihood for each input in a batch,
    assuming a Gaussian latent distribution.
    """
    outputs, latents, log_dets = layer.forward(inputs, reuse=(not init))
    assert outputs is None, 'extraneous non-latent outputs'
    return output_log_likelihood(latents, log_dets)


def output_log_likelihood(latents, log_dets):
    """
    Like log_likelihood(), but with a pre-computed output
    from an NVPLayer.
    """
    log_probs = log_dets
    for latent in latents:
        log_probs = log_probs + gaussian_log_prob(latent)
    return log_probs


def gaussian_log_prob(tensor):
    """
    For each sub-tensor in a batch, compute the Gaussian
    log-density.
    """
    dist = tf.distributions.Normal(0.0, 1.0)
    return sum_batch(dist.log_prob(tensor))

def log_likelihood_and_grad(layer, inputs, var_list=None):
    output, latents, logdets = layer.forward(inputs, reuse=True)
    assert output is None, 'extraneous non-latent outputs'
    log_probs = output_log_likelihood(latents, logdets)
    logpx = tf.reduce_mean(log_probs)
    grads = layer.gradients(output, latents, logdets, -logpx, var_list=var_list)
    return logpx, grads
