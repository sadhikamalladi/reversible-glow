import tensorflow as tf

def log_likelihood(layer, inputs, init=False, name='net'):
    """
    Compute the log likelihood for each input in a batch,
    assuming a Gaussian latent distribution.
    """
    outputs, latents, log_dets = layer.forward(inputs, reuse=(not init), name=name)
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
    # sum log prob over latent dimensions, average over batches
    return tf.reduce_mean(tf.reduce_sum(dist.log_prob(tensor), axis=[1,2,3]))

def log_likelihood_and_grad(layer, inputs, pp_logdet, var_list=None, name='net'):
    output, latents, logdets = layer.forward(inputs, reuse=True, name=name)
    assert output is None, 'extraneous non-latent outputs'
    logpx = output_log_likelihood(latents, logdets)
    logpx = tf.reduce_mean(logpx)
    grads = layer.gradients(output, latents, logdets, -logpx, var_list=var_list, name=name)
    return logpx, grads

def ll(latents, logdets, var_list=None, name='net'):
    logpx = output_log_likelihood(latents, logdets)
    logpx = tf.reduce_mean(logpx)
    return logpx
