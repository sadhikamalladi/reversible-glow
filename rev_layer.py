"""
Required blocks for normalizing flow networks, such as
RealNVP and Glow. Includes framework for adding new layers
easily.
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from custom_ops import matrix_exponential
import math


class NVPLayer(ABC):
    """
    A layer in an invertible flow model.
    Subclasses must override _forward() and _inverse().
    Subclasses may also override test_feed_dict() and
    num_latents() if appropriate.
    """
    @property
    def num_latents(self):
        """
        The number of latents that are discarded in this
        layer. Usually 0 except for explicit latent
        filtering.
        """
        return 0

    def test_feed_dict(self):
        """
        Get a feed_dict to pass to TensorFlow when testing
        the model. Typically, this will tell BatchNorm to
        use pre-computed statistics.
        """
        return {}

    @abstractmethod
    def _forward(self, x, reuse):
        """
        Apply the layer to a batch of inputs.
        Args:
          inputs: an input batch for the layer.
          reuse: whether or not to reuse the var scope.
        Returns:
          A tuple (outputs, latents, log_det):
            outputs: the values to be passed to the next
              layer of the network. May be None for the
              last layer of the network.
            latents: A tuple of factored out Tensors.
              This may be an empty tuple.
            log_det: a batch of log of the determinants.
        """
        pass

    @abstractmethod
    def _inverse(self, y, z, reuse):
        """
        Apply the inverse of the layer.
        Args:
          y: the outputs from the layer.
          z: the latent outputs from the layer.
          reuse: whether or not to reuse the var scope
        Returns:
          The recovered input batch for the layer.
        """
        pass

    def forward(self, x, reuse, name):
        """
        Apply the layer to a batch of inputs.
        Args:
          inputs: an input batch for the layer.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.
        Returns:
          A tuple (outputs, latents, log_det):
            outputs: the values to be passed to the next
              layer of the network. May be None for the
              last layer of the network.
            latents: A tuple of factored out Tensors.
              This may be an empty tuple.
            log_det: a batch of log of the determinants.
        """
        with tf.variable_scope(name, reuse=reuse):
            return self._forward(x, reuse=reuse)

    def inverse(self, y, z, reuse, name):
        """
        Apply the inverse of the layer.
        Args:
          outputs: the outputs from the layer.
          latents: the latent outputs from the layer.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.
        Returns:
          The recovered input batch for the layer.
        """
        with tf.variable_scope(name, reuse=reuse):
            return self._inverse(y, z, reuse=reuse)

    def backward(self, outputs, outputs_grad, latents,
                 latents_grad, log_det_grad,
                 var_list=None, name='', reuse=True):
        """
        Compute a gradient through the layer.
        This is optimized for memory consumption.
        Currently, it does not support 2nd derivatives.
        Args:
          outputs: the outputs of the layer. May be None.
          outputs_grad: the gradient of the objective with
            respect to the outputs. May be None.
          latents: the latent outputs from the layer.
          latents_grad: the gradient of the objective with
            respect to the latents.
          log_det_grad: the gradient of the objective with
            respect to the log determinant.
          var_list: the list of variables to differentiate
            with respect to. If None, use all trainable
            variables.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.
        Returns:
          A tuple (upstream, grads):
            inputs: the recovered inputs to the layer.
            upstream: a Tensor representing the gradient
              of the objective with respect to the inputs
              to the layer.
            grads: a list of (gradient, variable) pairs
              for the parameters of the layer.
        """
        inputs = tf.stop_gradient(self.inverse(outputs, latents,
                                               name=name, reuse=reuse))
        new_outputs, new_latents, new_log_dets = self.forward(inputs,
                                                              name=name,
                                                              reuse=True)
        objective = tf.reduce_sum(new_log_dets *
                                  tf.stop_gradient(log_det_grad))
        if new_outputs is not None:
            objective += tf.reduce_sum(new_outputs *
                                       tf.stop_gradient(outputs_grad))
        for latent, latent_grad in zip(new_latents, latents_grad):
            objective += tf.reduce_sum(latent *
                                       tf.stop_gradient(latent_grad))
        if var_list is not None:
            variables = var_list
        else:
            variables = tf.trainable_variables()
        grads = tf.gradients(objective, [inputs] + variables)
        input_grad = grads[0]
        if input_grad is None:
            input_grad = tf.Print(tf.zeros_like(input_grad), [],
                                  message='WARNING: gradient does \
                                  not flow to inputs',
                                  first_n=1)
        var_grads = [pair for pair in zip(grads[1:], variables)
                     if pair[0] is not None]
        return inputs, input_grad, var_grads

    def gradients(self, outputs, latents, log_det,
                  loss, var_list=None, name='', reuse=True):
        """
        Perform backpropagation through the layer using
        the backward() method.
        This computes gradients without needing to store
        intermediate Tensors from the forward pass.
        Currently, it does not support 2nd derivatives.
        Args:
          outputs: the layer outputs.
          latents: the layer's latent outputs.
          log_det: the output log determinants.
          loss: the loss value resulting from the latents
            and log determinants.
          var_list: the variables to find gradients for,
            or None to use all trainable variables.
          name: the name of the variable scope.
          reuse: the variable scope reuse flag.
        Returns:
          A list of (gradient, variable) pairs.
        """
        assert len(latents) == self.num_latents
        if outputs is not None:
            outputs_grad = tf.gradients(loss, outputs)[0]
            if outputs_grad is None:
                outputs_grad = tf.zeros_like(outputs)
        else:
            outputs_grad = None
        latents_grad = [grad if grad is not None else tf.zeros_like(latent)
                        for grad, latent in zip(tf.gradients(loss, latents),
                                                latents)]
        log_det_grad = tf.gradients(loss, log_det)[0]
        if log_det_grad is None:
            log_det_grad = tf.zeros_like(log_det)
        return self.backward(outputs, outputs_grad, latents,
                             latents_grad, log_det_grad,
                             var_list=var_list, name=name,
                             reuse=reuse)[2]


class Network(NVPLayer):
    """
    A feed-forward composition of NVP layers.
    """

    def __init__(self, layers, layer_names, shared=False):
        self.layers = layers
        self.layer_names = layer_names
        self.shared = shared

    @property
    def num_latents(self):
        return 1 + sum(l.num_latents for l in self.layers)

    def test_feed_dict(self):
        res = {}
        for layer in self.layers:
            res.update(layer.test_feed_dict())

    def _forward(self, inputs, reuse):
        """
        Computes the forward pass through all the
        layers and returns the latents
        and log determinant of the composed
        transformations.
        """
        latents = []
        outputs = inputs
        log_det = tf.zeros(shape=[tf.shape(inputs)[0]], dtype=inputs.dtype)
        if self.shared:
            coup_names = set([n for n in self.layer_names if 'coupling' in n])
            inds = [self.layer_names.index(n) for n in coup_names]
            layer_ind = 0
        for name, layer in zip(self.layer_names, self.layers):
            if self.shared:
                if layer_ind in inds:
                    true_reuse = reuse
                elif 'coupling' in name:
                    true_reuse = True
                else:
                    true_reuse = reuse
                layer_ind += 1
            else:
                true_reuse = reuse
            outputs, sub_latents, sub_log_det = layer.forward(outputs,
                                                              name=name,
                                                              reuse=true_reuse)
            # save the latents that were discarded along the forward pass
            latents.extend(sub_latents)
            log_det = log_det + sub_log_det
        # save the final latents to come out of the network
        latents.append(outputs)
        return None, tuple(latents), log_det

    def _inverse(self, outputs, latents, reuse):
        # make sure we get only latents into the inverse
        assert outputs is None
        assert len(latents) == self.num_latents
        inputs = latents[-1]
        latents = latents[:-1]
        # compute the inverse for all the layers in the network
        for layer_name, layer in reversed(list(zip(self.layer_names,
                                                   self.layers))):
            # special case: latents and outputs are
            # different in layer (some latents are
            # discarded between scale blocks)
            if layer.num_latents > 0:
                sub_latents = latents[-layer.num_latents:]
                latents = latents[:-layer.num_latents]
            else:
                sub_latents = ()
            inputs = layer.inverse(inputs, sub_latents,
                                   name=layer_name, reuse=reuse)
        return inputs

    def backward(self, outputs, outputs_grad,
                 latents, latents_grad, log_det_grad,
                 var_list=None, name='', reuse=True):
        with tf.variable_scope(name, reuse=reuse):
            assert outputs is None
            assert outputs_grad is None
            outputs = latents[-1]
            outputs_grad = latents_grad[-1]
            latents = latents[:-1]
            latents_grad = latents_grad[:-1]
            total_grads = {}
            prev_grads = []
            for layer_name, layer in reversed(list(zip(self.layer_names,
                                                       self.layers))):
                if layer.num_latents > 0:
                    sub_latents = latents[-layer.num_latents:]
                    sub_latents_grad = latents_grad[-layer.num_latents:]
                    latents = latents[:-layer.num_latents]
                    latents_grad = latents_grad[:-layer.num_latents]
                else:
                    sub_latents = ()
                    sub_latents_grad = ()
                outputs, outputs_grad, vars_grad = layer.backward(outputs,
                                                                  outputs_grad,
                                                                  sub_latents,
                                                                  sub_latents_grad,
                                                                  log_det_grad,
                                                                  var_list=var_list,
                                                                  name=layer_name)
                for grad, var in vars_grad:
                    if var in total_grads:
                        total_grads[var] += grad
                    else:
                        total_grads[var] = grad
                prev_grads = [g for g, _ in vars_grad]
            return (outputs,
                    outputs_grad,
                    [(grad, var) for var, grad in total_grads.items()])


class Squeeze(NVPLayer):
    def _forward(self, x, reuse, factor=2):
        assert factor >= 1
        height, width, n_channels = gs(x)[1:]
        assert height % factor == 0 and width % factor == 0
        x = tf.reshape(x, [-1, height//factor, factor, width//factor,
                           factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(x, [-1, height//factor, width//factor, n_channels*factor*factor])
        return x, (), tf.zeros(tf.shape(x)[0], dtype=x.dtype)

    def _inverse(self, y, z, reuse, factor=2):
        assert factor >= 1
        height, width, n_channels = gs(y)[1:]
        assert n_channels >= 4 and n_channels % 4 == 0
        x = tf.reshape(y, (-1, height, width, int(n_channels / factor ** 2),
                           factor, factor))
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, (-1, int(height * factor), int(width * factor),
                           int(n_channels / factor ** 2)))
        return x


class Actnorm(NVPLayer):
    def compute(self, x, logs, t, backward):
        if backward:
            return (x * tf.exp(-logs)) - t
        else:
            return (x + t) * tf.exp(logs)

    def logdet(self, x, logs):
        h, w = gs(x)[1:3]
        val = tf.reduce_sum(logs) * h * w
        return val

    def _forward(self, x, reuse, eps=1e-6):
        t = tf.get_variable("t", (1, 1, 1, gs(x)[-1]),
                            trainable=True, dtype=x.dtype)
        logs = tf.get_variable("logs", (1, 1, 1, gs(x)[-1]),
                               trainable=True, dtype=x.dtype)
        if not reuse:
            x_mean, x_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
            logs_init = tf.log(1. / (tf.sqrt(x_var) + eps))
            t_init = - x_mean
            logsop = logs.assign(logs_init)
            top = t.assign(t_init)
            with tf.control_dependencies([logsop, top]):
                an = self.compute(x, logs_init, t_init, backward=False)
        else:
            an = self.compute(x, logs, t, backward=False)

        return an, (), self.logdet(x, logs)

    def _inverse(self, y, z, reuse, eps=1e-6):
        t = tf.get_variable("t", (1, 1, 1, gs(y)[-1]),
                            trainable=True, dtype=y.dtype)
        logs = tf.get_variable("logs", (1, 1, 1, gs(y)[-1]),
                               trainable=True, dtype=y.dtype)
        if not reuse:
            x_mean, x_var = tf.nn.moments(y, axes=[0, 1, 2], keep_dims=True)
            logs_init = tf.log(1. / (tf.sqrt(x_var) + eps))
            t_init = -x_mean
            logsop = logs.assign(logs_init)
            top = t.assign(t_init)
            with tf.control_dependencies([logsop, top]):
                an = self.compute(y, logs_init, t_init, backward=True)
        else:
            an = self.compute(y, logs, t, backward=True)
        return an


def split(x):
    nc = gs(x)[-1] // 2
    return x[:, :, :, :nc], x[:, :, :, nc:]


def combine(x1, x2):
    return tf.concat([x1, x2], axis=3)


def default_initializer(dtype, std=0.05):
    return tf.random_normal_initializer(0., std, dtype=dtype)


class Coupling(NVPLayer):
    def __init__(self, dim=32, step=0, shared=False):
        self.dim = dim
        self.actnorm = Actnorm()
        self.step = step
        self.shared = shared

    def NN(self, x, name, reuse):
        def conv(h, d, k, name, nonlin=True):
            if nonlin:
                h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                     name=name, use_bias=False,
                                     kernel_initializer=default_initializer(h.dtype))
                h = self.actnorm.forward(h, reuse=reuse, name=name+'_an')[0]
                h = tf.nn.relu(h)
                return h
            else:
                h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                     name=name, use_bias=True,
                                     kernel_initializer=tf.zeros_initializer(dtype=h.dtype),
                                     bias_initializer=tf.zeros_initializer(dtype=h.dtype))
                logs = tf.get_variable("logs", shape=[1, 1, 1, d], dtype=h.dtype,
                                       initializer=tf.zeros_initializer(h.dtype))
                s = tf.exp(logs)
                return h * s

        with tf.variable_scope(name, reuse=reuse):
            nc = gs(x)[-1]
            h = conv(x, self.dim, 3, "h1")
            h = conv(h, self.dim, 1, "h2")
            h = conv(x, nc,  3, "h3", nonlin=False)
            return h

    def get_vars(self, feats, reuse, eps=1e-6):
        logit_s = self.NN(feats, "logit_s", reuse) + 2.
        s = tf.sigmoid(logit_s) + eps
        t = self.NN(feats, "t", reuse)
        logdet = tf.reduce_sum(tf.log_sigmoid(logit_s),
                               axis=[1, 2, 3]) + tf.cast(tf.log(eps), feats.dtype)
        return s, t, logdet

    def _forward(self, x, reuse):
        x1, x2 = split(x)
        s, t, logdet = self.get_vars(x1, reuse)
        y1 = x1
        y2 = (x2 + t) * s
        y = combine(y1, y2)
        return y, (), logdet

    def _inverse(self, y, z, reuse):
        y1, y2 = split(y)
        s, t, logdet = self.get_vars(y1, True)
        x1 = y1
        x2 = (y2 / s) - t
        x = combine(x1, x2)
        return x


def random_rotation_matrix(nc):
    return np.linalg.qr(np.random.randn(nc, nc))[0]


class Invconv(NVPLayer):

    def conv(self, f, k):
        return tf.nn.conv2d(f, k, [1, 1, 1, 1], "SAME")

    def _forward(self, x, reuse, eps=1e-6):
        hh, ww, nc = gs(x)[1:]
        if x.dtype == tf.float32:
            nptype = np.float32
        else:
            nptype = np.float64
        w = tf.get_variable("w", dtype=x.dtype,
                            initializer=random_rotation_matrix(nc).astype(nptype))
        det = tf.matrix_determinant(w)
        logdet = tf.log(tf.abs(det) + eps) * hh * ww
        kernel = tf.reshape(w, [1, 1, nc, nc])
        y = self.conv(x, kernel)
        return y, (), logdet

    def _inverse(self, y, z, reuse):
        hh, ww, nc = gs(y)[1:]
        if y.dtype == tf.float32:
            nptype = np.float32
        else:
            nptype = np.float64
        w = tf.get_variable("w", dtype=y.dtype,
                            initializer=random_rotation_matrix(nc).astype(nptype))
        kernel = tf.reshape(tf.matrix_inverse(w), [1, 1, nc, nc])
        x = self.conv(y, kernel)
        return x


# uses a skew symmetric matrix to make the invertible 1x1 conv kernel instead of
# QR decomposition of a random matrix
class SkewInvconv(NVPLayer):

    def conv(self, f, k):
        return tf.nn.conv2d(f, k, [1, 1, 1, 1], "SAME")

    def _forward(self, x, reuse):
        hh, ww, nc = gs(x)[1:]
        if x.dtype == tf.float32:
            nptype = np.float32
        else:
            nptype = np.float64
        rotation = tf.get_variable("1x1_conv_weight",
                                   dtype=x.dtype,
                                   initializer=random_rotation_matrix(nc).astype(nptype),
                                   trainable=True)
        rotation = (rotation - tf.matrix_transpose(rotation)) / 2.0  # make skew symmetric
        rotation = matrix_exponential(rotation, name="MatrixExpFor1x1Convolution")

        # since the rotation matrix is orthogonal now we
        # know that the determinant is +/- 1 and its absolute
        # value is 1. Hence the logarithm is 0 always
        logdet = 0
        _rot = tf.cast(rotation, x.dtype)
        kernel = tf.reshape(_rot, shape=[1, 1, nc, nc])
        y = self.conv(x, kernel)

        return y, (), logdet

    def _inverse(self, y, z, reuse):
        hh, ww, nc = gs(y)[1:]
        if y.dtype == tf.float32:
            nptype = np.float32
        else:
            nptype = np.float64
        rotation = tf.get_variable("1x1_conv_weight",
                                   dtype=y.dtype,
                                   initializer=random_rotation_matrix(nc).astype(nptype),
                                   trainable=True)
        rotation = (rotation - tf.matrix_transpose(rotation)) / 2.0  # make skew symmetric
        rotation = matrix_exponential(rotation, name="MatrixExpFor1x1Convolution")
        _rot = tf.cast(tf.matrix_transpose(rotation), y.dtype)
        kernel = tf.reshape(_rot, shape=[1, 1, nc, nc])
        x = self.conv(y, kernel)

        return x


def split(x):
    nc = gs(x)[-1] // 2
    return x[:, :, :, :nc], x[:, :, :, nc:]


def combine(x1, x2):
    return tf.concat([x1, x2], axis=3)


class FilterLatents(NVPLayer):
    @property
    def num_latents(self):
        return 1

    def _forward(self, x, reuse):
        x, xi = split(x)
        return x, (xi,), tf.zeros([tf.shape(x)[0]], dtype=x.dtype)

    def _inverse(self, y, z, reuse):
        z = combine(y, z[0])
        return z


def add_noise(x, dtype=tf.float32):
    x = tf.cast(x, dtype)
    noise = tf.random_uniform(tf.shape(x), dtype=dtype)
    x = x + noise
    x = x / 256
    return x


class ImageProcessing(NVPLayer):
    def __init__(self, alpha=1e-6):
        self.alpha = alpha

    def logdetgrad(self, x):
        s = self.alpha + (1-2*self.alpha)*x
        ldg = -tf.log(s - s * s) + math.log(1 - 2*self.alpha)
        ldg = tf.reshape(ldg, [tf.shape(x)[0], -1])
        ldg = tf.reduce_sum(ldg, axis=-1)
        return ldg

    def _forward(self, x, reuse):
        noisy = add_noise(x)
        s = self.alpha + (1-2*self.alpha)*noisy
        y = tf.log(s) - tf.log(1-s)
        ldgrad = self.logdetgrad(noisy)
        return y, (), ldgrad

    def _inverse(self, y, z, reuse):
        x = (tf.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        return x


def sum_batch(tensor):
    """
    Compute a 1-D batch of sums.
    """
    return tf.reduce_sum(tf.reshape(tensor, [tf.shape(tensor)[0], -1]), axis=1)


def gs(x):
    return x.get_shape().as_list()
