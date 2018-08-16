from random import random

import numpy as np
import pytest
import tensorflow as tf

from rev_layer import Network, Coupling, Invconv, Actnorm, Squeeze, FilterLatents

def test_inverses():
    l = Coupling()
    inp = np.random.normal(size=(2, 32, 32, 3))
    init_inp = np.random.normal(size=(1024,32,32,3))
    _inverse_test(l, init_inp, inp)

def _inverse_test(layer, init_inp, inputs):
    with tf.Graph().as_default():  # pylint: disable=E1129
        with tf.Session() as sess:
            in_constant = tf.constant(inputs)
            init_constant = tf.constant(init_inp)
            with tf.variable_scope('model', reuse=False):
                init_op = layer.forward(init_constant, name='layer', reuse=False)
            with tf.variable_scope('model', reuse=True):
                out, latent, _ = layer.forward(in_constant, name='layer', reuse=True)
                inverse = layer.inverse(out, latent, name='layer', reuse=True)
            sess.run(tf.global_variables_initializer())
            sess.run(init_op)
            actual = sess.run(inverse)
            assert not np.isnan(actual).any()
            assert np.allclose(actual, inputs, atol=1e-4, rtol=1e-4)

"""
def test_gradients():
#Test that manual gradient computation works properly.
    with tf.Graph().as_default():  # pylint: disable=E1129
        layers = [
            Coupling()
        ]
        layer_names = [str(i) for i in range(len(layers))]
        network = Network(layers, layer_names)
        inputs = tf.random_uniform([3, 8, 8, 4])
        outputs, latents, log_det = network.forward(inputs, reuse=False, name='net')
        outputs, latents, log_det = network.forward(inputs, reuse=True, name='net')
        loss = (tf.reduce_sum(tf.stack([(random() + 1) * tf.reduce_sum(x) for x in latents])) +
                (random() + 1) * tf.reduce_sum(log_det))

        manual_grads = network.gradients(outputs, latents, log_det, loss, name='net')
        manual_grads = {var: grad for grad, var in manual_grads}
        manual_grads = [manual_grads[v] for v in tf.trainable_variables()]

        true_grads = tf.gradients(loss, tf.trainable_variables())

        diffs = [tf.reduce_max(x - y) for x, y in zip(manual_grads, true_grads)]
        max_diff = tf.reduce_max(tf.stack(diffs))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            _randomized_init(sess)
            assert sess.run(max_diff) < 1e-4


@pytest.mark.parametrize("layer,shape",
                         [(Squeeze(), (2,32,32,3))])
def test_log_det(layer, shape):
    #Tests log determinants.
    inputs = np.random.normal(size=shape).astype('float32')
    _log_det_test(layer, inputs)


def _log_det_test(layer, inputs):
    with tf.Graph().as_default():  # pylint: disable=E1129
        with tf.Session() as sess:
            in_vecs = tf.constant(np.reshape(inputs, [inputs.shape[0], -1]))
            in_tensor = tf.reshape(in_vecs, inputs.shape)
            with tf.variable_scope('model'):
                out, _, log_dets = layer.forward(in_tensor, name='layer', reuse=False)
                out_vecs = tf.reshape(out, in_vecs.get_shape())
            jacobians = _compute_jacobians(in_vecs, out_vecs)
            real_log_dets = tf.linalg.slogdet(jacobians)[1]
            _randomized_init(sess)
            real_log_dets, log_dets = sess.run([real_log_dets, log_dets])
            assert log_dets.shape == (inputs.shape[0],)
            assert not np.isnan(log_dets).any()
            assert not np.isnan(real_log_dets).any()
            assert np.allclose(real_log_dets, log_dets, atol=1e-4, rtol=1e-4)


def _compute_jacobians(in_vecs, out_vecs):
    num_dims = in_vecs.get_shape()[-1].value
    res = []
    for i in range(in_vecs.get_shape()[0].value):
        rows = []
        for comp in range(num_dims):
            rows.append(tf.gradients(out_vecs[i, comp], in_vecs)[0][i])
        res.append(tf.stack(rows, axis=0))
    return tf.stack(res, axis=0)
"""

def _randomized_init(sess):
    sess.run(tf.global_variables_initializer())
    for variable in tf.trainable_variables():
        shape = [x.value for x in variable.get_shape()]
        val = tf.glorot_uniform_initializer(dtype=tf.float64)(shape)
        sess.run(tf.assign(variable, val))

test_inverses()
