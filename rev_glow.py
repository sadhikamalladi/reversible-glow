import tensorflow as tf
from rev_layer import Actnorm, SkewInvconv, Coupling
from rev_layer import Squeeze, FilterLatents, Network
from rev_obj import ll
import argparse
import os
import json
import shutil
import utils
import time
import numpy as np

"""
O(1) implementation of glow model. Default params are SOTA
for invertible flow models on MNIST/CIFAR/ImageNet.
"""


def model(hps):
    """
    Construct a model per the specifications in the hps parameters.
    hps: args parsed from command line to detail depth/width/scales of network
    Returns the network's layer objects as a list and layer names (scopes)
    """
    ls = []
    l_names = []
    for i in range(hps.n_levels):
        ls.append(Squeeze())
        block_name = f'block{i}/'
        l_names.append(block_name+'squeeze')
        if hps.shared_coupling:
            coup = Coupling()
            coup_name = block_name+'coupling'
        for j in range(hps.depth):
            flow_name = block_name + f'flow{j}/'
            an = Actnorm()
            inv = SkewInvconv()
            if not hps.shared_coupling:
                coup = Coupling(dim=hps.width)
                coup_name = flow_name + 'coupling'
            flow = [an, inv, coup]
            flow_names = [flow_name + 'actnorm',
                          flow_name + 'invconv',
                          coup_name]
            ls += flow
            l_names += flow_names
        if i != hps.n_levels - 1:
            ls.append(FilterLatents())
            l_names.append('block%d/latent_filtering' % i)

    assert len(ls) == len(l_names)
    return ls, l_names


def create_experiment_directory(args):
    """
    Creates a directory to copy the running version of the code to,
    the results, and the parameters. Assumes directory doesn't exist.
    """
    # write params
    with open(os.path.join(args.train_dir, "params.txt"), 'w') as f:
        f.write(json.dumps(args.__dict__))
    # copy code
    code_dest_dir = os.path.join(args.train_dir, "code")
    os.mkdir(code_dest_dir)
    code_dir = os.path.dirname(__file__)
    code_dir = '.' if code_dir == '' else code_dir
    python_files = [os.path.join(code_dir, fn)
                    for fn in os.listdir(code_dir) if fn.endswith(".py")]
    for pyf in python_files:
        print(pyf, code_dest_dir)
        shutil.copy2(pyf, code_dest_dir)
    os.mkdir(os.path.join(args.train_dir, "best"))
    os.mkdir(os.path.join(args.train_dir, "backup"))


def get_lr(epoch, args):
    """
    Computes the learning rate for a given epoch based on the hyperparameters
    provided in args. args should specify the learning rate, # warmup epochs,
    decay factor, epochs between decays, and
    if the learning rate should be scaled or not.

    Returns the learning rate for a given epoch, subject to constraints in args
    """
    if epoch < args.epoch_warmup + 1:
        epoch_lr = args.lr * (epoch + 1) / args.epoch_warmup
    else:
        epoch_lr = args.lr
    # get decayed lr
    if args.lr_scalemode == 0:
        return epoch_lr

    lr_scale = args.decay_factor ** (epoch // args.epochs_decay)
    epoch_lr *= lr_scale
    return epoch_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/tmp/train",
                        help="Directory to save tensorboard data to")
    parser.add_argument("--dataset", type=str,
                        default='mnist', help="Problem (mnist/cifar10/svhn)")
    parser.add_argument("--num_valid", type=int, default=None,
                        help="Validation size for svhn/cifar")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="Number of labeled examples to use")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path for load saved checkpoint from")
    parser.add_argument("--log_iters", type=int, default=100,
                        help="iters per each print and summary")

    # Optimization hyperparams:
    parser.add_argument("--epochs", type=int, default=100000,
                        help="Train epoch size")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")
    parser.add_argument("--init_batch_size", type=int, default=1024,
                        help="batch size for init")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0,
                        help="type of learning rate scaling. 0=none, 1=step.")
    parser.add_argument("--epochs_warmup", type=int, default=10,
                        help="warmup epochs")
    parser.add_argument("--epochs_valid", type=int, default=1,
                        help="epochs between valid")
    parser.add_argument("--epochs_backup", type=int, default=10,
                        help="epochs between backup saving")
    parser.add_argument("--epochs_decay", type=int, default=250,
                        help="epochs between lr decay")
    parser.add_argument("--decay_factor", type=float, default=.1,
                        help="multiplier on learning rate")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=512,
                        help="width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="depth of network")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="number of levels")
    parser.add_argument("--n_bits_x", type=int, default=3,
                        help="number of bits of x")
    parser.add_argument('--precision', type=str, default='float32',
                        help='(float32, float64) precision for the network')

    # Testing hyperparams:
    parser.add_argument('--test_grad_divergence', action='store_true',
                        default=False,
                        help='print diff between real and computed grads')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='run a few train ops and save profiling info')
    parser.add_argument('--shared_coupling', action='store_true',
                        default=False,
                        help='shares params across coupling layers in block')

    args = parser.parse_args()

    # setup experiment directory, copy current code, save params
    assert not os.path.exists(args.train_dir), "This directory already exists."
    train_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "train"))
    test_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "test"))
    valid_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "valid"))
    create_experiment_directory(args)

    sess = tf.Session()

    # pick the dataset per args
    dataset_fn = {'svhn': utils.SVHNDataset,
                  'cifar10': utils.CIFAR10Dataset,
                  'mnist': utils.MNISTDataset}[args.dataset]
    dataset = dataset_fn(
        args.batch_size,
        init_size=args.init_batch_size,
        n_labels=args.num_labels,
        n_valid=args.num_valid,
        n_bits_x=args.n_bits_x
    )

    # get data from iterator
    x = dataset.x
    x = utils.old_preprocess(x, n_bits_x=args.n_bits_x, dtype=args.precision)

    # build the model and initialize variables
    layers, layer_names = model(args)
    m = Network(layers, layer_names, shared=args.shared_coupling)
    _, z_init, _ = m.forward(x, reuse=False, name='net')
    _, z, logdet = m.forward(x, reuse=True, name='net')
    x_recons = m.inverse(None, z, reuse=True, name='net')
    z_samp = [tf.cast(tf.random_normal(tf.shape(_z)),
                      args.precision) for _z in z]
    x_samp = m.inverse(None, z_samp, reuse=True, name='net')

    # objectives computation
    logpx = ll(z, logdet)
    grads = m.gradients(None, z, logdet, -logpx, name='net')
    variables = [g[1] for g in grads]
    real_gradients = tf.gradients(-logpx, variables)

    # optimizer initialization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lr = tf.placeholder(tf.float32, [])
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(lr)
        opt = optim.apply_gradients(grads)

    # summaries
    x_recons = utils.old_postprocess(x_recons, n_bits_x=args.n_bits_x)
    x_samp = utils.old_postprocess(x_samp, n_bits_x=args.n_bits_x)
    recons_error = tf.reduce_mean(tf.square(
        tf.to_float(utils.old_postprocess(x, n_bits_x=args.n_bits_x) -
                    x_recons)))
    tf.summary.image("x_sample", x_samp)
    tf.summary.image("x_recons", x_recons)
    tf.summary.image("x", x)
    tf.summary.scalar("recons", recons_error)
    tf.summary.scalar("lr", lr)
    loss_summary = tf.summary.scalar("loss", logpx)
    m_logdet = tf.reduce_mean(logdet)
    logdet_summary = tf.summary.scalar("logdet", m_logdet)
    logpx_logit_summary = tf.summary.scalar('logpx_logit', logpx)
    # the summary op to call for the training data
    train_summary = tf.summary.merge_all()
    test_summaries = [loss_summary, logdet_summary]
    # get the values we need to call to get mean stat
    test_values = [logpx, m_logdet]
    test_value_names = ['loss', 'logdet']
    test_summary = tf.summary.merge(test_summaries)

    sess.run(tf.global_variables_initializer())
    sess.run(dataset.use_init)
    print('running initialization...')
    sess.run(z_init)

    # evaluation code block
    def evaluate(init_op, writer, name):
        sess.run(init_op)
        summary_values = []
        while True:
            try:
                summary_values.append(sess.run(test_values))
            except tf.errors.OutOfRangeError:
                summary_values = np.array(summary_values).mean(axis=0)
                print("{}: ...".format(name))
                for val_name, val_val in zip(test_value_names, summary_values):
                    print("    {}: {}".format(val_name, val_val))
                fd = {node: val for node, val in
                      zip(test_values, summary_values)}
                sstr = sess.run(test_summary, feed_dict=fd)
                writer.add_summary(sstr, cur_iter)
                # return loss to determine best model
                return summary_values[0]

    # profiling run
    if args.profile:
        from tensorflow.python.client import timeline
        # few iterations of SGD
        for i in range(3):
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            sess.run([opt, logpx],
                     {lr: 0.001},
                     options=run_options, run_metadata=run_metadata)
        # write timeline information to timeline.json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
    else:
        # training loop
        cur_iter = 0
        best_valid = np.inf
        for epoch in range(args.epochs):
            sess.run(dataset.use_train)
            t_start = time.time()
            # get lr for this epoch
            epoch_lr = get_lr(epoch, args)
            while True:
                try:
                    # log infomration
                    if cur_iter % args.log_iters == 0:
                        # main op to run training + logging
                        _re, _l, _, sstr = sess.run([recons_error,
                                                     logpx,
                                                     opt,
                                                     train_summary],
                                                    feed_dict={lr: epoch_lr})
                        train_writer.add_summary(sstr, cur_iter)
                        print(cur_iter, _l, _re)

                        # compute max difference between true gradients and
                        # backward computed ones
                        if args.test_grad_divergence:
                            _g, _b = sess.run([grads, real_gradients])
                            _grads = [__g[0] for __g in _g]
                            _grads = np.array(_grads)
                            _b = np.array(_b)
                            diff = np.abs(_b - _grads)
                            diff = [np.max(d) for d in diff]
                            max_diff = np.max(diff)
                            max_inds = np.where(diff == max_diff)
                            max_names = [variables[i] for i in max_inds[0]]
                            print(f'{max_diff}: div for tensors {max_names})')
                    else:
                        _ = sess.run(opt, feed_dict={lr: epoch_lr})

                # epoch ends when we're out of data
                except tf.errors.OutOfRangeError:
                    print(cur_iter)
                    print("Completed epoch {} in {}".format(epoch,
                                                            time.time() -
                                                            t_start))
                    if epoch % args.epochs_valid == 0:
                        evaluate(dataset.use_test, test_writer, "Test")
                        # if we have a validation set, get validation accuracy
                        if dataset.use_valid is not None:
                            valid_loss = evaluate(dataset.use_valid,
                                                  valid_writer,
                                                  "Valid")
                            print(valid_loss)
                    break

                cur_iter += 1
