import tensorflow as tf
from rev_layer import Actnorm, SkewInvconv, Coupling, ImageProcessing, Squeeze, FilterLatents, Network
from rev_obj import ll
import argparse
import os
import json
import shutil
import utils
import time
import numpy as np

def model(hps):
    ls = []
    l_names = []
    for i in range(hps.n_levels):
        ls.append(Squeeze())
        block_name = f'block{i}/'
        l_names.append(block_name+'squeeze')
        for j in range(hps.depth):
            flow_name = block_name + f'flow{j}/'
            an = Actnorm()
            inv = SkewInvconv()
            coup = Coupling()
            flow = [an, inv, coup]
            flow_names = [flow_name + 'actnorm', flow_name + 'invconv', flow_name + 'coupling']
            ls += flow
            l_names += flow_names
        if i != hps.n_levels - 1:
            ls.append(FilterLatents())
            l_names.append('block%d/latent_filtering' % i)

    assert len(ls) == len(l_names)
    return ls, l_names

def create_experiment_directory(args):
    # write params
    with open(os.path.join(args.train_dir, "params.txt"), 'w') as f:
        f.write(json.dumps(args.__dict__))
    # copy code
    code_dest_dir = os.path.join(args.train_dir, "code")
    os.mkdir(code_dest_dir)
    code_dir = os.path.dirname(__file__)
    code_dir = '.' if code_dir == '' else code_dir
    python_files = [os.path.join(code_dir, fn) for fn in os.listdir(code_dir) if fn.endswith(".py")]
    for pyf in python_files:
        print(pyf, code_dest_dir)
        shutil.copy2(pyf, code_dest_dir)
    os.mkdir(os.path.join(args.train_dir, "best"))
    os.mkdir(os.path.join(args.train_dir, "backup"))


def get_lr(epoch, args):
    epoch_lr = (args.lr * (epoch + 1) / args.epochs_warmup) if epoch < args.epochs_warmup + 1 else args.lr
    # get decayed lr
    if args.lr_scalemode == 0:
        return epoch_lr
    else:
        lr_scale = args.decay_factor ** (epoch // args.epochs_decay)
        epoch_lr *= lr_scale
        return epoch_lr


def test_divergence(a, b, tol):
    eqs = [np.isclose(a[i], b[i], atol=tol) for i in range(len(a))]
    equs = [e.all() for e in eqs]
    equs = np.array(equs)
    return equs.all()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/tmp/train")
    parser.add_argument("--dataset", type=str, default='mnist', help="Problem (mnist/cifar10/svhn)")
    parser.add_argument("--num_valid", type=int, default=None,
                        help="The number of examples to place into the validaiton set (only for svhn and cifar10)")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labeled examples to use")
    parser.add_argument("--load_path", type=str, default=None, help="Path for load saved checkpoint from")
    parser.add_argument("--log_iters", type=int, default=100, help="iters per each print and summary")

    # Optimization hyperparams:
    parser.add_argument("--epochs", type=int, default=100000, help="Train epoch size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--init_batch_size", type=int, default=1024, help="batch size for init")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0, help="Type of learning rate scaling. 0=none, 1=step.")
    parser.add_argument("--epochs_warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--epochs_valid", type=int, default=1, help="Epochs between valid")
    parser.add_argument("--epochs_backup", type=int, default=10, help="Epochs between backup saving")
    parser.add_argument("--epochs_decay", type=int, default=250, help="Epochs between lr decay")
    parser.add_argument("--decay_factor", type=float, default=.1, help="Multiplier on learning rate")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=512, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32, help="Depth of network")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--n_bits_x", type=int, default=3, help="Number of bits of x")
    parser.add_argument("--share_parameters", default=False, action='store_true', help='when on, shares parameters aross coupling layers in steps of a block')
    parser.add_argument('--time_input', default=False, action='store_true', help='when on, provides time as input to coupling layer (only use w shared params')

    # Finetuning arguments
    parser.add_argument("--finetune", type=int, default=0, help="if 0, then train generaitve, 1 then finetune")
    parser.add_argument("--clf_type", type=str, default="unwrap")
    parser.add_argument('--alpha', type=float, default=1e-6, help='alpha for preprocessing')
    parser.add_argument('--precision', type=str, default='float32', help='float## (float32, float64) used as precision for the whole network')
    parser.add_argument('--test_grad_divergence', action='store_true', default=False)

    args = parser.parse_args()
    args.n_bins_x = 2.**args.n_bits_x
    assert args.finetune in (0, 1)
    assert args.clf_type in ("unwrap", "pool")
    assert not os.path.exists(args.train_dir), "This directory already exists..."
    train_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "train"))
    test_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "test"))
    valid_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "valid"))
    # setup experiment directory, copy current version of the code, save parameters
    create_experiment_directory(args)
    
    sess = tf.Session()

    dataset_fn = {'svhn': utils.SVHNDataset, 'cifar10': utils.CIFAR10Dataset, 'mnist': utils.MNISTDataset}[args.dataset]
    dataset = dataset_fn(
        args.batch_size,
        init_size=args.init_batch_size, n_labels=args.num_labels, n_valid=args.num_valid, n_bits_x=args.n_bits_x
    )

    # get data from iterator
    x = dataset.x
    x = utils.old_preprocess(x, n_bits_x=args.n_bits_x, dtype=args.precision)
    
    layers, layer_names = model(args)
    m = Network(layers, layer_names)
    _, z_init, _ = m.forward(x, reuse=False, name='net')
    _, z, logdet = m.forward(x, reuse=True, name='net')
    x_recons = m.inverse(None, z, reuse=True, name='net')
    z_samp = [tf.cast(tf.random_normal(tf.shape(_z)), args.precision) for _z in z]
    x_samp = m.inverse(None, z_samp, reuse=True, name='net')

    # objectives computation
    logpx = ll(z, logdet)
    grads = m.gradients(None, z, logdet, -logpx, name='net')
    variables = [g[1] for g in grads]
    real_gradients = tf.gradients(-logpx, variables)
    """
    noisy_grads = []
    for g in grads:
        noise = tf.random_uniform(g[0].shape, minval=1e-4, maxval=1e-3)
        noisy_g = g[0] + noise
        noisy_grads.append((noisy_g, g[1]))
    """

    # optimizer initialization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lr = tf.placeholder(tf.float32, [])
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(lr)
        opt = optim.apply_gradients(grads)

    # summaries
    x_recons = utils.old_postprocess(x_recons, n_bits_x=args.n_bits_x)
    x_samp = utils.old_postprocess(x_samp, n_bits_x=args.n_bits_x)
    recons_error = tf.reduce_mean(tf.square(tf.to_float(utils.old_postprocess(x, n_bits_x=args.n_bits_x) - x_recons)))
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
                fd = {node: val for node, val in zip(test_values, summary_values)}
                sstr = sess.run(test_summary, feed_dict=fd)
                writer.add_summary(sstr, cur_iter)
                # return loss to determine best model
                return summary_values[0]
    
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
                if cur_iter % args.log_iters == 0:
                    
                    _re, _l, _, sstr = sess.run([recons_error, logpx, opt, train_summary], feed_dict={lr: epoch_lr})
                    train_writer.add_summary(sstr, cur_iter)
                    print(cur_iter, _l, _re)
                    if args.test_grad_divergence:
                        _g, _b = sess.run([grads, real_gradients])
                        _grads = [__g[0] for __g in _g]
                        _grads = np.array(_grads)
                        _b = np.array(_b)
                        diff = np.abs(_b - _grads)
                        diff = [np.max(d) for d in diff]
                        max_diff = np.max(diff)
                        max_inds = np.where(diff==max_diff)
                        max_names = [variables[i] for i in max_inds[0]]
                        print(f'{max_diff}: div for tensors {max_names})')
                else:
                    _ = sess.run(opt, feed_dict={lr: epoch_lr})
        
            except tf.errors.OutOfRangeError:
                print(cur_iter)
                print("Completed epoch {} in {}".format(epoch, time.time() - t_start))
                if epoch % args.epochs_valid == 0:
                    evaluate(dataset.use_test, test_writer, "Test")
                    # if we have a validation set, get validation accuracy
                    if dataset.use_valid is not None:
                        valid_loss = evaluate(dataset.use_valid, valid_writer, "Valid")
                        print(valid_loss)
                break
            

            cur_iter += 1


