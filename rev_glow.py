import tensorflow as tf
from rev_layer import Actnorm, Invconv, Coupling, ImageProcessing, Squeeze, FilterLatents, Network
from rev_obj import log_likelihood_and_grad, log_likelihood
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
    ls.append(ImageProcessing(hps.alpha))
    l_names.append('img_proc')
    for i in range(hps.n_levels):
        ls.append(Squeeze())
        l_names.append('block%d/squeeze' % i)
        for j in range(hps.depth):
            name = 'block%d/flow%d/' % (i,j)
            an = Actnorm()
            inv = Invconv()
            coup = Coupling()
            flow = [an, inv, coup]
            flow_names = [name + 'actnorm', name + 'invconv', name + 'coupling']
            ls += flow
            l_names += flow_names
        if i != hps.n_levels - 1:
            ls.append(FilterLatents())
            l_names.append('block%d/latent_filtering' % i)

    assert len(ls) == len(l_names)
    return ls, l_names

def model_dbg(hps):
    ls = [Squeeze(), Actnorm(), Invconv(), Coupling(), Actnorm(), Invconv(), Coupling()]
    l_names = ['squeeze', 'an1','inv1','coup1','an2','inv2','coup2']
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
    parser.add_argument("--width", type=int, default=128, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8, help="Depth of network")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--n_bits_x", type=int, default=5, help="Number of bits of x")

    # Finetuning arguments
    parser.add_argument("--finetune", type=int, default=0, help="if 0, then train generaitve, 1 then finetune")
    parser.add_argument("--clf_type", type=str, default="unwrap")
    parser.add_argument('--alpha', type=float, default=1e-6, help='alpha for preprocessing')

    args = parser.parse_args()
    args.n_bins_x = 2.**args.n_bits_x
    assert args.finetune in (0, 1)
    assert args.clf_type in ("unwrap", "pool")

    sess = tf.Session()

    dataset_fn = {'svhn': utils.SVHNDataset, 'cifar10': utils.CIFAR10Dataset, 'mnist': utils.MNISTDataset}[args.dataset]
    dataset = dataset_fn(
        args.batch_size,
        init_size=args.init_batch_size, n_labels=args.num_labels, n_valid=args.num_valid, n_bits_x=args.n_bits_x
    )

    # unpack labeled examples
    x, y = tf.cast(dataset.x, tf.float32), tf.to_int64(dataset.y)
    layers, layer_names = model_dbg(args)
    m = Network(layers, layer_names)
    _, z_init, _ = m.forward(x, reuse=False)
    _, z, logdets = m.forward(x, reuse=True)
    # run dataset.use_valid before computing validation loss

    logpx, grads = log_likelihood_and_grad(m, x, var_list=tf.trainable_variables())
    val_loss = log_likelihood(m, x)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    lr = tf.placeholder(tf.float32, [])
    with tf.control_dependencies(update_ops):
        optim = tf.train.AdamOptimizer(lr)
        opt = optim.apply_gradients(grads)

    sess.run(tf.global_variables_initializer())
    sess.run(dataset.use_init)
    print('running initialization...')
    sess.run(z_init)
    
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
                    _logpx, _ = sess.run([logpx, opt], feed_dict={lr: epoch_lr})
                    print(cur_iter, _logpx)
                else:
                    _ = sess.run(opt, feed_dict={lr: epoch_lr})

                cur_iter += 1
            except tf.errors.OutOfRangeError:
                print("Completed epoch {} in {}".format(epoch, time.time() - t_start))
                break
