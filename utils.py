import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io
import cv2
import os
import pickle
from collections import defaultdict
import math


def old_preprocess(x, dtype, n_bits_x=None, rand=True):
    x = tf.cast(x, dtype)
    if n_bits_x < 8:
        x = tf.floor(x / 2 ** (8 - n_bits_x))
    n_bins = 2. ** n_bits_x
    # add [0, 1] random noise
    if rand:
        x = x + tf.random_uniform(tf.shape(x), 0., 1., dtype=x.dtype)
    else:
        x = x + .5
    x = x / n_bins - .5
    return x


def old_postprocess(x, n_bits_x=None):
    n_bins = 2. ** n_bits_x
    x = tf.floor((x + .5) * n_bins) * (2 ** (8 - n_bits_x))
    return tf.cast(tf.clip_by_value(x, 0, 255), 'uint8')


# real NVP pre/post processing
def pp_sigmoid(y, alpha):
    x = (tf.sigmoid(y) - alpha) / (1 - 2 * alpha)
    return x


def pp_logits(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    y = tf.log(s) - tf.log(1 - s)
    ldgrad = logdetgrad(x, alpha)
    return y, ldgrad


def logdetgrad(x, alpha):
    s = alpha + (1 - 2 * alpha) * x
    ldg = -tf.log(s - s * s) + math.log(1 - 2 * alpha)
    ldg = tf.reshape(ldg, [tf.shape(x)[0], -1])
    ldg = tf.reduce_sum(ldg, axis=-1)
    return ldg


def preprocess(x, alpha=1e-6):
    noisy = add_noise(x)
    return pp_logits(noisy, alpha)


def postprocess(y, alpha=1e-6):
    return pp_sigmoid(y, alpha)


def add_noise(x):
    x = tf.cast(x, tf.float32)
    noise = tf.random_uniform(tf.shape(x))
    x = x + noise
    x = x / 256
    return x


def split_dataset(xs, ys, n_labels, seed=1234):
    data_dict = defaultdict(list)
    for x, y in zip(xs, ys):
        data_dict[y].append(x)
    np.random.seed(seed)
    xs_u = []
    xs_l = []
    ys_l = []
    ys_u = []
    n_class = len(data_dict.keys())
    assert n_labels % n_class == 0, "num class must divide num labels"
    n_per_class = n_labels // n_class
    for y in data_dict.keys():
        cur_xs = data_dict[y]
        np.random.shuffle(cur_xs)
        cur_xs_l = cur_xs[:n_per_class]
        cur_xs_u = cur_xs[n_per_class:]
        xs_u.extend(cur_xs_u)
        xs_l.extend(cur_xs_l)
        ys_l.extend([y] * n_per_class)
        ys_u.extend([y] * len(cur_xs_u))
    xs_l = np.array(xs_l, dtype=xs.dtype)
    xs_u = np.array(xs_u, dtype=xs.dtype)
    ys_l = np.array(ys_l, dtype=ys.dtype)
    ys_u = np.array(ys_u, dtype=ys.dtype)
    return xs_u, ys_u, xs_l, ys_l


def create_dataset(x, y, batch_size, shuffle=True,
                   repeat=False, ind_aug=None, batch_aug=None):
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    if ind_aug is not None:
        ds_x = ds_x.map(ind_aug)

    if y is None:
        ds = ds_x
    else:
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds_x, ds_y))

    if shuffle:
        ds = ds.shuffle(len(x))

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    if batch_aug is not None:
        ds = ds.map(batch_aug)

    return ds


class Dataset(object):
    def batch_aug_train(self, x, y):
        return x, y

    def batch_aug_train_unsup(self, x):
        return x

    def batch_aug_test(self, x, y):
        return x, y

    def __init__(self, trainx, trainy, testx, testy, batch_size,
                 valx=None, valy=None,
                 train_aug=lambda x: x, test_aug=lambda x: x,
                 init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        self.n_bits_x = n_bits_x

        # create validation set if requested
        if n_valid is not None:
            # ensure that we are not given a validation set if we are asked to make one
            assert valx is None and valy is None, "two validation sets"
            trainx, trainy, valx, valy = split_dataset(trainx, trainy, n_valid)

        # store original trainx so we can use it go generate a large init batch
        # since no labels are used in initialization, this is ok
        x_orig, y_orig = trainx, trainy

        # restrict training set if asked
        if n_labels is not None:
            _, _, trainx, trainy = split_dataset(trainx, trainy, n_labels)

        self.n_train_l = len(trainx)
        train = create_dataset(trainx, trainy, batch_size,
                               ind_aug=train_aug, batch_aug=self.batch_aug_train)
        test = create_dataset(testx, testy, batch_size,
                              shuffle=False, ind_aug=test_aug,
                              batch_aug=self.batch_aug_test)
        iterator = tf.data.Iterator.from_structure(train.output_types,
                                                   train.output_shapes)
        self.x, self.y = iterator.get_next()
        self.use_train = iterator.make_initializer(train)
        self.use_test = iterator.make_initializer(test)
        if valx is not None:
            valid = create_dataset(valx, valy, batch_size,
                                   shuffle=False, ind_aug=test_aug,
                                   batch_aug=self.batch_aug_test)
            self.use_valid = iterator.make_initializer(valid)
        else:
            self.use_valid = None

        if init_size is not None:
            init = create_dataset(x_orig, y_orig, init_size,
                                  ind_aug=train_aug, batch_aug=self.batch_aug_train)
            self.use_init = iterator.make_initializer(init)


class MNISTDataset(Dataset):
    def __init__(self, batch_size, init_size=None,
                 n_labels=None, n_valid=None, n_bits_x=None):
        assert n_valid is None
        self.n_class = 10

        def train_aug(x):
            x = tf.image.resize_image_with_crop_or_pad(x, 36, 36)
            x = tf.random_crop(x, [32, 32, 1])
            return x

        def test_aug(x):
            return tf.image.resize_image_with_crop_or_pad(x, 32, 32)

        cvt = lambda x: ((255 * x).astype(np.uint8)).reshape([-1, 28, 28, 1])
        mnist = input_data.read_data_sets("MNIST_data")
        trainx = cvt(mnist.train.images)
        valx = cvt(mnist.validation.images)
        testx = cvt(mnist.test.images)

        super(MNISTDataset, self).__init__(
            trainx, mnist.train.labels,
            testx, mnist.test.labels,
            batch_size,
            valx=valx, valy=mnist.validation.labels,
            train_aug=train_aug, test_aug=test_aug,
            init_size=init_size, n_labels=n_labels, n_bits_x=n_bits_x
        )


class CIFAR10Dataset(Dataset):

    def __init__(self, batch_size, init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        self.n_class = 10

        def load(f):
            with open(f, 'rb') as f:
                stuff = pickle.load(f, encoding="bytes")
                return stuff[b'data'], stuff[b'labels']

        dname = 'cifar-10-batches-py'
        tr_names = ['data_batch_1', 'data_batch_2',
                    'data_batch_3', 'data_batch_4', 'data_batch_5']
        tr_names = [os.path.join(dname, tr) for tr in tr_names]
        te_name = os.path.join(dname, "test_batch")
        train_data = [load(f) for f in tr_names]
        trainx = [td[0] for td in train_data]
        trainy = [td[1] for td in train_data]
        trainx = np.concatenate(trainx)
        trainy = np.concatenate(trainy)
        testx, testy = load(te_name)
        trainx = trainx.reshape([-1, 3, 32, 32])
        testx = testx.reshape([-1, 3, 32, 32])
        trainx = np.transpose(trainx, [0, 2, 3, 1])
        testx = np.transpose(testx, [0, 2, 3, 1])
        trainy = np.array(trainy, dtype=np.uint8)
        testy = np.array(testy, dtype=np.uint8)

        def train_aug(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(CIFAR10Dataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug, init_size=init_size,
            n_labels=n_labels, n_valid=n_valid, n_bits_x=n_bits_x
        )


class SVHNDataset(Dataset):
    def __init__(self, batch_size, init_size=None,
                 n_labels=None, n_valid=None, n_bits_x=None):
        self.n_class = 10
        train = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        trainx, trainy = train['X'], train['y'][:, 0] - 1
        trainx = trainx.transpose((3, 0, 1, 2))
        test = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        testx, testy = test['X'], test['y'][:, 0] - 1
        testx = testx.transpose((3, 0, 1, 2))

        def train_aug(x):
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(SVHNDataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug, init_size=init_size,
            n_labels=n_labels, n_valid=n_valid, n_bits_x=n_bits_x
        )


def gs(x):
    return x.get_shape().as_list()


def normal_logpdf(x, mu, logvar):
    logp = -.5 * (np.log(2. * np.pi) + logvar + ((x - mu) ** 2) / tf.exp(logvar))
    return logp


def mog_sample(mus, shape, stddev=1.):
    n_class = gs(mus)[0]
    inds = tf.one_hot(tf.argmax(tf.random_uniform([shape[0], n_class]), axis=1), n_class)
    chosen_mus = tf.reduce_sum(mus[None, :, :, :, :] * inds[:, :, None, None, None], axis=1)
    samples = tf.random_normal(shape, stddev=stddev) + chosen_mus
    return samples
