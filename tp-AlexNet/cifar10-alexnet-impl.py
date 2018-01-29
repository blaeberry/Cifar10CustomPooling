#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar10-resnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import os


from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset

import tensorflow as tf

"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.

To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH_SIZE = 128
NUM_UNITS = None


class Model(ModelDesc):

    def __init__(self, n):
        super(Model, self).__init__()
        self.n = n

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        #image = tf.transpose(image, [0, 3, 1, 2])

        with argscope([Conv2D, FullyConnected], nl=tf.nn.relu):
            l = Conv2D('conv1', image, out_channel=64, kernel_shape=5)
            l = tf.nn.lrn(l, 3, bias=1.0, alpha=2e-5, beta=0.75, name='norm1')
            l = custom_pooling2d(l, 'pool1', 'VALID')
    #    l = Dropout(l)
    #    l = tf.nn.dropout(l, 0.5)

            l = Conv2D('conv2', l, out_channel=64, kernel_shape=5)
            l = tf.nn.lrn(l, 3, bias=1.0, alpha=2e-5, beta=0.75, name='norm2')
            l = custom_pooling2d(l, 'pool2', 'VALID')
     #   l = Dropout(l)
     #   l = tf.nn.dropout(l, 0.5)
            
            l = Conv2D('conv3', l, out_channel=128, kernel_shape=3)

        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        #wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
        #                                  480000, 0.2, True)
        wd_cost = tf.multiply(0.0004, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def custom_pooling2d(inputs, var_scope, padding, ps=3, strides=[1, 2, 2, 1]):
    max_inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=ps, strides=strides[1], padding=padding)

    avg_inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=ps, strides=strides[1], padding=padding)

    weights_shape = (1)
    with tf.variable_scope(var_scope):
        max_weights = tf.tanh(tf.get_variable("max_weights", weights_shape, initializer=tf.constant_initializer(0.5)))
        avg_weights = tf.tanh(tf.get_variable("avg_weights", weights_shape, initializer=tf.constant_initializer(0.5)))
    return (tf.multiply(max_inputs, max_weights) + tf.multiply(avg_inputs, avg_weights))

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(300, 0.001), (50, 0.0001), (300, 0.00001)])
        ],
        max_epoch=400,
        session_init=SaverRestore(args.load) if args.load else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
