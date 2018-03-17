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
        image = tf.transpose(image, [0, 3, 1, 2])

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
            #     stride1 = 2
            else:
                out_channel = in_channel
            
            if first:
                out_channel = out_channel * 8

            stride1 = 1

            with tf.variable_scope(name):
                if increase_dim:
                    l = custom_pooling2d(l, 'pools', padding = 'VALID')
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])
                if first:
                    l = tf.pad(l, [[0, 0], [int(in_channel*3.5), int(in_channel*3.5)], [0, 0], [0, 0]])

                l = c2 + l
                return l

        with argscope([Conv2D, AvgPooling, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
            l = Conv2D('conv0', image, 16, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

        logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)
        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0005, get_global_step_var(),
                                          10000, 0.75, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.01, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

# def custom_pooling2d(inputs, var_scope, padding, strides = [1, 2, 2], data_format='NCHW'):
#     max_inputs = MaxPooling('pool_max', inputs, 2)

#     #we want to do 1 channel at a time, so we're turning channels into a dim and saying there is 1 channel
#     cinputs = tf.expand_dims(inputs, -1)
#     weights_shape = (1, 2, 2, 1, 1)
#     #print(cinputs.get_shape())
#     with tf.variable_scope(var_scope):
#         max_weight = tf.get_variable("max_weights", (1), initializer=tf.constant_initializer(0.5))
#         pcon0 = tf.get_variable("pcon0", weights_shape, initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
#    #     pcon1 = tf.get_variable("pcon1", weights_shape, initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
#    #     pcon2 = tf.get_variable("pcon2", weights_shape, initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
#    #     pcon3 = tf.get_variable("pcon3", weights_shape, initializer=tf.contrib.layers.variance_scaling_initializer(2.0))
#    # convolves = tf.add(tf.add(tf.nn.convolution(cinputs, pcon0, 'SAME', strides = strides),
#    #     tf.nn.convolution(cinputs, pcon1, 'SAME', strides = strides)), tf.add(tf.nn.convolution(cinputs, pcon2, 'SAME', strides = strides),
#    #     tf.nn.convolution(cinputs, pcon3, 'SAME', strides = strides)))
#     convolves = tf.nn.convolution(cinputs, pcon0, 'VALID', strides = strides)
#     convolves = tf.squeeze(convolves, axis = -1, name = "convolves")
#     #print(convolves.get_shape())
#     outputs = tf.add(tf.multiply(max_inputs, max_weight), convolves, name = "output")
#     return outputs

def custom_pooling2d(inputs, var_scope, padding, nf = 4, strides = [1, 1, 2, 2], data_format='NCHW'):
    with tf.variable_scope(var_scope):
        l = BatchNorm('bn', inputs)
        l = tf.nn.relu(l)
        max_inputs = MaxPooling('pool_max', l, 2)
        in_shape = l.get_shape().as_list()

        weights_shape = (2, 2, 1, 1)
        p = tf.zeros([tf.shape(l)[0], in_shape[1], int(in_shape[2] // 2), int(in_shape[3]//2)])

        mw = tf.get_variable("mw", (1), initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
        for k in range(nf):
            pw = tf.get_variable('pw{}'.format(k), (1), initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            pcon = tf.get_variable('pcon{}'.format(k), weights_shape, 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            pcon = tf.tile(pcon, [1, 1, in_shape[1], 1], name='tiled')
            print(in_shape)
            print(p.get_shape().as_list())
            print(pcon.get_shape().as_list())
            print(tf.nn.depthwise_conv2d(l, pcon, strides, padding, data_format=data_format, name='depthwise').get_shape().as_list())
            p = p + (pw)*tf.nn.depthwise_conv2d(l, pcon, strides, padding, data_format=data_format, name='depthwise')
        p = tf.add((mw)*max_inputs, p, name = "outputs")    
    return p


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
                        type=int, default=5)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()
    NUM_UNITS = args.num_units

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir(action='k')

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(n=NUM_UNITS),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(max_to_keep = 1, keep_checkpoint_every_n_hours = 10000),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (60, 0.01), (120, 0.001)])#, ScalarPrinter(whitelist=[".*pools.*"])
        ],
        max_epoch=150,
        session_init=SaverRestore(logger.get_logger_dir() + '/checkpoint') if tf.gfile.Exists(logger.get_logger_dir() + '/checkpoint') else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
