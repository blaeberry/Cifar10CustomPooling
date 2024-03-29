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
NUM_GROUPS = 2

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

        def grouped_conv(name, x, kernel, stride, num_groups, padding='SAME', nl=tf.identity):
            with tf.variable_scope(name) as scope:
                sz = x.get_shape()[1].value // num_groups
                szk = kernel.get_shape()[-1].value // num_groups
                conv_side_layers = [
                    tf.nn.conv2d(x[:, i * sz:i * sz + sz, :, :], kernel[:,:,:, i * szk:i * szk + szk], [1,1,stride,stride],
                        padding=padding, data_format='NCHW', name=name + "_" + str(i)) for i in range(num_groups)]
                ret = tf.concat(conv_side_layers, axis=1)
                ret = nl(ret, name = name)
                ret.variables = VariableHolder(W=kernel)
                return ret

        def channel_shuffle(name, x, num_groups, nchw = True):
            with tf.variable_scope(name) as scope:
                if nchw:
                    x = tf.transpose(x, [0, 2, 3, 1]) #nchw -> nhwc
                n, h, w, c = x.shape.as_list()
                x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
                x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
                output = tf.reshape(x_transposed, [-1, h, w, c])
                if nchw:
                    output = tf.transpose(x, [0, 3, 1, 2]) #nhwc -> nchw
                return output

        # def conv(name, l, kernel, stride, nl = tf.identity):
        #     conv = tf.nn.conv2d(l, kernel, [1,1,stride,stride], padding = 'SAME', data_format='NCHW')
        #     ret = nl(conv, name = name)
        #     ret.variables = VariableHolder(W=kernel)
        #     return ret

        def expand_filters(num_filters, in_channels, num_groups, kernel_size=3):
            #256 in, g = 8, each=32
            each_g = in_channels//num_groups
            filters = tf.get_variable('filters', (kernel_size, kernel_size, num_filters), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            ch_prefs = tf.get_variable('ch_prefs', (each_g, num_filters), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            kernel = tf.expand_dims(filters, axis = 2) #w,h,1,num_filters
            kernel = kernel * ch_prefs #w,h,1,num_filters * each_g,num_filters = w,h,each_g,num_filters
            #returns the updated filters and the filters expanded across the depth for convolution
            return kernel

        def expand_filters_g(num_filters, in_channels, num_groups, kernel_size = 3):
            #3,3,in=256, out=512, num_group = 8
            #in_each = 256/8 = 32
            filters = tf.get_variable('filters', (kernel_size, kernel_size, num_filters), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            group_prefs = tf.get_variable('group_prefs', (num_groups, num_filters), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            kernel = tf.expand_dims(filters, axis = 2) #w,h,1,num_filters
            kernel = kernel * group_prefs #w,h,1,num_filters * in_channels,num_filters
            #returns the updated filters and the filters expanded across the depth for convolution
            return kernel

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            if first:
                out_channel = out_channel * 8

            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)

                kernel = expand_filters(out_channel, in_channel, NUM_GROUPS)
                with tf.variable_scope('first'):
                    c1 = grouped_conv('conv1', b1, kernel, stride1, NUM_GROUPS, nl=BNReLU)

                with tf.variable_scope('second'):
                    kernel = expand_filters(out_channel, out_channel, NUM_GROUPS)
                    c2 = grouped_conv('conv2', c1, kernel, 1, NUM_GROUPS)
                
                if first:
                    l = tf.pad(l, [[0, 0], [int(in_channel*3.5), int(in_channel*3.5)], [0, 0], [0, 0]])
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l

        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
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

    logger.auto_set_dir()

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
                                      [(1, 0.1), (60, 0.01), (120, 0.001)])
        ],
        max_epoch=150,
        session_init=SaverRestore(args.load) if args.load else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
