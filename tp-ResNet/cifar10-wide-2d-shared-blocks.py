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

        def conv(name, l, kernel, stride, nl = tf.identity):
            conv = tf.nn.conv2d(l, kernel, [1,1,stride,stride], padding = 'SAME', data_format='NCHW')
            ret = nl(conv, name = name)
            ret.variables = VariableHolder(W=kernel)
            return ret

        def expand_filters(in_channels, out_channels, x, reuse = True):
            num_used_filters = out_channels
            if x == -1:
                x = 0
                reuse = False
            i = x + 1
            num_total_filters = 512 #number of filters used in final ResNet block
            with tf.variable_scope('reused', reuse = False) as scope:
                if reuse:
                    scope.reuse_variables() 
                filters = tf.get_variable('filters', (3, 3, num_total_filters), 
                    initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            with tf.variable_scope('prefs.{}'.format(out_channels), reuse = False) as scope:
                ch_prefs = tf.get_variable('ch_prefs.{}'.format(i), (in_channels, num_used_filters), 
                    initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            kernel = filters[:, :, 0:(num_used_filters)]
            kernel = tf.expand_dims(kernel, axis = 2) #[3,3,1,num_used_filters]
            kernel = kernel * ch_prefs #[3,3,1,num_used_filters] * [in_channels,num_used_filters]
            #returns the selected filters expanded across the depth for convolution
            return kernel


        def residual(name, l, k, increase_dim=False, first=False):
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

            kernel1 = expand_filters(in_channel, out_channel, k)
            with tf.variable_scope('second'):
                kernel2 = expand_filters(out_channel, out_channel, k)
            with tf.variable_scope(name):
                b1 = l if first else BNReLU(l)

                with tf.variable_scope('first'):
                    c1 = conv('conv1', b1, kernel1, stride1, nl=BNReLU)

                with tf.variable_scope('second'):
                    c2 = conv('conv2', c1, kernel2, 1)
                
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
            l = residual('res1.0', l, -1, first=True) # -1 used to signal filters' variable to be initialized
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l, k)
            # 32,c=16

            l = residual('res2.0', l, 0, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l, k)
            # 16,c=32

            l = residual('res3.0', l, 0, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l, k)
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
                                      [(1, 0.1), (60, 0.01), (120, 0.001)])
        ],
        max_epoch=150,
        session_init=SaverRestore(logger.get_logger_dir() + '/checkpoint') if tf.gfile.Exists(logger.get_logger_dir() + '/checkpoint') else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
