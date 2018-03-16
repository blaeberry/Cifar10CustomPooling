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

BATCH_SIZE = 64
NUM_BLOCKS = 3
START_DEPTH = 24

class Model(ModelDesc):

    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - (NUM_BLOCKS + 1)) // NUM_BLOCKS) #layers per block
        #the updated growthRate can be calculated as new_growthRate = (2*old_growthRate*N)/(N^2+N)
        self.growthRate = 1

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 - 1
        assert tf.test.is_gpu_available()

        def conv(name, l, kernel, stride):
            conv = tf.nn.conv2d(l, kernel, [1, stride, stride, 1], padding = 'SAME', data_format='NHWC')
            ret = tf.identity(conv, name = name)
            ret.variables = VariableHolder(W=kernel)
            return ret

        def add_layer(name, l, kernel):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name, reuse=False) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                c = conv('conv1', c, kernel, 1)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name, reuse=False) as scope:
                kernel = expand_filters_transition(in_channel)
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = conv('conv1', l, kernel, 1)
                l = AvgPooling('pool', l, 2)
            return l

        def expand_filters(in_channels, x):
            num_total_filters = (self.N * self.growthRate * 3) #12 * 1 * 3
            num_used_filters = (x+1)*self.growthRate
            with tf.variable_scope('reused') as scope:
                if x > 0:
                    scope.reuse_variables() 
                filters = tf.get_variable('filters', (3, 3, num_total_filters), 
                    initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            ch_prefs = tf.get_variable('ch_prefs.{}'.format(x), (in_channels, num_used_filters), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            kernel = filters[:, :, 0:(num_used_filters)]
            kernel = tf.expand_dims(kernel, axis = 2) #[3,3,1,num_used_filters]
            kernel = kernel * ch_prefs #[3,3,1,num_used_filters] * [in_channels,num_used_filters]
            #returns the selected filters expanded across the depth for convolution
            return kernel

        def expand_filters_transition(in_channels):
            filters = tf.get_variable('filters', (1, 1, in_channels), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            ch_prefs = tf.get_variable('ch_prefs', (in_channels, in_channels), 
                initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))
            kernel = tf.expand_dims(filters, axis = 2) #w,h,1,num_filters
            kernel = kernel * ch_prefs #w,h,1,num_filters * in_channels,num_filters
            #returns the updated filters and the filters expanded across the depth for convolution
            return kernel

        def densenet(name):
            l = Conv2D('conv0', image, START_DEPTH, 3, stride=1,
                nl=tf.identity, use_bias=False,
                W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out'))

            for i in range(self.N):
                in_channel = l.get_shape().as_list()[3]
                kernel = expand_filters(in_channel, i) # kernel = [3, 3, in_channel, num_used_filters]
                with tf.variable_scope('block1') as scope:
                    l = add_layer('dense_layer.{}'.format(i), l, kernel)
            with tf.variable_scope('block1') as scope:
                l = add_transition('transition1', l)

            for i in range(self.N):
                in_channel = l.get_shape().as_list()[3]
                kernel = expand_filters(in_channel, i+self.N)
                with tf.variable_scope('block2') as scope:
                    l = add_layer('dense_layer.{}'.format(i), l, kernel)
            with tf.variable_scope('block2') as scope:
                l = add_transition('transition2', l)

            for i in range(self.N):
                in_channel = l.get_shape().as_list()[3]
                kernel = expand_filters(in_channel, i+(self.N*2))
                with tf.variable_scope('block3') as scope:
                    l = add_layer('dense_layer.{}'.format(i), l, kernel)

            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

            return logits

        logits = densenet('densenet')
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        # wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
        #                                   480000, 0.2, True)
        wd_w = 1e-4
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
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
    parser.add_argument('--depth',default=40, help='The depth of densenet')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir(action='k')

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = TrainConfig(
        model=Model(depth=args.depth),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(max_to_keep = 5, keep_checkpoint_every_n_hours = 10000),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate',
                                      [(1, 0.1), (150, 0.01), (225, 0.001)])
        ],
        max_epoch=300,
        session_init=SaverRestore(logger.get_logger_dir() + '/checkpoint') if tf.gfile.Exists(logger.get_logger_dir() + '/checkpoint') else None
    )
    nr_gpu = max(get_nr_gpu(), 1)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu))
