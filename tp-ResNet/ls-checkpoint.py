#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ls-checkpoint.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import six
import sys
import pprint
import re

from tensorpack.tfutils.varmanip import get_checkpoint_path
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

if __name__ == '__main__':
    fpath = sys.argv[1]

    if fpath.endswith('.npy'):
        params = np.load(fpath, encoding='latin1').item()
        dic = {k: v.shape for k, v in six.iteritems(params)}
    elif fpath.endswith('.npz'):
        params = dict(np.load(fpath))
        dic = {k: v.shape for k, v in six.iteritems(params)}
    else:
        path = get_checkpoint_path(sys.argv[1])
    #    print_tensors_in_checkpoint_file(path, '', all_tensors = True)
        reader = tf.train.NewCheckpointReader(path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            if(re.search(r".*valid.*", key)):
               print("tensor_name: ", key)
               print(reader.get_tensor(key)) 
        #dic = reader.get_variable_to_shape_map()
    #pprint.pprint(dic)
