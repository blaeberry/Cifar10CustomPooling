#!/bin/bash

# Submit Description File Executable
source /scratch/cluster/software/bin/tensorflow-setup
python cifar10-resnet.py --gpu 0
