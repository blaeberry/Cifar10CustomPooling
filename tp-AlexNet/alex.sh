#!/bin/bash

# Submit Description File Executable
source /scratch/cluster/software/bin/tensorflow-setup
python cifar10-alexnet.py --gpu 0
