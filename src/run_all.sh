#!/bin/bash

ARG_DEVICE=--device=${1:-'cpu'}

ARG_COMPILE=--${2:-'no-compile'}

ARG_BACKEND=--backend=${3:-'inductor'}

ARG_BATCHSIZE=--meta_batch_size=${4:-'128'}

# Experiment 1
export CONFIG_K=1
export CONFIG_N=2
python main.py --num_shot 1 --num_classes 2 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

# Experiment 2
export CONFIG_K=2
export CONFIG_N=2
python main.py --num_shot 2 --num_classes 2 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

# Experiment 3
export CONFIG_K=1
export CONFIG_N=3
python main.py --num_shot 1 --num_classes 3 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE

# Experiment 4
export CONFIG_K=1
export CONFIG_N=4
python main.py --num_shot 1 --num_classes 4 $ARG_DEVICE $ARG_COMPILE $ARG_BACKEND $ARG_BATCHSIZE
