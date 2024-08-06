#!/bin/bash
### Platform check
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=info
export OMP_NUM_THREADS=1
ulimit -l 131072
export JOB_NAME=$(cat /etc/hostname | cut -d '-' -f 1,2,3)
export MASTER_FILE=$HOME/master_ip.${JOB_NAME}

torchrun --nproc_per_node=8 train_ddp_mineclip.py --use_mask --batch_size 420 --batch_size_val 256
