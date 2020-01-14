#!/bin/sh

cd ~/git/examples/dcgan/

img_sz=$4
data_dir=$1
out_dir=$2
gpus=$3
n_feats=$5
batch=$6

python main.py --dataset folder --dataroot ${data_dir} --workers 4 \
--batchSize $6 --imageSize ${img_sz} --nz 100 \
--ngf ${n_feats} --ndf ${n_feats} --niter 200 --lr 0.0002 \
--beta1 0.5 --cuda --ngpu ${gpus} --outf ${out_dir}
