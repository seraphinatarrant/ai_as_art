#!/bin/sh

cd ~/git/ai_as_art/image_generation/

img_sz=$4
data_dir=$1
out_dir=$2
gpus=$3
n_feats=$5
bsz=$6

python main.py --dataset folder --dataroot ${data_dir} --workers 4 \
--batchSize ${bsz} --imageSize ${img_sz} --nz 100 \
--ngf ${n_feats} --ndf ${n_feats} --niter 200 --lr 0.0002 \
--beta1 0.5 --cuda --ngpu ${gpus} --outf ${out_dir}
#--netG temp/run1/netG_epoch_19.pth --netD temp/run1/netD_epoch_19.pth \
#--manualSeed 9234       
