#!/bin/sh
#SBATCH -N 1 # Nodes requested
#SBATCH -n 1  # Tasks requested
#SBATCH --gres=gpu:4   # Tasks requested
#SBATCH --mem=14000 # Memory in Mb
#SBATCH --time=0-24:00:00 # Time required
#SBATCH --partition=PGR-Standard

cd ~
source ~/.bashrc
export STUDENT_ID=$(whoami)

export PYTHON_PATH=$PATH

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}
export RESDIR=/disk/scratch/${STUDENT_ID}/${SLURM_JOB_ID}/
export REPO=git/examples/dcgan/
export DATADIR=~/data/asa/
#later add STILLS to this....in case need validation and train data separately
GPUS=4
IMG_SZ=256
N_FEAT=64
#shift

conda activate dcgan

# Some logging of settings and time
echo "Machine:"
hostname
echo "Number of GPUS: "${SLURM_GPUS_PER_TASK}
#echo "Date:"
#date -u
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo "Timestamp: "${TIMESTAMP}

export HOME_RESDIR=~/${TIMESTAMP}_sl${SLURM_JOB_ID}/
mkdir -p $HOME_RESDIR
mkdir -p $TMPDIR
mkdir -p $RESDIR

echo "Copying Files:"
### Copy Files to Scratch on Node
#rsync archive verbose update compress human readable FROM TO
rsync -auzhr ${DATADIR} ${TMPDIR}

### Set or grab params
#embed_size=300
#lr=30
#exp=$RANDOM
#model="small_test_${exp}"
#vocab="small_test"

echo "Starting Training..."
bash ${REPO}run_asa_dcgan.sh ${TMPDIR} ${RESDIR} ${GPUS} ${IMG_SZ}


echo "Training Complete, Copying Results"
### Copy Results from Scratch on Node back to filesystem
rsync -avuzhr ${RESDIR} ${HOME_RESDIR}

echo "Fin."
