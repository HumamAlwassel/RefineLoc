#!/bin/bash
#SBATCH --job-name RNDHACS
#SBATCH --array=0-499
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --constraint=[gtx1080ti|v100]
#SBATCH --mem 64GB
##SBATCH --qos=ivul

echo `hostname`
# conda activate refineloc
source activate refineloc

DIR=$HOME/refineloc-dev/src
cd $DIR
echo `pwd`

TRAIN_FEATURES_FILENAME=../data/hacs/tsn/HACS_kinetics_train_tsn_features.hdf5
VALID_FEATURES_FILENAME=../data/hacs/tsn/HACS_kinetics_valid_tsn_features.hdf5
TRAIN_LABELS_FILENAME=../data/hacs/tsn/HACS_kinetics_train_groundtruth.csv
VALID_LABELS_FILENAME=../data/hacs/tsn/HACS_kinetics_valid_groundtruth.csv
LOG_DIR=$HOME/pytorch_experiments/refineloc/hacs
DEVICE=cuda:0
CONFIG_TYPE=random_hacs

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE \

