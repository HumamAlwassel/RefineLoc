#!/bin/bash
#SBATCH --job-name I3D_ANET1.2
#SBATCH --array=0-499
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 40GB
##SBATCH --qos=ivul

echo `hostname`
# conda activate refineloc
source activate refineloc

DIR=$HOME/refineloc-dev/src
cd $DIR
echo `pwd`

TRAIN_FEATURES_FILENAME=../data/activitynet_v1-2/i3d_wtalc/activitynet_v1-2_train_i3d_features.hdf5
VALID_FEATURES_FILENAME=../data/activitynet_v1-2/i3d_wtalc/activitynet_v1-2_valid_i3d_features.hdf5
TRAIN_LABELS_FILENAME=../data/activitynet_v1-2/i3d_wtalc/activitynet_v1-2_train_groundtruth.csv
VALID_LABELS_FILENAME=../data/activitynet_v1-2/i3d_wtalc/activitynet_v1-2_valid_groundtruth.csv
LOG_DIR=$HOME/pytorch_experiments/refineloc/i3d_activitynet_v1-2_bg_dropout_not_weighted
DEVICE=cuda:0
CONFIG_TYPE=random_i3d_activitynet_v1-2

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE \
                --save_best_valid_predictions

