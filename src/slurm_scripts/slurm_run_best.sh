#!/bin/bash
#SBATCH --job-name PLYGRND
#SBATCH --array=0-1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 30GB
##SBATCH --qos=ivul

echo `hostname`
# conda activate refineloc
source activate refineloc

DIR=$HOME/refineloc-dev/src
cd $DIR
echo `pwd`

TRAIN_FEATURES_FILENAME=../data/thumos14/untrimmednet_autoloc/thumos14_valid_untrimmednet_features.hdf5
VALID_FEATURES_FILENAME=../data/thumos14/untrimmednet_autoloc/thumos14_test_untrimmednet_features.hdf5
TRAIN_LABELS_FILENAME=../data/thumos14/untrimmednet_autoloc/thumos14_valid_groundtruth.csv
VALID_LABELS_FILENAME=../data/thumos14/untrimmednet_autoloc/thumos14_test_groundtruth.csv
LOG_DIR=$HOME/shared/refineloc-dev/logs/thumos14/playground_sigmoid
DEVICE=cuda:0
CONFIG_TYPE=best_activitynet_v1-2 #for THUMOS14 change it to: best_thumos14

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE \
