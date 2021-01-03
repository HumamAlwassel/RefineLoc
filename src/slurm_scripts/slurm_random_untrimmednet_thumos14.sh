#!/bin/bash
#SBATCH --job-name UNT_TH14
#SBATCH --array=0-499%50
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 30GB
#SBATCH --constraint=[v100|p100|p6000|gtx1080ti]
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
LOG_DIR=$HOME/pytorch_experiments/refineloc/untrimmednet_thumos_dropout_weighted_no_shots
DEVICE=cuda:0
CONFIG_TYPE=random_untrimmednet_thumos14

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE \
                --save_best_valid_predictions
