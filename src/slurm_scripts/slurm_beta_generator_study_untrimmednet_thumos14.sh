#!/bin/bash
#SBATCH --job-name AB_UNT_TH
#SBATCH --array=0-24
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 30GB
##SBATCH --qos=ivul
#SBATCH --constraint=[v100|gtx1080ti]

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
LOG_DIR=$HOME/pytorch_experiments/refineloc/beta_generator_study_untrimmednet_thumos14
DEVICE=cuda:0
CONFIG_TYPE=beta_generator_study_untrimmednet_thumos14
ALPHA_PSEUDO_GT_LOSS=$((2 ** (${SLURM_ARRAY_TASK_ID} / 5)))
PSEUDO_GT_GENERATOR_TYPE=$((${SLURM_ARRAY_TASK_ID} % 5 + 1))

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --alpha_pseudo_gt_loss $ALPHA_PSEUDO_GT_LOSS \
                --pseudo_gt_generator_type $PSEUDO_GT_GENERATOR_TYPE \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE

