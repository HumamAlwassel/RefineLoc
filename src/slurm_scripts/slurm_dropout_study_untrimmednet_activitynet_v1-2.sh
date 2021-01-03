#!/bin/bash
#SBATCH --job-name DO_UNT_AN
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 40GB
##SBATCH --qos=ivul
#SBATCH --constraint=[v100|gtx1080ti]

echo `hostname`
# conda activate refineloc
source activate refineloc

DIR=$HOME/refineloc-dev/src
cd $DIR
echo `pwd`

TRAIN_FEATURES_FILENAME=../data/activitynet_v1-2/untrimmednet_autoloc/activitynet_v1-2_train_untrimmednet_features.hdf5
VALID_FEATURES_FILENAME=../data/activitynet_v1-2/untrimmednet_autoloc/activitynet_v1-2_valid_untrimmednet_features.hdf5
TRAIN_LABELS_FILENAME=../data/activitynet_v1-2/untrimmednet_autoloc/activitynet_v1-2_train_groundtruth.csv
VALID_LABELS_FILENAME=../data/activitynet_v1-2/untrimmednet_autoloc/activitynet_v1-2_valid_groundtruth.csv
LOG_DIR=$HOME/pytorch_experiments/refineloc/dropout_study_untrimmednet_activitynet_v1-2
DEVICE=cuda:0
CONFIG_TYPE=dropout_study_untrimmednet_activitynet_v1-2
PSEUDO_GT_DROPOUT=$((100*${SLURM_ARRAY_TASK_ID}/5))

mkdir -p $LOG_DIR
python train.py --train_features_filename $TRAIN_FEATURES_FILENAME \
                --valid_features_filename $VALID_FEATURES_FILENAME \
                --train_labels_filename $TRAIN_LABELS_FILENAME \
                --valid_labels_filename $VALID_LABELS_FILENAME \
                --pseudo_gt_dropout $PSEUDO_GT_DROPOUT \
                --log_dir $LOG_DIR \
                --device $DEVICE \
                --config_type $CONFIG_TYPE
