#!bin/bash

#python tensorboard_summaries_to_csv.py -r ~/pytorch_experiments/refineloc/ -p thumos14 -tags tags.json -c ibex -o thumos14_summaries.csv -n 10 -v
python tensorboard_summaries_to_csv.py -r /run/user/1001/gvfs/sftp:host=104.196.137.174,user=pardogl/home/pardogl/pytorch_experiments/refineloc/ \ #/run/user/1001/gvfs/sftp:host=35.196.3.245,user=pardogl/home/pardogl/pytorch_experiments/refineloc/ \
                                        -p i3d_activitynet_v1-2_logistic_regression \
                                        -tags tags.json \
                                        -c gcloud2 \
                                        -o activitynet_v1-2_summaries__.csv \
                                        -n 10 -v
#python tensorboard_summaries_to_csv.py -r ~/pytorch_experiments/refineloc/ -p ablation_study_activitynet_v1-2 -tags tags.json -c ibex -o ablation_study_activitynet_v1-2_summaries.csv -n 10 -v
#python tensorboard_summaries_to_csv.py -r ~/pytorch_experiments/refineloc/ -p ablation_study_thumos14 -tags tags.json -c ibex -o ablation_study_thumos14_summaries.csv -n 10 -v


