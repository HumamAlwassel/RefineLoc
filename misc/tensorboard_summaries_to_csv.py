from __future__ import division, print_function, absolute_import

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import pandas as pd
import json 
import tensorflow as tf
import glob
from tqdm import tqdm
from collections import OrderedDict
import numpy as np

def filter_top_n(df, tags, top_n):
    top_n_df = pd.DataFrame()
    df.sort_values(by=tags, ascending=False, inplace=True)
    top_n_df = top_n_df.append(df.reset_index(drop=True).iloc[:top_n], sort=False)
    return top_n_df

def main(roots, phase, tags_filename, csv_filename, append_cvs_filename, top_n, cluster_name, verbose):
    with open(tags_filename, 'r') as fobj:
        tags = json.load(fobj)

    df = pd.DataFrame()
    for root in roots:
        print('Processing files from {}'.format(root))
        rows = []
        for experiment_foldername in tqdm(glob.glob('{}/{}/*'.format(root, phase))):
            for events_filename in glob.glob('{}/events*'.format(experiment_foldername)):
                this_row = OrderedDict()
                this_row['cluster_name'] = cluster_name
                this_row['phase'] = phase
                class_id = experiment_foldername.split('/')[-2]
                this_row['experiment_folder'] = experiment_foldername
                for t in tags:
                    this_row[t] = None

                try:
                    for e in tf.compat.v1.train.summary_iterator(events_filename):
                        for v in e.summary.value:
                            if verbose:
                                this_row[v.tag] = v.simple_value
                            else:
                                if v.tag in tags:
                                    this_row[v.tag] = v.simple_value
                except Exception as e:
                    logging.info('{} {}'.format(experiment_foldername, e))
                logging.debug(this_row)
                rows += [this_row]

        _df = pd.DataFrame(rows)
        df.append(_df, sort=False)

    if append_cvs_filename is not None:
        append_df = pd.read_csv(append_cvs_filename)
        df = df.append(append_df, sort=False)

    df.to_csv(csv_filename, index=False)

    top_n_df =  filter_top_n(df, tags, top_n)
    top_n_df.to_csv('top_{}_{}'.format(top_n, csv_filename), index=False)    


if __name__ == '__main__':
    parser = ArgumentParser(description='extract values of summary tags from tensorboard events files and dump it in a csv file',
                        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-r', '--roots', required=True, nargs='+',
                      help='root directory containing the experiments')
    parser.add_argument('-p', '--phase', default='train', type=str,
                      help='s directory containing the tensorflow experiments')
    parser.add_argument('-tags', '--tags_filename', required=True, type=str,
                      help='a json filename with a list of tags: [<summary_tag1>, <summary_tag2>, ..., <summary_tagN>]')
    parser.add_argument('-c', '--cluster_name', required=True, type=str,
                      help='the cluster name where the experiment folder lives')
    parser.add_argument('-o', '--csv_filename', required=True, type=str,
                      help='output csv filename with columns: feature_type, phase, experiment_folder, cluster_name, <summary_tag1>, <summary_tag2>, ..., <summary_tagN>')
    parser.add_argument('-a', '--append_cvs_filename', type=str,
                      help='a csv filename from a previous run of this script to be be appended to the results of this run.')
    parser.add_argument('-n', '--top_n', default=1, type=int,
                      help='summarize the results and keep the top-n best experiments from each class for each feature type. Result is saved in <top_n>_<csv_filename>')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                      help='dump all model summaries not just the ones specified in tags')
    parser.add_argument('-log', '--loglevel', default='INFO',
                      help='logging level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')

    main(**vars(args))
