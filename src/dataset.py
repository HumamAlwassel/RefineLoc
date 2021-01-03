import logging
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from scipy.special import softmax
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label as label_connected_component
from scipy.stats import entropy
from matplotlib import pyplot as plt
import json
class Dataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 features_filename,
                 labels_filename,
                 snippet_size,
                 stream,
                 seed=424242):
        """
        Parameters
        ----------
        features_filename : str.
            Path to a features HDF5 file.
        labels_filename : str.
            Path to a CSV file. columns: video-name,t-start,t-end,label,video-duration,fps,video-num-frames
        snippet_size : int.
            Number of frames per snippet
        stream : int
            Selects which stream features to use. Cab be 0 for temporal only, 1 for spatial only, and 2 for both streams
        """
        logging.info(f'Dataset: features_filename {features_filename} labels_filename {labels_filename} snippet_size {snippet_size} stream {stream}')

        np.random.seed(seed)
        self.snippet_size = snippet_size
        self.stream = stream
        self._video_names = None
        self._video_name_to_labels = None
        self._video_name_to_metadata = None
        self._video_name_to_snippets_bg_fg_gt = None
        self._video_name_to_snippets_pseudo_bg_fg_gt = None
        self.num_classes = None
        self.label_name_to_label_id = None
        self.label_id_to_label_name = None
        self._set_labels(labels_filename)

        self._video_name_to_snippets_features = None
        self.feature_size = None
        self.dtype = None
        self._set_features(features_filename)

        self._save_cas_predictions        = dict(zip(self._video_names, len(self._video_names)*[None]))
        self._save_attention_predictions = dict(zip(self._video_names, len(self._video_names)*[None]))

        self._previous_predictions = []

        self._iter_number = 0

        self._fg_class_loss_weight = None

    def _set_labels(self, labels_filename):
        try:
            df = pd.read_csv(labels_filename)

            unique_labels = np.unique(df['label'])
            self.label_name_to_label_id = dict(zip(unique_labels, range(len(unique_labels))))
            self.label_id_to_label_name = dict(zip(range(len(unique_labels)), unique_labels))
            df['label'] = df['label'].map(lambda x: self.label_name_to_label_id[x])

            df_by_video_name = df.groupby(by='video-name')
        except Exception as e:
            raise IOError(f'Invalid labels_filename. Error message: {e}')

        self._video_names = list(df_by_video_name.groups.keys())
        self._video_name_to_labels = {}
        self._video_name_to_metadata = {}
        self._video_name_to_snippets_bg_fg_gt = {}
        self._video_name_to_snippets_pseudo_bg_fg_gt = {}

        num_classes = -1
        for video_name, this_df in df_by_video_name:
            labels = np.unique(this_df['label'])
            num_classes = max(num_classes, max(labels))

            duration =this_df['video-duration'].values[0]
            fps = this_df['fps'].values[0]
            num_frames = this_df['video-num-frames'].values[0]

            bg_fg = np.zeros(int(np.ceil(num_frames/self.snippet_size)), dtype=np.int)
            for _, row in this_df.iterrows():
                bg_fg[int(round(row['t-start']*fps/self.snippet_size)):int(round(row['t-end']*fps/self.snippet_size))] = 1

            self._video_name_to_labels[video_name] = labels
            self._video_name_to_metadata[video_name] = {'duration': duration, 'fps': fps, 'num_frames': num_frames}
            self._video_name_to_snippets_bg_fg_gt[video_name] = bg_fg
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name] = (-1)*np.ones_like(bg_fg)

        self.num_classes = num_classes + 1

    def _set_features(self, features_filename):
        try:
            features = h5py.File(features_filename, 'r')
        except Exception as e:
            raise IOError(f'Invalid HDF5 for the visual observations. Error message: {e}')

        self._video_name_to_snippets_features = {}

        num_fg = 0
        num_bg = 0
        total = 0
        for k,v in features.items():
            if self.stream == 0:
                # get only the temporal stream
                self._video_name_to_snippets_features[k] = v[:,:v.shape[-1]//2]
            elif self.stream == 1:
                # get only the spatial stream
                self._video_name_to_snippets_features[k] = v[:,-v.shape[-1]//2:]
            elif self.stream == 2:
                # temporal + spatial streams
                self._video_name_to_snippets_features[k] = v[:]
            else:
                raise ValueError(f'Got an undefined stream type {self.stream}. Possible values are 0 for temporal only, 1 for spatial only, and 2 for both streams.')

            try:
                self._video_name_to_snippets_bg_fg_gt[k] = self._video_name_to_snippets_bg_fg_gt[k][:len(self._video_name_to_snippets_features[k])]
                self._video_name_to_snippets_pseudo_bg_fg_gt[k] = self._video_name_to_snippets_pseudo_bg_fg_gt[k][:len(self._video_name_to_snippets_features[k])]
                total += len(self._video_name_to_snippets_bg_fg_gt[k])
                num_fg += sum(self._video_name_to_snippets_bg_fg_gt[k])
                num_bg += len(self._video_name_to_snippets_bg_fg_gt[k]) - sum(self._video_name_to_snippets_bg_fg_gt[k])
            except:
                pass

        temp = self._video_name_to_snippets_features[self._video_names[0]]
        self.feature_size = temp.shape[-1]
        self.dtype = temp.dtype

        logging.info(f'num_fg {num_fg} num_bg {num_bg} total {total} num_fg/total {num_fg/total} num_bg/total {num_bg/total}')

    def __len__(self):
        return len(self._video_names)

    def __getitem__(self, idx):
        video_name = self._video_names[idx]
        features = torch.tensor(self._video_name_to_snippets_features[video_name], dtype=torch.float32)
        # TODO: accommodate multi-label videos
        label = torch.tensor([self._video_name_to_labels[video_name][0]], dtype=torch.long)
        pseudo_gt = torch.tensor(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name], dtype=torch.long)

        return video_name, features, label, pseudo_gt

    def collate_fn(self, data_lst):
        video_names = [_video_name for _video_name, _, _, _ in data_lst]
        video_name_to_slice = {}
        current_ind = 0
        for this_video_name, this_features, this_label, this_pseudo_gt in data_lst:
            video_name_to_slice[this_video_name] = (current_ind, current_ind + this_features.shape[0])
            current_ind += this_features.shape[0]
        features    = torch.cat([_features for _, _features, _, _ in data_lst],dim=0)
        labels      = torch.cat([_label for _, _, _label, _ in data_lst],dim=0)
        pseudo_gts  = torch.cat([_pseudo_gt for _, _, _, _pseudo_gt in data_lst],dim=0)

        targets = {'video-names': video_names,
                   'video-name-to-slice': video_name_to_slice,
                   'labels': labels,
                   'pseudo-gt': pseudo_gts,
                   'fg_class_loss_weight': self._fg_class_loss_weight,
                   }
        return features, targets

    def save_predictions(self, targets, cas, attention):
        cas = cas.cpu().detach().numpy()
        attention = attention.cpu().detach().numpy()
        for i, video_name in enumerate(targets['video-names']):
            start, end = targets['video-name-to-slice'][video_name]
            self._save_cas_predictions[video_name]        = cas[start:end]
            self._save_attention_predictions[video_name]  = attention[start:end]

    def _eval_video_label_accuracy(self):
        labels = [self._video_name_to_labels[video_name][0] for video_name in self._video_names]
        predictions_with_attention = []
        predictions_without_attention = []
        for video_name in self._video_names:
            attention = self._save_attention_predictions[video_name]
            cas = self._save_cas_predictions[video_name]
            attention_across_time, _ = smooth_attention_function(attention)
            predictions_with_attention.append(np.sum(attention_across_time * softmax(cas, axis=-1), axis=0).tolist())
            predictions_without_attention.append(np.mean(softmax(cas, axis=-1), axis=0).tolist())
        video_label_accuracy_with_attention = accuracy_score(labels, np.argmax(predictions_with_attention, axis=-1).tolist())
        video_label_accuracy_without_attention = accuracy_score(labels, np.argmax(predictions_without_attention, axis=-1).tolist())
        results = {'video_label_accuracy_with_attention': video_label_accuracy_with_attention*100,
                   'video_label_accuracy_without_attention': video_label_accuracy_without_attention*100,}
        return results

    def _eval_pseudo_gt_label_accuracy(self):
        labels, predictions = [], []
        pseudo_labels, actual_labels, predictions_on_only_pseudo_gt = [], [], []

        for video_name in self._video_names:
            attention = self._save_attention_predictions[video_name]
            _, attention_across_class = smooth_attention_function(attention)
            this_labels = self._video_name_to_snippets_bg_fg_gt[video_name]
            labels.extend(this_labels.tolist())
            if attention.shape[-1] == 2:
                predictions.extend(np.argmax(attention,axis=-1).tolist())
            else:
                predictions.extend((attention_across_class.squeeze(axis=-1) >= 0.5).astype(np.int).tolist())

            this_pseudo_labels = self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]
            idx = this_pseudo_labels >= 0
            pseudo_labels.extend(this_pseudo_labels[idx].tolist())
            actual_labels.extend(this_labels[idx].tolist())
            if attention.shape[-1] == 2:
                predictions_on_only_pseudo_gt.extend(np.argmax(attention[idx],axis=-1).tolist())
            else:
                predictions_on_only_pseudo_gt.extend((sigmoid(attention[idx].squeeze(axis=-1)) >= 0.5).astype(np.int).tolist())

        gt_label_accuracy_using_pseudo = accuracy_score(pseudo_labels, predictions_on_only_pseudo_gt)
        gt_label_accuracy_using_actual = accuracy_score(actual_labels, predictions_on_only_pseudo_gt)
        gt_label_accuracy = accuracy_score(labels, predictions)
        results = {'gt_label_accuracy_using_pseudo': gt_label_accuracy_using_pseudo*100,
                   'gt_label_accuracy_using_actual': gt_label_accuracy_using_actual*100,
                   'gt_label_accuracy': gt_label_accuracy*100,}
        return results

    def eval_saved_predictions(self):
        results = {}
        results.update(self._eval_video_label_accuracy())
        results.update(self._eval_pseudo_gt_label_accuracy())
        return results

    def get_detection_predictions(self, top_k_labels, min_cas_score, min_attention_score, padding):
        logging.info(f'Computing predictions top_k_labels {top_k_labels} min_cas_score {min_cas_score} min_attention_score {min_attention_score}')
        predictions_df_lst = []
        for video_name in self._video_names:
            predictions_df_lst.append(get_detection_predictions_for_one_video(
                                            video_name=video_name,
                                            fps=self._video_name_to_metadata[video_name]['fps'],
                                            snippet_size=self.snippet_size,
                                            cas=self._save_cas_predictions[video_name],
                                            attention=self._save_attention_predictions[video_name],
                                            top_k_labels=top_k_labels,
                                            min_cas_score=min_cas_score,
                                            min_attention_score=min_attention_score,
                                            padding=padding)
                                    )
        predictions_df = pd.concat(predictions_df_lst, axis=0).reset_index(drop=True)
        predictions_df['label'] = predictions_df['label'].map(lambda x: self.label_id_to_label_name[x])
        return predictions_df

    def get_oracle_predictions(self, padding):
        #logging.info(f'Computing predictions top_k_labels {top_k_labels} min_cas_score {min_cas_score} min_attention_score {min_attention_score}')
        predictions_df_lst = []
        for video_name in self._video_names:
            predictions_df_lst.append(get_oracle_predictions_for_one_video(
                        oracle_predictions=self._video_name_to_snippets_bg_fg_gt[video_name],
                        label=self._video_name_to_labels[video_name][0],
                        video_name=video_name,
                        fps=self._video_name_to_metadata[video_name]['fps'],
                        snippet_size=self.snippet_size,
                        padding=padding)
                        )
        predictions_df = pd.concat(predictions_df_lst, axis=0).reset_index(drop=True)
        predictions_df['label'] = predictions_df['label'].map(lambda x: self.label_id_to_label_name[x])
        return predictions_df

    def update_pseudo_bg_fg_gt(self, pseudo_gt_generator_type, top_k_labels, min_cas_score, min_attention_score,
                               padding, from_m_iterations, pseudo_gt_dropout):

        switcher = {0: self._oracle_pseudo_gt_generator,
                    1: self._prediction_based_pseudo_gt_generator,
                    2: self._attention_based_pseudo_gt_generator,
                    3: self._cas_based_pseudo_gt_generator,
                    4: self._biased_random_pseudo_gt_generator,
                    5: self._uniform_random_pseudo_gt_generator,
                    }
        generator = switcher.get(pseudo_gt_generator_type, None)

        if generator is None:
            raise ValueError(f'Invalid pseudo_gt_generator_type {pseudo_gt_generator_type}.')

        self._iter_number += 1
        args = {'top_k_labels': top_k_labels,
                'min_cas_score': min_cas_score,
                'min_attention_score': min_attention_score,
                'padding': padding,
                'from_m_iterations': from_m_iterations}
        logging.info(f'Generating pseudo background/foreground ground truth using {generator.__name__}. args {args}')

        generator(args)

        stats = self._compute_stats_for_pseudo_gt()


        # Drop out some of the pseudo gt labels
        for video_name in self._video_names:
            this_video = self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]
            idx = np.random.choice(range(len(this_video)), size=int(len(this_video)*pseudo_gt_dropout), replace=False)
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name][idx] = -1


        # update fg_class_loss_weight
        self._fg_class_loss_weight = stats['fg_class_loss_weight']

        return stats

    def _compute_stats_for_pseudo_gt(self):
        num_true_fg, num_true_bg, total_pseudo_fg, total_pseudo_bg, total_fg, total_bg = 0, 0, 0, 0, 0, 0

        for video_name in self._video_names:
            num_true_fg += sum(self._video_name_to_snippets_bg_fg_gt[video_name][self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]==1])
            num_true_bg += sum(1 - self._video_name_to_snippets_bg_fg_gt[video_name][self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]==0])
            total_pseudo_fg += sum(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]==1)
            total_pseudo_bg += sum(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name]==0)
            total_fg += sum(self._video_name_to_snippets_bg_fg_gt[video_name]==1)
            total_bg += sum(self._video_name_to_snippets_bg_fg_gt[video_name]==0)

        stats = {'num_true_fg': num_true_fg, 'num_true_bg': num_true_bg,
                 'total_pseudo_fg': total_pseudo_fg, 'total_pseudo_bg': total_pseudo_bg,
                 'num_true_fg/total_pseudo_fg': 0 if total_pseudo_fg == 0 else num_true_fg/total_pseudo_fg,
                 'num_true_bg/total_pseudo_bg': 0 if total_pseudo_bg == 0 else num_true_bg/total_pseudo_bg,
                 'num_true_fg/total_fg': 0 if total_fg == 0 else num_true_fg/total_fg,
                 'num_true_bg/total_bg': 0 if total_bg == 0 else num_true_bg/total_bg,
                 'fg_class_loss_weight': total_pseudo_bg / (total_pseudo_bg + total_pseudo_fg)}
        return stats

    def _oracle_pseudo_gt_generator(self, args):
        for video_name in self._video_names:
            # rest all snippets to background
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name][:] = 0 # pseudo background

        predictions_df = self.get_oracle_predictions(padding=args['padding'])
        self._previous_predictions.append(predictions_df)
        if len(self._previous_predictions) > args['from_m_iterations']:
            self._previous_predictions = self._previous_predictions[-args['from_m_iterations']:]
        predictions_df = pd.concat(self._previous_predictions, axis=0).reset_index(drop=True)

        for video_name, this_df in predictions_df.groupby(by='video-name'):
            fps = self._video_name_to_metadata[video_name]['fps']
            # set the snippets in the prediction segments to foreground
            for _, row in this_df.iterrows():
                start_snippet = max(min(int(round(row['t-start']*fps/self.snippet_size)), len(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name])), 0)
                end_snippet = max(min(int(round(row['t-end']*fps/self.snippet_size)), len(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name])), 0)
                self._video_name_to_snippets_pseudo_bg_fg_gt[video_name][start_snippet:end_snippet+1] = 1 # pseudo foreground

    def _prediction_based_pseudo_gt_generator(self, args):
        for video_name in self._video_names:
            # rest all snippets to background
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name][:] = 0 # pseudo background

        predictions_df = self.get_detection_predictions(args['top_k_labels'], args['min_cas_score'], args['min_attention_score'], args['padding'])
        self._previous_predictions.append(predictions_df)
        if len(self._previous_predictions) > args['from_m_iterations']:
            self._previous_predictions = self._previous_predictions[-args['from_m_iterations']:]
        predictions_df = pd.concat(self._previous_predictions, axis=0).reset_index(drop=True)

        for video_name, this_df in predictions_df.groupby(by='video-name'):
            fps = self._video_name_to_metadata[video_name]['fps']
            # set the snippets in the prediction segments to foreground
            for _, row in this_df.iterrows():
                start_snippet = max(min(int(round(row['t-start']*fps/self.snippet_size)), len(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name])), 0)
                end_snippet = max(min(int(round(row['t-end']*fps/self.snippet_size)), len(self._video_name_to_snippets_pseudo_bg_fg_gt[video_name])), 0)
                self._video_name_to_snippets_pseudo_bg_fg_gt[video_name][start_snippet:end_snippet+1] = 1 # pseudo foreground

    def _attention_based_pseudo_gt_generator(self, args):
        args['min_cas_score'] = 0
        self._prediction_based_pseudo_gt_generator(args)

    def _cas_based_pseudo_gt_generator(self, args):
        args['min_attention_score'] = 0
        self._prediction_based_pseudo_gt_generator(args)

    def _biased_random_pseudo_gt_generator(self, args):
        for video_name in self._video_names:
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name] = np.random.choice([0, 1], size=len(self._video_name_to_snippets_bg_fg_gt[video_name]), p=[0.4, 0.6])

    def _uniform_random_pseudo_gt_generator(self, args):
        for video_name in self._video_names:
            self._video_name_to_snippets_pseudo_bg_fg_gt[video_name] = np.random.choice([0, 1], size=len(self._video_name_to_snippets_bg_fg_gt[video_name]), p=[0.5, 0.5])


def sigmoid(X):
    return 1/(1+np.exp(-X))

def smooth_cas(cas):
    cas_shifted = np.append(cas[1:,...], cas[:1,...], axis=0)
    return (cas + cas_shifted) / 2

def min_or_max(x, shift):
    return max(0,x) if shift<=0 else min(x,1)


def smooth_attention_function(input_vector):
    if input_vector.shape[-1] == 2:
        out_attention_class = softmax(input_vector, axis=-1)
        out_attention_time = softmax(out_attention_class, axis=-2)[..., 1:2]
        out_attention_class = out_attention_class[...,1]
    else:
        out_attention_time = softmax(input_vector, axis=-2)
        out_attention_class = sigmoid(input_vector)
    return out_attention_time, out_attention_class

def get_detection_predictions_for_one_video(video_name, fps, snippet_size, cas, attention,
                                            top_k_labels, min_cas_score, min_attention_score, padding):
    cas = softmax(cas, axis=-1)
    attention_across_time, attention_across_class = smooth_attention_function(attention)
    video_score = np.sum(attention_across_time * cas, axis=0)

    predictions = {'t-start': [],
                   't-end': [],
                   'label': [],
                   'score': [],
                   'video-name': []
                   }

    num_snippets = cas.shape[0]
    for label in np.argsort(video_score)[::-1][:top_k_labels]:
        this_label_cas = cas[:, label]
        mask = np.logical_and(this_label_cas > min_cas_score, attention_across_class.squeeze() > min_attention_score)
        mask = binary_dilation(mask)
        labeled_mask, num_components = label_connected_component(mask.astype(int), connectivity=1,
                                                                     return_num=True)
        for k in range(1, num_components + 1):
            idxs, = np.where(labeled_mask == k)
            snippet_start = max(idxs[0] - padding, 0)
            snippet_end = min(idxs[-1] + padding, num_snippets - 1)
            t_start = snippet_start * snippet_size / fps
            t_end = (snippet_end + 1) * snippet_size / fps

            if attention.shape[-1] == 1:
                score = np.mean(this_label_cas[idxs]) + np.mean(
                    sigmoid(attention[idxs])) + video_score[label]
            else:
                score = np.mean(this_label_cas[idxs]) + np.mean(
                    attention_across_class[idxs]) + video_score[label]

            predictions['t-start'].append(t_start)
            predictions['t-end'].append(t_end)
            predictions['label'].append(label)
            predictions['score'].append(score)
            predictions['video-name'].append(video_name)

    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def get_oracle_predictions_for_one_video(oracle_predictions, label, video_name, fps, snippet_size, padding):

    predictions = {'t-start': [],
                   't-end': [],
                   'label': [],
                   'score': [],
                   'video-name': []
                   }

    mask = oracle_predictions
    num_snippets = mask.shape[0]
    mask = binary_dilation(mask)
    labeled_mask, num_components = label_connected_component(mask.astype(int), connectivity=1, return_num=True)
    for k in range(1,num_components+1):
        idxs, = np.where(labeled_mask == k)
        snippet_start = max(idxs[0] - padding, 0)
        snippet_end = min(idxs[-1] + padding, num_snippets - 1)
        t_start = snippet_start*snippet_size/fps
        t_end = (snippet_end + 1)*snippet_size/fps

        predictions['t-start'].append(t_start)
        predictions['t-end'].append(t_end)
        predictions['label'].append(label)
        predictions['score'].append(1)
        predictions['video-name'].append(video_name)

    predictions_df = pd.DataFrame(predictions)
    return predictions_df

def oic_loss(cas, inner_start, inner_end, outer_pcn=0.25):
    """ Computes Outer Inner Loss (OIC)

        Parameters
        ----------
        cas: ndarray
            Class activation sequence of size T where T is number
            of snippets and C is number of classes.
        t_start: float
            Normalized starting time.
        t_end: float
            Normalized ending time.
        outer_pcn: float, default 0.25
            Ratio of segment length that will be used to augment
            the outer area.
    """

    num_snippets = cas.shape[0]

    # Compute inner area.
    # inner_start = np.round(num_snippets * t_start).astype(int)
    # inner_end = np.round(num_snippets * t_end).astype(int)
    inner_area = np.mean(cas[inner_start:inner_end])

    # Compute outer area.
    inflation_length = np.round((inner_end - inner_start) * outer_pcn).astype(int)
    inflation_start_limit = inner_start - inflation_length if (inner_start - inflation_length) > 0 else 0
    inflation_end_limit = inner_end + inflation_length if (inner_end + inflation_length) < num_snippets else num_snippets - 1
    left_activation_aggregation = cas[inflation_start_limit:inner_start].sum()
    rigth_activation_aggreagation = cas[inner_end:inflation_end_limit].sum()
    num_outer_snippets = (inner_start - inflation_start_limit) + (inflation_end_limit - inner_end)
    outer_area = left_activation_aggregation + rigth_activation_aggreagation
    if num_outer_snippets > 0:
        outer_area /= num_outer_snippets
    else:
        outer_area = 0.0

    # Compute OIC loss.
    if outer_area == inner_area:
        oic = 1.0
    else:
        oic = outer_area - inner_area

    return oic
