import pandas as pd
import numpy as np

def nms_detections(dets, score, overlap=0.65):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.
    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.
    arguments:
    ---------
    dets : ndarray.
        Each row is ['f-init', 'f-end']
    score : 1darray.
        Detection score.
    overlap : float.
        Minimum overlap ratio (0.65 default).
    returns:
    -------
    pick : 1darray.
        Remaining after suppression.
    pick_scores : 1darray.
        Scores of remaining segments after suppression. score = unsuppressed_segment_score + sum over all suppressed_segments_scores.
    """
    t1 = dets[:, 0]
    t2 = dets[:, 1]
    ind = np.argsort(score)

    area = (t2 - t1).astype(float)

    pick = []
    pick_scores = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        this_pick_score = score[i]
        ind = ind[:-1]

        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])

        intersection = np.maximum(0., tt2 - tt1)
        iou = intersection / (area[i] + area[ind] - intersection)
        suppressed_idx = np.nonzero(iou > overlap)[0]
        unsuppressed_idx = np.nonzero(iou <= overlap)[0]

        pick_scores.append(score[i] + np.sum(score[ind[suppressed_idx]]))
        ind = ind[unsuppressed_idx]

    return pick, pick_scores

def nms_with_hough_voting(predictions_df, nms_overlap=0.65):
    """Apply non-max-suppression to predictions and add the scores of the suppressed segments to the unsuppressed segment.
    """
    new_predictions_df = []
    for (video_name, label), this_df in predictions_df.groupby(by=['video-name', 'label']):
        this_df = this_df.reset_index(drop=True)
        loc = np.stack((this_df['t-start'], this_df['t-end']), axis=-1)
        pick, pick_scores = nms_detections(loc, np.array(this_df['score']), nms_overlap)
        filtered_df = this_df.loc[pick].reset_index(drop=True)
        filtered_df['score'] = pick_scores
        new_predictions_df.append(filtered_df)
    return pd.concat(new_predictions_df, axis=0).reset_index(drop=True)

def remove_nonsense_predictions(predictions_df, ground_truth_df):
    """ remove predictions that are outside the bound of the video. Clip to the bound of the video if some of the prediction is outside
    """
    video_name_to_duration = dict(zip(ground_truth_df['video-name'], ground_truth_df['video-duration']))
    predictions_df['video-duration'] = predictions_df['video-name'].map(lambda x: video_name_to_duration[x])
    predictions_df['t-start'] = np.maximum(0.0, predictions_df['t-start'])
    predictions_df['t-end'] = np.minimum(predictions_df['video-duration'], predictions_df['t-end'])
    predictions_df = predictions_df.loc[predictions_df['t-start'].values < predictions_df['t-end'].values].reset_index(drop=True)
    return predictions_df

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union between segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in xrange(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    return tiou
