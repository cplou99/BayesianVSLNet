import glob
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer

def load_json(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode="w", encoding="utf-8") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_lines(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        return [e.strip("\n") for e in f.readlines()]


def save_lines(data, filename):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write("\n".join(data))


def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_sentences():
    data_dir = os.path.join("data", "dataset", 'ego4d_goalstep')
    sentences = dict()
    for split in ['train', 'val', 'test']:
        data = load_json(os.path.join(data_dir, f"{split}.json"))
        for vid in data.keys():
            vid_dict = dict()
            for sentence in data[vid]['sentences']:
                vid_dict[sentence] = None               
        print("Dictionary of vid", vid, "is loaded as", vid_dict)
        sentences[vid] = vid_dict
    return sentences

def load_objects():
    objects_dir = os.path.join("data", "object_detection_dicts")
    objects = dict()
    for split in ['train', 'val', 'test']:
        data = load_json(os.path.join(objects_dir, f"{split}_objects.json"))
        for vid in data.keys():
           objects[vid] = data[vid]         
    print("Length of objects dictionary:", len(objects))
    return objects



def load_video_features(root, max_position_length, load_omnivore=False, sampling="uniform"):
    video_features = dict()
    video_sampling_idxs = dict()
    extension = "*.pt"
    filenames = glob.glob(os.path.join(root, extension))
    min_features_dim = 1e10
    # Load sentences to perform a sampling specific for each text query
    for filename in tqdm(filenames, total=len(filenames), desc="load video features"):
        video_id = filename.split("/")[-1].split(".")[0]
        egovlp_feature = torch.load(filename).cpu().numpy()
        if load_omnivore:
            onmivore_feature = torch.load(filename.replace("EgoVLPv2_video_NO_head", "omnivore_video_swinl") ).cpu().numpy()
        else:
            print("Not loading Omnivore features")
        
        if max_position_length is None:
            if load_omnivore:
                egovlp_frames = egovlp_feature.shape[0]
                onmivore_feature = onmivore_feature[:egovlp_frames]
                video_features[video_id] = np.concatenate([egovlp_feature, onmivore_feature], axis=1) #(128, 1536 + 768)
            else: 
                video_features[video_id] = egovlp_feature
            
        else:
            egovlp_sampled_feature, sampling_idxs = visual_feature_sampling(egovlp_feature, max_num_clips=max_position_length)

            if load_omnivore:
                omnivore_sampled_feature, _ = visual_feature_sampling(onmivore_feature, max_num_clips=max_position_length)
                    
                if omnivore_sampled_feature.shape[0] > egovlp_sampled_feature.shape[0]:
                    omnivore_sampled_feature = omnivore_sampled_feature[:egovlp_sampled_feature.shape[0]]
                    sampling_idxs = sampling_idxs[:egovlp_sampled_feature.shape[0]]
                if omnivore_sampled_feature.shape[0] < egovlp_sampled_feature.shape[0]:
                    egovlp_sampled_feature = egovlp_sampled_feature[:omnivore_sampled_feature.shape[0]]
                    sampling_idxs = sampling_idxs[:omnivore_sampled_feature.shape[0]]
                video_features[video_id] = np.concatenate([egovlp_sampled_feature, omnivore_sampled_feature], axis=1)
            else:
                video_features[video_id] = egovlp_sampled_feature
            
            video_sampling_idxs[video_id] = sampling_idxs
    return video_features, video_sampling_idxs  


def visual_feature_sampling(visual_feature, max_num_clips):
    num_clips = visual_feature.shape[0]
    if num_clips <= max_num_clips:
        idxs = np.arange(num_clips).astype(np.int32)
        return visual_feature, idxs
    idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_visual_feature = []
    for i in range(max_num_clips):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
        else:
            new_visual_feature.append(visual_feature[s_idx])
    new_visual_feature = np.asarray(new_visual_feature)
    return new_visual_feature, idxs



def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def compute_overlap_assuming_pred_list(pred, gt):
    # check format
    assert isinstance(gt, list)
    gt_is_list = isinstance(gt[0], list)
    gt = gt if gt_is_list else [gt]
    # compute overlap
    gt = np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    return overlap


def time_to_index(
    start_time, end_time, num_units, duration, cache_candidate_units=None
):
    if cache_candidate_units is None:
        s_times = (
            np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
        )
        e_times = (
            np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
        )
        candidates = np.stack(
            [
                np.repeat(s_times[:, None], repeats=num_units, axis=1),
                np.repeat(e_times[None, :], repeats=num_units, axis=0),
            ],
            axis=2,
        ).reshape((-1, 2))
    else:
        candidate_units = cache_candidate_units[:num_units, :num_units, :]
        candidates = candidate_units / float(num_units) * duration
        candidates = candidates.reshape((-1, 2))
    # This significantly speeds up calculations.
    overlaps = compute_overlap_assuming_pred_list(
        candidates, [start_time, end_time]
    ).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def time_to_one_hot_index(
    start_time, end_time, num_units, duration, cache_candidate_units=None
):
    start_index, end_index, _ = time_to_index(
        start_time, end_time, num_units, duration, cache_candidate_units
    )

    one_hot_idx = np.zeros(num_units)
    one_hot_idx[start_index:end_index] = 1
    return one_hot_idx

def new_time_to_index(
    start_time, end_time, sampling_idxs, duration, original_feat_len
):
    num_units = len(sampling_idxs)
    s_times = (
        np.arange(0, original_feat_len).astype(np.float32) / float(original_feat_len) * duration
    )
    e_times = (
        np.arange(1, original_feat_len + 1).astype(np.float32) / float(original_feat_len) * duration
    )
    
    s_times = s_times[sampling_idxs]
    e_times = e_times[sampling_idxs]
    
    candidates = np.stack(
        [
            np.repeat(s_times[:, None], repeats=num_units, axis=1),
            np.repeat(e_times[None, :], repeats=num_units, axis=0),
        ],
        axis=2,
    ).reshape((-1, 2))

    # This significantly speeds up calculations.
    overlaps = compute_overlap_assuming_pred_list(
        candidates, [start_time, end_time]
    ).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    print()
    return start_index, end_index, overlaps

def batched_time_to_index(start_end_times, num_units, duration):
    start_end_times = np.array(start_end_times)  # (N, 2)
    start_end_idxs = np.rint((start_end_times / duration) * num_units)
    start_end_idxs = np.clip(start_end_idxs, 0, num_units - 1).astype(int)
    return start_end_idxs


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time

def vfeat_index_to_time(start_index, end_index, num_units, duration, vfeat_idx):
    start_index = vfeat_idx[:-1][start_index]
    end_index = vfeat_idx[1:][end_index]
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time

def pad_seq(sequences, pad_tok=None, max_length=None):
    if pad_tok is None:
        pad_tok = 0  # 0: "PAD" for words and chars, "PAD" for tags
    if max_length is None:
        max_length = max([len(seq) for seq in sequences])
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))
    return sequence_padded, sequence_length


def pad_char_seq(sequences, max_length=None, max_length_2=None):
    sequence_padded, sequence_length = [], []
    if max_length is None:
        max_length = max(map(lambda x: len(x), sequences))
    if max_length_2 is None:
        max_length_2 = max([max(map(lambda x: len(x), seq)) for seq in sequences])
    for seq in sequences:
        sp, sl = pad_seq(seq, max_length=max_length_2)
        sequence_padded.append(sp)
        sequence_length.append(sl)
    sequence_padded, _ = pad_seq(
        sequence_padded, pad_tok=[0] * max_length_2, max_length=max_length
    )
    sequence_length, _ = pad_seq(sequence_length, max_length=max_length)
    return sequence_padded, sequence_length


def pad_video_seq(sequences, max_length=None):
    if max_length is None:
        max_length = max([vfeat.shape[0] for vfeat in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_length - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = np.zeros(shape=[add_length, feature_length], dtype=np.float32)
            seq_ = np.concatenate([seq, add_feature], axis=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length
