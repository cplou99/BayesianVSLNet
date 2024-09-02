import copy
import glob
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm
from scipy.stats import binom, norm
import utils.evaluate_ego4d_nlq as ego4d_eval
from utils.data_util import index_to_time, vfeat_index_to_time, batched_time_to_index
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix="t7", max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split("_")[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix="t7"):
    model_filenames = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split("_")[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def eval_test(
    model,
    data_loader,
    device,
    mode="test",
    result_save_path=None,
    gt_json_path="",
    epoch=None,
    global_step=None,
    return_results_dict=False,
):
    gt_json_path = gt_json_path if gt_json_path != "" else None
    predictions = []
    logits = {}
    with torch.no_grad():
        for idx, (records, vfeats, vfeat_lens, vfeat_idxs, word_ids, char_ids) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="evaluate {}".format(mode),
        ):

            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)

            if isinstance(word_ids, dict):
                word_ids = {key: val.to(device) for key, val in word_ids.items()}
                # generate mask
                query_mask = (
                    (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                    .float()
                    .to(device)
                )
            else:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                # generate mask
                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)

            # generate mask
            video_mask = convert_length_to_mask(vfeat_lens).to(device)

            # compute predicted results
            probs = model(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )


            vfeat_lens = vfeat_lens.cpu().detach().tolist()
            all_probs = probs.cpu().detach().tolist()
            all_probs = [probs[:vfeat_len] for probs, vfeat_len in zip(all_probs, vfeat_lens)]
            
            num_times = [len(record['queries_idxs']) for record in records]
            
            num_text_queries = [record['num_text_queries'] for record in records]
            text_query_pos = [record['query_idx'] for record in records]
            bin_p = [(record['query_idx']+1)/record['num_text_queries'] for record in records]
            
            prior_probs = [norm.pdf(np.arange(0, record['v_len']), bin_p[bs]*record['v_len'], int(record['v_len']/10)) for bs, record in enumerate(records)]
            prior_probs = [prob/max(prob) for prob in prior_probs]
            
            bayesian_probs = []
            for bs, record in enumerate(records):
                new_probs = prior_probs[bs]*all_probs[bs]
                bayesian_probs.append(new_probs.tolist())
            
            all_nn_probs = all_probs
            
            # Modified in version3
            num_times = [1 for record in records]
            start_indices, end_indices = model.extract_index_from_probs(bayesian_probs, num_times=num_times, num_text_queries=num_text_queries, topk=5) # The output should be two lists of length (batch_size, num_times, topk)
            

            # Record output and use standard evalution script for NLQ.
            for record, vfeat_idx, starts, ends, probs, nn_probs in zip(records, vfeat_idxs, start_indices, end_indices, all_probs, all_nn_probs):
                # Convert all indices to times.

                if record["vid"] not in logits.keys():
                    logits[record["vid"]] = {"duration": record["duration"], "v_len": record["v_len"], "text_queries": {}}
                
                # Get the text query of the current prediction
                queries_idxs = record["queries_idxs"]
                num_queries = record["num_id_queries"]
                curr_query_idx = record["query_idx"]
                idx_idx = queries_idxs.index(curr_query_idx)
        

                start = starts[0]
                end = ends[0]

            
                timewindow_predictions = []
                start_times = []
                end_times = []
                
                if len(vfeat_idx) == record["v_original_len"] or len(vfeat_idx) == record["v_len"]:
                    start = [vfeat_idx[s] if s<len(vfeat_idx) else vfeat_idx[-1] for s in start]
                    end = [vfeat_idx[e] if e<len(vfeat_idx) else vfeat_idx[-1] for e in end]
                else:
                    start = vfeat_idx[:-1][start]
                    end = vfeat_idx[1:][end]
                
                start = [s if s<record["v_original_len"] else record["v_original_len"]-1 for s in start]
                end = [e if e<record["v_original_len"] else record["v_original_len"]-1 for e in end]

                start_time, end_time = index_to_time(
                    start, end, record["v_original_len"], record["duration"]
                )
                
                # Iterate over the topk predictions 
                for s_time, e_time in zip(start_time, end_time):
                    timewindow_predictions.append([float(s_time), float(e_time)])
                    start_times.append(s_time)
                    end_times.append(e_time)

                new_datum = {
                    "clip_uid": record["vid"],
                    "annotation_uid": record["annotation_uid"],
                    "query_idx": int(record["query_idx"]),
                    "predicted_times": copy.deepcopy(timewindow_predictions),
                }
                predictions.append(new_datum)
                
                if True:
                    logits[record["vid"]]["text_queries"][record["query_idx"]] = {
                        "query": record["query"],
                        "probs_pred": probs,
                        "start_index_pred": starts[0],
                        "end_index_pred": ends[0],

                        "start_times_pred": start_times,
                        "end_times_pred": end_times,
                        "start_times_gt": record["s_time"],
                        "end_times_gt": record["e_time"],
                        "start_index_gt": record["s_indxs"],
                        "end_index_gt": record["e_indxs"],
                        "v_original_len": record["v_original_len"],
                        "v_len": record["v_len"],
                        "vfeat_idx": vfeat_idx,
                    }


    # Save predictions if path is provided.
    if result_save_path:
        with open(result_save_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": predictions,
                },
                file_id,
            )

    # Evaluate if ground truth JSON file is provided.
    if gt_json_path:
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        title = f"Epoch {epoch}, Step {global_step}"
        display_results = ego4d_eval.display_results(
            results, mIoU, thresholds, topK, title=title
        )
        if return_results_dict:
            results_dict = ego4d_eval.get_results_dict(results, mIoU, thresholds, topK)
        else:
            results_dict = None
    else:
        results = None
        mIoU = None
        display_results = None
        results_dict = None
    if results_dict is not None:
        display_results = (display_results, results_dict)
    
    
    return results, mIoU, display_results

