import pandas as pd
import json
import ast
import tqdm
import torch
import os
import argparse
import shutil

def run(src_dir, dst_dir, annot_dir):

    video_uids = set()
    for split in ['train', 'valid']:
        json_fl = f'{annot_dir}/goalstep_{split}.json'
        annotations = json.load(open(json_fl))['videos']
        video_uids |= set([annot['video_uid'] for annot in annotations])

    gvideo_uids = set([video_uid for video_uid in video_uids if video_uid.startswith('grp-')])
    print('gvideo_uids', gvideo_uids, len(gvideo_uids))
    video_uids = video_uids - gvideo_uids

    feature_shapes = {}


    # symlink features for videos if they already exist
    for video_uid in tqdm.tqdm(video_uids):
        
        src = f'{src_dir}/{video_uid}.pt' 
        dst = f'{dst_dir}/{video_uid}.pt'

        if not os.path.exists(src):
            print(src)
            raise Exception(f'{video_uid} video feature missing. Check if all features are downloaded?')

        if os.path.exists(dst):
            os.remove(dst)
        
        # Changed by me #####
        # os.symlink(src, dst)
        shutil.copyfile(src, dst)

        feature_shapes[video_uid] = torch.load(dst).shape[0]
        print('video_uid', video_uid, 'shape', feature_shapes[video_uid])

    # Merge features for grouped videos
    video_groups = pd.read_csv(f'{annot_dir}/goalstep_video_groups.tsv', delimiter='\t')
    for entry in tqdm.tqdm(video_groups.to_dict('records')):
        video_group = ast.literal_eval(entry['video_group'])
        gvideo_uid = f'grp-{video_group[0]}'
        print(gvideo_uid)

        if gvideo_uid not in gvideo_uids:
            continue

        #gfeats = []
        #for video_uid in video_group:
        #    print(f"{src_dir}/grp-{video_uid}.pt")
        #    feats = torch.load(f"{src_dir}/grp-{video_uid}.pt")
        #    gfeats.append(feats)
        #gfeats = torch.cat(gfeats, 0)

        src = f'{src_dir}/{gvideo_uid}.pt'
        dst = f'{dst_dir}/{gvideo_uid}.pt'

        shutil.copyfile(src, dst)
        gfeats = torch.load(dst)
        feature_shapes[gvideo_uid] = gfeats.shape[0]
        torch.save(gfeats, f'{dst_dir}/{gvideo_uid}.pt')


    json.dump(feature_shapes, open(f'{dst_dir}/feature_shapes.json', 'w'))
    print(feature_shapes)
    print(f'{dst_dir}/feature_shapes.json')
    print("Helloo")

def check_shapes():
    egovlp = '/home/ego_exo4d/TAS/ego4d-goalstep/step_grounding/data/features/EgoVLPv2_video'
    onmivore = '/home/ego_exo4d/TAS/ego4d-goalstep/step_grounding/data/features/omnivore_video_swinl'
    for f in os.listdir(egovlp):
        if f.endswith('.pt'):
            egovlp_shape = torch.load(os.path.join(egovlp, f)).shape[0]
            onmivore_shape = torch.load(os.path.join(onmivore, f)).shape[0]
            if egovlp_shape != onmivore_shape:
                print(f, egovlp_shape, onmivore_shape)
            egovlp_device = torch.load(os.path.join(egovlp, f)).device
            onmivore_device = torch.load(os.path.join(onmivore, f)).device
            print(f, "egovlp_device:", egovlp_device, "omnivore_device:", onmivore_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate GoalStep video features')
    parser.add_argument('--annot_dir', default='/disk/TAS/ego4d-goalstep/data', help='path that contains goalstep_video_groups.tsv and split jsons')
    #parser.add_argument('--feature_dir', default='./datasets/ego4d_goal_step/v2/EgoVLP_joined_per_video', help='path that contains Ego4D videos')
    parser.add_argument('--feature_dir', default='/disk/TAS/datasets/ego4d_goal_step/v2/EgoVLP_NO_head_joined_per_video', help='path that contains Ego4D videos')
    parser.add_argument('--out_dir', default='/disk/TAS/ego4d-goalstep/step_grounding/data/features/EgoVLPv2_video_NO_head')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    run(args.feature_dir, args.out_dir, args.annot_dir)
    #check_shapes()
