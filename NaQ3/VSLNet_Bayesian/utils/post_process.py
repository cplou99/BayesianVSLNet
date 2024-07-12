import json

dir_file = '/disk/TAS/ego4d-goalstep/step_grounding/experiments/vslnet/goalstep/~WEEK5/checkpoints/'
json_file = f'{dir_file}/vslnet_ego4d_goalstep_EgoVLPv2_video_NO_head_Omnivore_512_128_RObertA_EgoVLPv2_uniform/model/vslnet_29440_val_result.json'

with open(json_file, 'r') as f:
    data = json.load(f)

gt_annotations_path = '/disk/TAS/ego4d-goalstep/step_grounding/data/dataset/ego4d_goalstep/val.json'
annotations = json.load(open(gt_annotations_path, 'r'))

for vid in annotations.keys():
    data_vid = data.query(f'video_id == "{vid}"')