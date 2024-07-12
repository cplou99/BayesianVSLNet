EXPT_ROOT=$1

FEAT_DIM=2304
FV="EgoVLPv2_video_NO_head"  
OMNIVORE=1
# --mode test \
# --eval_gt_json data/annotations/test.json \

CUDA_VISIBLE_DEVICES="0 1" python ../../NaQ/VSLNet_Bayesian/main.py \
 --task ego4d_goalstep \
 --predictor RObertA_EgoVLPv2 \
 --mode val \
 --video_feature_dim $FEAT_DIM \
 --max_pos_len 512 \
 --fv $FV \
 --model_dir $EXPT_ROOT/checkpoints \
 --eval_gt_json data/annotations/val.json \
 --gradual_mlp 0 \
 --load_omnivore $OMNIVORE \