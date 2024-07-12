#!/bin/bash
function rand(){
    min=$1
    max=$(($2-$min+1))
    num=$(($RANDOM+1000000000))
    echo $(($num%$max+$min))
}
export MAIN_PORT=$(rand 1024 2048)
export MAIN_PORT_TCL=$(rand 1024 2048)

EXPT_ROOT=$1

mkdir -p $EXPT_ROOT

FEAT_DIM=2304
FV="EgoVLPv2_video_NO_head"  
OMNIVORE=1
SAMPLING="uniform"


CUDA_VISIBLE_DEVICES="0" python -W ignore -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port $MAIN_PORT_TCL \
    ../../NaQ/VSLNet_Bayesian/main.py \
        --task ego4d_goalstep \
        --predictor RObertA_EgoVLPv2 \
        --dim 128 \
        --mode train \
        --video_feature_dim $FEAT_DIM \
        --max_pos_len 512 \
        --epochs 100 \
        --fv $FV \
        --num_workers 4 \
        --model_dir $EXPT_ROOT/checkpoints \
        --eval_gt_json data/annotations/val.json \
        --log_to_tensorboard "nlq" \
        --tb_log_dir $EXPT_ROOT/runs \
        --remove_empty_queries_from train val \
        --batch_size 32 \
        --init_lr 0.001 \
        --gradual_mlp 0 \
        --load_omnivore $OMNIVORE \
        --sampling $SAMPLING \