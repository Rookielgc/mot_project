#!/usr/bin/env bash

#python3 main_track.py  --output_dir val/val_9.5e-5 --dataset_file mot --coco_path mot --batch_size 1 --resume output/output_lr_9.5e-5/checkpoint.pth --eval --with_box_refine --num_queries 500
python3 main_track.py  --output_dir val_lr1e-4/matcherpp5 --dataset_file mot --coco_path mot --batch_size 1 --resume output/output_lr_1e-5/matcherpp5/checkpoint.pth --eval --with_box_refine --num_queries 500

# Inference
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_track.py  \
--dataset_file mot \
--coco_path mot \
--batch_size 1 \
--resume outputs/transtrack-hybrid-branch/r50_hybrid_branch_lambda0.5_group5_t1000_dp0_lft_transtrack_for_val/checkpoint.pth \
--eval \
--with_box_refine \
--num_queries_one2one 500 \
--dim_feedforward 2048 \
--track_eval_split val \
--dist_video \
--output_dir outputs/transtrack-hybrid-branch/r50_hybrid_branch_lambda0.5_group5_t1000_dp0_lft_transtrack_for_val \
--track_thresh 0.3

python3 main_track.py --output_dir val_lr1e-4/HR --dataset_file mot --coco_path mot --batch_size 1 --resume output/output_lr_1e-4/HR/checkpoint.pth --eval --with_box_refine --num_queries_one2one 500  --dim_feedforward 2048  --track_eval_split val --track_thresh 0.3
