#!/usr/bin/env bash

#python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 2  --with_box_refine  --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10
python3 main_track.py  --output_dir output/output_lr_1e-5/matcherpp3 --dataset_file mot --coco_path mot --batch_size 1  --with_box_refine  --num_queries 500 --resume crowdhuman_final.pth --epochs 20 --lr_drop 10

python3 main_track.py  --output_dir output_mot17/HR5all --dataset_file mot --coco_path mot --batch_size 1  --with_box_refine --resume crowdhuman_hybrid_branch.pth --epochs 20 --lr_drop 10  --track_train_split train  --dim_feedforward 2048  --num_queries_one2one 500 --num_queries_one2many 1000  --k_one2many 5  --lambda_one2many 0.5  --drop 0.0  --look_forward_twice
