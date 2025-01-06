#!/usr/bin/env bash

#python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_track.py  --output_dir ./output --dataset_file mot --coco_path mot --batch_size 2  --with_box_refine  --num_queries 500 --epochs 150 --lr_drop 100
python3 main_track.py  --output_dir output_lr1e-4/matcher3 --dataset_file mot --coco_path mot --batch_size 1  --with_box_refine  --num_queries 500 --epochs 150 --lr_drop 100
