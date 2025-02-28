## Hybrid Matching with Box-Aware Refinement: Enhancing TransTrack for Robust Multi-Pedestrian Tracking


## MOT challenge
Dataset | MOTA% | IDF1% | HOTA% | MT% | ML% |  FP | FN | IDS 
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MOT17 | 74.7 | 65.6| 54.7 | 48.7 | 10.7 | 33906 | 104715 | 4320


## Validation set

Dataset | Training time | MOTA% | HOTA% | FP% | FN% | IDs%  
:---:|:---:|:---:|:---:|:---:|:---:|:---:
crowdhuman, mot17_half |  ~45h + 1h  | 68.1 | 3.0  | 28.2 | 0.6 | 
crowdhuman, mot20_half |  ~45h       | 71.7 | 5.6 | 1.7 | 1.5 | 


#### Notes
- We observe about 1 MOTA noise.
- If the resulting MOTA of your self-trained model is not desired, playing around with the --track_thresh sometimes gives a better performance.
- The default track_thresh is 0.4, except for 0.5 in crowdhuman.
- We use the models pre-trained on imagenet.
- (crowdhuman, mot17_half) is first training on crowdhuman, then fine-tuning on mot17_half.


## Demo
<img src="assets/MOT17-11.gif" width="400"/>  <img src="assets/MOT17-04.gif" width="400"/>


## Installation
The codebases are built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CenterTrack](https://github.com/xingyizhou/CenterTrack) and [Enhancing Your Trained DETRs with Box Refinement](https://github.com/YiqunChen1999/RefineBox.).

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- detrex, detectron2
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone https://github.com/Rookielgc/mot_project.git
cd mot_project
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare datasets and annotations
```
mkdir crowdhuman
cp -r /path_to_crowdhuman_dataset/CrowdHuman_train crowdhuman/CrowdHuman_train
cp -r /path_to_crowdhuman_dataset/CrowdHuman_val crowdhuman/CrowdHuman_val
mkdir mot
cp -r /path_to_mot_dataset/train mot/train
cp -r /path_to_mot_dataset/test mot/test
```
CrowdHuman dataset is available in [CrowdHuman](https://www.crowdhuman.org/). 
```
python3 track_tools/convert_crowdhuman_to_coco.py
```
MOT dataset is available in [MOT](https://motchallenge.net/).
```
python3 track_tools/convert_mot_to_coco.py
```

3. Pre-train on crowdhuman
```
sh track_exps/crowdhuman_train.sh
python3 track_tools/crowdhuman_model_to_mot.py
```
The pre-trained model is available [crowdhuman_final.pth]([https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing](https://github.com/HDETR/H-TransTrack/releases/download/v1.0.1/crowdhuman_hybrid_branch.pth)).

4. Train HMBR-Trans
```
sh track_exps/crowdhuman_mot_trainhalf.sh
```

5. Evaluate HMBR-Trans
```
sh track_exps/mot_val.sh
sh track_exps/mota.sh
```

6. Visualize HMBR-Trans
```
python3 track_tools/txt2video.py
```


## Test set
Dataset | MOTA% | IDF1% | HOTA% | MT% | ML% |  FP | FN | IDS 
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MOT17 | 74.7 | 65.6| 54.7 | 48.7 | 10.7 | 33906 | 104715 | 4320

#### Notes
- Performance on test set is evaluated by [MOT challenge](https://motchallenge.net/).
- (crowdhuman + mot17) is training on mixture of crowdhuman and mot17.
- We won't release trained models for test test. Running as in Steps could reproduce them. 
 
#### Steps
1. Train HMBR-Trans
```
sh track_exps/crowdhuman_mot_train.sh
```

or

1. Mix crowdhuman and mot17
```
mkdir -p mix20/annotations
cp mot20/annotations/val_half.json mix20/annotations/val_half.json
cp mot20/annotations/test.json mix20/annotations/test.json
cd mix
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
cd ..
python3 track_tools/mix_data20.py
```
2. Train HMBR-Trans
```
sh track_exps/crowdhuman_plus_mot_train.sh
```

## Citation
```bibtex
@article{jia2022detrs,
  title={DETRs with Hybrid Matching},
  author={Jia, Ding and Yuan, Yuhui and He, Haodi and Wu, Xiaopei and Yu, Haojun and Lin, Weihong and Sun, Lei and Zhang, Chao and Hu, Han},
  journal={arXiv preprint arXiv:2207.13080},
  year={2022}
}

@article{sun2020transtrack,
  title={Transtrack: Multiple object tracking with transformer},
  author={Sun, Peize and Cao, Jinkun and Jiang, Yi and Zhang, Rufeng and Xie, Enze and Yuan, Zehuan and Wang, Changhu and Luo, Ping},
  journal={arXiv preprint arXiv:2012.15460},
  year={2020}
}

@misc{chen2023enhancing,
      title={Enhancing Your Trained DETRs with Box Refinement}, 
      author={Yiqun Chen and Qiang Chen and Peize Sun and Shoufa Chen and Jingdong Wang and Jian Cheng},
      year={2023},
      eprint={2307.11828},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

