# FovealSeg with classification
## Environment Setup
```bash
pip install -r requirements.txt
```

## Data preparation
1. Generated LVIS data (50 classes, sp20000 for train, sp4000 for valid)  can be reused. The data structure should be the same as in DynamicFoucs
````bash
b_data_train
├── data_a_raw
│   ├── lvis_v1_train
│   ├── lvis_v1_val
│   └── coco2017
├── data_b_cache
├── data_c_cook
│   └── lvis
│       ├── train
│       │   └── sp20000
│       └── valid
│           └── sp4000
````

2. Set up the paths in ```/DynamicFocus/preset.py```. Point ```dpath_data_raw```, ```dpath_data_cache```, ```dpath_data_cook``` to the paths in step 1.

3. If need to generate new data, 
```bash
cd DynamicFocus
python e_preprocess_scripts/b2_preprocess_lvis.py --task preprocess --dataset_partition train valid --sample_num 20000
```

## Run command
### Evaluate the current ckpt on lvis 50 classes data
1. ckpt address: https://drive.google.com/drive/folders/1sxYLCFNCaei7IbXFGQ69lWGE3TSEgraz?usp=sharing
download lvis_50cls_ckpt.zip and unzip

2. Copy ckpt
```bash
mkdir ckpt
copy -r lvis_50cls ./ckpt/lvis_50cls
```

3. Run
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_deform_semantic.py --gpus 0-1 --cfg config/deform-cityscape.yaml DATASET.root_dataset '/home/xth/Deformation-Segmentation/Deformation-Segmentation/data/cityscapes' TRAIN.task_input_size '(80,80)' DIR "./ckpt/lvis_50cls" TRAIN.deform_joint_loss True TRAIN.epoch_iters 745 TRAIN.checkpoint_per_epoch 20 VAL.no_upsample True TRAIN.num_epoch 121 TRAIN.start_epoch 120 TRAIN.eval_per_epoch 1 TRAIN.skip_train_for_eval True VAL.no_upsample True DATASET.dataset_marker_train 'sp20000' DATASET.dataset_marker_valid 'sp4000' MODEL.gaussian_radius 45 TRAIN.saliency_input_size '(80, 80)'
```

<!-- ## Data preparation
1. Download the [Cityscapes](https://www.cityscapes-dataset.com/), [DeepGlobe](https://competitions.codalab.org/competitions/18468) and [PCa-histo](to-be-released) datasets.

2. Your directory tree should be look like this:
````bash
$SEG_ROOT/data
├── cityscapes
│   ├── annotations
│   │   ├── testing
│   │   ├── training
│   │   └── validation
│   └── images
│       ├── testing
│       ├── training
│       └── validation
├── histomri
│   ├── train
│   │   ├── images
│   │   ├── labels
│   └── val
│   │   ├── images
│   │   ├── labels
├── deepglob
│   ├── land-train
│   └── land_train_gt_processed
````
note Histo_MRI is the PCa-histo dataset

3. Data list .odgt files are provided in ```./data``` prepare correspondingly for local datasets. (Note: for cityscapes please check its ```./data/Cityscape/*.odgt```, in my example I removed the city subfolders and put all images under one folder, if your data tree is different please modify accordingly
```e.g. change "images/training/tubingen_000025_000019_leftImg8bit.png" to "images/training/tubingen/000025_000019_leftImg8bit.png"```


## Reproduce
full configuration bash provided to reproduced paper results, suitable for large scale experiment in multiple GPU Environment, Syncronized Batch Normalization are deployed.

### Training
Train a model by selecting the GPUs (```$GPUS```) and configuration file (```$CFG```) to use. During training, last checkpoints by default are saved in folder ```ckpt```.
```bash
python3 train_deform.py --gpus $GPUS --cfg $CFG
```
- To choose which gpus to use, you can either do ```--gpus 0-7```, or ```--gpus 0,2,4,6```.

* Bashes and configurations are provided to reproduce our results:

- note you will need to specify your root path 'SEG_ROOT' for ```DATASET.root_dataset``` option in those scripts.

```bash
bash quick_start_bash/cityscape_64_128_ours.sh
bash quick_start_bash/cityscape_64_128_uniform.sh
bash quick_start_bash/deepglob_300_300_ours.sh
bash quick_start_bash/deepglob_300_300_uniform.sh
bash quick_start_bash/pcahisto_80_800_ours.sh
bash quick_start_bash/pcahisto_80_800_uniform.sh
```

* You can also override options in commandline, for example  ```python3 train_deform.py TRAIN.num_epoch 10 ```.


### Evaluation
1. Evaluate a trained model on the validation set, simply override following options ```TRAIN.start_epoch 125 TRAIN.num_epoch 126 TRAIN.eval_per_epoch 1 TRAIN.skip_train_for_eval True```

* Alternatively, you can quick start with provided bash script:
```bash
bash quick_start_bash/eval/cityscape_64_128_ours.sh
bash quick_start_bash/eval/cityscape_64_128_uniform.sh
bash quick_start_bash/eval/deepglob_300_300_ours.sh
bash quick_start_bash/eval/deepglob_300_300_uniform.sh
bash quick_start_bash/eval/pcahisto_80_800_ours.sh
bash quick_start_bash/eval/pcahisto_80_800_uniform.sh
```

## Citation
If you use this code for your research, please cite our paper:

```
@article{jin2021learning,
  title={Learning to Downsample for Segmentation of Ultra-High Resolution Images},
  author={Jin, Chen and Tanno, Ryutaro and Mertzanidou, Thomy and Panagiotaki, Eleftheria and Alexander, Daniel C},
  journal={arXiv preprint arXiv:2109.11071},
  year={2021}

@inproceedings{
jin2022learning,
title={Learning to Downsample for Segmentation of Ultra-High Resolution Images},
author={Chen Jin and Ryutaro Tanno and Thomy Mertzanidou and Eleftheria Panagiotaki and Daniel C. Alexander},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=HndgQudNb91}
}
``` -->
