# detcon-pytorch

PyTorch implementation of the DetconB model from: ["Efficient Visual Pretraining with Contrastive Detection " Henaff et al. (ICCV 2021)](https://arxiv.org/abs/2103.10957)

Installation of [Apex](https://github.com/NVIDIA/apex) is required to enable DDP.

To log metrics to [wandb](https://github.com/wandb/client) switch to `enable_wandb:True` in `train_imagenet_300.yaml`

## Pretrained Weights

We release the pretrained detcon weights on ImageNet-1k for 300 epochs in original, torchvision and d2 format.

**Original** [[Download](https://drive.google.com/file/d/15a7jJ1XVmSVZVo0xFE4gDn1Uw2Mns9Ui/view?usp=share_link)]

Converted: **Torchvision** (MMSegmentation) [[Download](https://drive.google.com/file/d/1izUBGYX_3PkaurhP3bRK1EVhXKjOc5ep/view?usp=sharing)] **D2** (Detectron2) [[Download](https://drive.google.com/file/d/15a7jJ1XVmSVZVo0xFE4gDn1Uw2Mns9Ui/view?usp=share_link)]


For the detcon weights conversion we used initially [convert_fcn.py](https://github.com/KKallidromitis/detcon-pytorch/blob/main/utils/convert_fcn.py) for torchvision and then the official Detectron2 [convert-torchvision-to-d2.py](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py) for d2 format.


The evaluation baselines are as follows:

|         Metric         | Value  |
|------------------|---|
|  PASCAL VOC mIoU | 76.0 |
| Cityscapes mIoU  | 76.2  |
|    MS COCO $\text{AP}^{\text{bb}}$ | 41.5  |
|    MS COCO $\text{AP}^{\text{mk}}$ |  38.3 |

## Requirements
Before installing requirements.txt ensure the environment is updated with the correct [PyTorch](https://pytorch.org/) and Torchvision release

```
python>=3.9
pytorch>=1.10.0
torchvision>=0.11.0
joblib
scikit-image
matplotlib
opencv-python
tqdm
tensorflow
pyyaml
tensorboardx
wandb
pycocotools
classy_vision
```

First run ```gen_masks_tf.py``` for train and val seperately to generate FH masks according to the original [DetCon](https://github.com/deepmind/detcon).

Masks need to be generated in imagenet/masks path and have ```train_tf``` and ```val_tf``` names.
```bash
python gen_masks_tf.py --dataset_dir="/path/to/dataset/train" --output_dir="/path/to/dataset/masks" --mask_type="fh" --experiment_name="exp_train"
```

This repo uses `torch.distributed.launch` for pretraining:

```bash
python -m torch.distributed.launch --nproc_per_node=4--nnodes=32 --node_rank=0 --master_addr="" --master_port=12345 detconb_main.py --cfg={CONFIG_FILENAME}
```

## Dataset Structure

```none
imagenet
├── images
│   ├── train
│   │   ├── n01440764
│   │   ├── ...
│   │   ├── n15075141
│   ├── val
│   │   ├── n01440764
│   │   ├── ...
│   │   ├── n15075141
```

## Reproduce Results

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for PASCAL VOC and Cityscapes semantic segmentation. We use [detectron2](https://github.com/facebookresearch/detectron2) for MS COCO object detection and instance segmentation. The corresponding config can be found in `eval` folder.

## Acknowledgement

This repo is based on the BYOL implementation from Yao: https://github.com/yaox12/BYOL-PyTorch and K-Means implementation from Ali Hassani https://github.com/alihassanijr/TorchKMeans
