# COCO Object detection with HorNet

## Getting started 

 Please refer to [README.md](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/6a979e2164e3fb0de0ca2546545013a4d71b2f7d/README.md) for installation and dataset preparation instructions.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Lr Schd | box mAP | mask mAP | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|
| HorNet-T (7x7)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/1ca970586c6043709a3f/?dl=1) | Cascade Mask R-CNN | 3x | 51.7 | 44.8 | 80M | 730G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/98fe75fe5d544c409852/?dl=1) |
| HorNet-T (GF)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/511faad0bde94dfcaa54/?dl=1) | Cascade Mask R-CNN | 3x | 52.4 | 45.6 | 80M | 728G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c6cbd11caf4449b49265/?dl=1) |
| HorNet-S (7x7)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/46422799db2941f7b684/?dl=1) | Cascade Mask R-CNN | 3x | 52.7 | 45.6 | 107M | 830G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8da73119497745f39c3f/?dl=1) |
| HorNet-S (GF)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8405c984bf084d2ba85a/?dl=1) | Cascade Mask R-CNN | 3x | 53.3 | 46.3 | 108M | 827G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c20b1ee6ed55479ab56d/?dl=1) |
| HorNet-B (7x7)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/5c86cb3d655d4c17a959/?dl=1)| Cascade Mask R-CNN | 3x | 53.3 | 46.1 | 144M | 969G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/80537e4cbd53465fa0ec/?dl=1) |
| HorNet-B (GF)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6c84935e63b547f383fb/?dl=1) | Cascade Mask R-CNN | 3x | 54.0 | 46.9 | 146M | 965G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/d0c0027d31e144aaa260/?dl=1) |
| HorNet-L (7x7)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4de41e26cb254c28a61a/?dl=1) | Cascade Mask R-CNN | 3x | 55.4 | 48.0 | 251M | 1363G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a76f7ce114ad4e9c8ffb/?dl=1) |
| HorNet-L (GF)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8679b6acf63c41e285d9/?dl=1) | Cascade Mask R-CNN | 3x | 56.0 | 48.6 | 259M | 1358G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/9253bd0d50c84ab1b71e/?dl=1) |


### Training

To train a detector with pre-trained models, run:
```
# multi-gpu training
bash dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `HorNet-T (GF)` backbone and 8 gpus, run:
```
bash dist_train.sh configs/hornet/cascade_mask_rcnn_hornet_tiny_gf_3x_coco_in1k.py 8 --cfg-options model.pretrained=/path/to/pretrained
```

More config files can be found at [`configs/hornet`](configs/hornet).

### Inference
```
# multi-gpu testing
bash dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

## Acknowledgment 

This code is built using [mmdetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
