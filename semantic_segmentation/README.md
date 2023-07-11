# ADE20k Semantic segmentation with HorNet

 Please refer to [README.md](https://github.com/facebookresearch/ConvNeXt/tree/main/semantic_segmentation) for installation and dataset preparation instructions.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|
| HorNet-T (7x7) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/1ca970586c6043709a3f/?dl=1) | UPerNet | 512x512 | 160K | 48.1 | 48.9 | 52M | 926G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/55136f1d57074dcb894e/?dl=1) |
| HorNet-T (GF) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/511faad0bde94dfcaa54/?dl=1) | UPerNet | 512x512 | 160K | 49.2 | 49.3 | 55M | 924G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/5e1a6d01cd824f3b95f4/?dl=1) |
| HorNet-S (7x7) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/46422799db2941f7b684/?dl=1) | UPerNet | 512x512 | 160K | 49.2 | 49.8 | 81M | 1030G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/75aca9a8a67b4fd591ca/?dl=1) |
| HorNet-S (GF) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8405c984bf084d2ba85a/?dl=1) | UPerNet | 512x512 | 160K | 50.0 | 50.5 | 85M | 1027G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/3e426d1b6f994907bb2c/?dl=1) |
| HorNet-B (7x7) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/5c86cb3d655d4c17a959/?dl=1) | UPerNet | 512x512 | 160K | 50.0 | 50.5 | 121M | 1174G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/543c70ce127440879374/?dl=1) |
| HorNet-B (GF) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/6c84935e63b547f383fb/?dl=1) | UPerNet | 640x640 | 160K | 50.5 | 50.9 | 126M | 1171G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/dd1ae599a8b3467d8dba/?dl=1) |
| HorNet-L (7x7) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/4de41e26cb254c28a61a/?dl=1) | UPerNet | 640x640 | 160K | 54.1 | 54.5 | 232M | 2473G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a76f7ce114ad4e9c8ffb/?dl=1) |
| HorNet-L (GF) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/8679b6acf63c41e285d9/?dl=1) | UPerNet | 640x640 | 160K | 55.0 | 55.2 | 239M | 2465G | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/9253bd0d50c84ab1b71e/?dl=1) |

## Training

```
bash dist_train.sh <CONFIG_PATH> <NUM_GPUS> --work-dir <SAVE_PATH> --options model.pretrained=<PRETRAIN_MODEL>
```

For example, using a `HorNet-T (GF)` backbone with UperNet:
```bash
bash dist_train.sh \
    configs/hornet/upernet_hornet_tiny_gf_512_160k_ade20k.py 8 \
    --work-dir /path/to/save \
    --options model.pretrained=/path/to/pretrained/weight
```

More config files can be found at [`configs/hornet`](configs/hornet).


## Evaluation

Command format for multi-scale testing:
```
bash dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a `HorNet-T (GF)` backbone with UperNet:
```bash
bash dist_test.sh configs/hornet/upernet_hornet_tiny_gf_512_160k_ade20k.py \ 
    /path/to/checkpoint 8 --eval mIoU --aug-test
```

Command format for single-scale testing:
```
bash dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a `HorNet-T (GF)` backbone with UperNet:
```bash
bash dist_test.sh configs/hornet/upernet_hornet_tiny_gf_512_160k_ade20k.py \ 
    /path/to/checkpoint 8 --eval mIoU
```

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
