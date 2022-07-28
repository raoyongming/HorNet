# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    '../_base_/models/upernet_hornet.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)

model = dict(
    pretrained='pretrained/hornet_large_gf_in22k.pth',
    backbone=dict(
        depths=[2, 3, 18, 2], 
        base_dim=192,
        gnconv=[
            'partial(gnconv, order=2, s=1/3)',
            'partial(gnconv, order=3, s=1/3)',
            'partial(gnconv, order=4, s=1/3, h=24, w=13, gflayer=GlobalLocalFilter)',
            'partial(gnconv, order=5, s=1/3, h=12, w=7, gflayer=GlobalLocalFilter)',
        ],
        drop_path_rate=0.4,
        out_indices=[0, 1, 2, 3],
        use_checkpoint=True,
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536], 
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=150
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
)

optimizer = dict(constructor='LearningRateDecayOptimizerConstructorHorNet', _delete_=True, type='AdamW', 
                 lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg={'decay_rate': 0.8,
                                'decay_type': 'stage_wise',
                                'num_layers': 12})

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))

runner = dict(type='IterBasedRunner')

# do not use mmdet version fp16
fp16 = None
