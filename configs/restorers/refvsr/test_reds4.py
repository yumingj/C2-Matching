exp_name = 'exp015_refnasicvsr_lr_feat_convattn_large_reds4'

# model settings
model = dict(
    type='RefBasicVSR',
    generator=dict(
        type='RefBasicVSRLRFeatConvAttnLargeNet',
        mid_channels=64,
        num_blocks=30,
        groups=8,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        extractor_pretrained=
        './pretrained_model/feature_extraction.pth'
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRREDSMultipleGTRefDataset'
val_dataset_type = 'SRREDSMultipleGTRefDataset'
test_dataset_type = 'SRTestMultipleGTRefDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='TemporalReverse', keys='lq_path', reverse_ratio=0),
    dict(type='GetKeyFramePath', ref_from_current_frames=False),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='keyframe',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'keyframe']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='CropRefImage', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['lq', 'gt', 'keyframe'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['lq', 'gt', 'keyframe'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(
        type='RandomTransposeHW',
        keys=['lq', 'gt', 'keyframe'],
        transpose_ratio=0.5),
    dict(type='GenerateUpImages', redownsample=True),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'up']),
    dict(type='ImageToTensor', keys=['keyframe']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'up', 'keyframe'],
        meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='GetKeyFramePath', ref_from_current_frames=False),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='keyframe',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt', 'keyframe']),
    dict(type='GenerateUpImages', redownsample=True),
    dict(type='FramesToTensor', keys=['lq', 'gt', 'up']),
    dict(type='ImageToTensor', keys=['keyframe']),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'up', 'keyframe'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder=
            'path-to-REDS/REDS/train_sharp_bicubic/X4',
            gt_folder='path-to-REDS/REDS/train_sharp',
            ref_folder='path-to-REDS/reds_orig',
            num_input_frames=15,
            pipeline=train_pipeline,
            scale=4,
            val_partition='REDS4',
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder=
        'path-to-REDS/REDS/train_sharp_bicubic/X4',
        gt_folder='path-to-REDS/REDS/train_sharp',
        ref_folder='path-to-REDS/reds_orig',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='path-to-REDS/REDS/train/train_sharp_bicubic/X4',
        gt_folder='path-to-REDS/REDS/train/train_sharp',
        ref_folder='path-to-REDS/reds_orig',
        num_input_frames=100,
        pipeline=test_pipeline,
        scale=4,
        val_partition='REDS4',
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={
            'spynet': dict(lr_mult=0.125),
        })))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=5000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
