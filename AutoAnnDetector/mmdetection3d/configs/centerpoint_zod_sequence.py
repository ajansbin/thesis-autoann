point_cloud_range = [-51.2, 0, -5.0, 51.2, 204.8, 3.0]
class_names = ["Vehicle", "VulnerableVehicle", "Pedestrian"]
dataset_type = "ZodSequenceDataset"
data_root = "data/zod/"
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False
)
file_client_args = dict(backend="disk")
train_pipeline = [
    dict(
        type="ZodLoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
]
test_pipeline = [
    dict(
        type="ZodLoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(type="Collect3D", keys=["points"]),
        ],
    ),
]
eval_pipeline = [
    dict(
        type="ZodLoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="LoadImageFromFile"),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="Collect3D", keys=["points"]),
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type="RepeatDataset",
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file="storage/detection/CenterPoint/zod_seq_full/zod-seq-full_infos_train.pkl",
            pipeline=[
                dict(
                    type="ZodLoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=5,
                    use_dim=5,
                    file_client_args=file_client_args,
                ),
                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
                dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
                dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
                dict(type="ObjectNameFilter", classes=class_names),
                dict(type="PointShuffle"),
                dict(type="DefaultFormatBundle3D", class_names=class_names),
                dict(type="Collect3D", keys=["points", "gt_bboxes_3d", "gt_labels_3d"]),
            ],
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="storage/detection/CenterPointzod_seq_full/zod-seq-full_infos_val.pkl",
        pipeline=[
            dict(
                type="ZodLoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args,
            ),
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug3D",
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type="DefaultFormatBundle3D",
                        class_names=class_names,
                        with_label=False,
                    ),
                    dict(type="Collect3D", keys=["points"]),
                ],
            ),
        ],
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="storage/detection/CenterPoint/zod_seq_full/zod-seq-full_infos_val.pkl",
        pipeline=[
            dict(
                type="ZodLoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args,
            ),
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug3D",
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type="DefaultFormatBundle3D",
                        class_names=class_names,
                        with_label=False,
                    ),
                    dict(type="Collect3D", keys=["points"]),
                ],
            ),
        ],
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type="ZodLoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=5,
            use_dim=5,
            file_client_args=file_client_args,
        ),
        dict(type="LoadImageFromFile"),
        dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
        dict(type="Collect3D", keys=["points"]),
    ],
)
voxel_size = [0.2, 0.2, 8]
model = dict(
    type="CenterPoint",
    pts_voxel_layer=dict(
        max_num_points=20,
        voxel_size=voxel_size,
        max_voxels=(60000, 40000),
        point_cloud_range=point_cloud_range,
    ),
    pts_voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type="BN1d", eps=0.001, momentum=0.01),
        legacy=False,
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter", in_channels=64, output_shape=(1024, 512)
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type="BN", eps=0.001, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        use_conv_for_no_stride=True,
    ),
    pts_bbox_head=dict(
        type="CenterHead",
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=["Vehicle"]),
            dict(num_class=1, class_names=["VulnerableVehicle"]),
            dict(num_class=1, class_names=["Pedestrian"]),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            post_center_range=[-61.2, -10.0, -10.0, 61.2, 224.8, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=[0.2, 0.2],
            code_size=7,
            pc_range=[-51.2, 0],
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 1024, 1],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range,
        )
    ),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -10.0, -10.0, 61.2, 224.8, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            pc_range=[-51.2, 0],
            out_size_factor=4,
            voxel_size=[0.2, 0.2],
            nms_type="rotate",
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
        )
    ),
)
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="cyclic", target_ratio=(10, 0.0001), cyclic_times=1, step_ratio_up=0.4
)
momentum_config = dict(
    policy="cyclic",
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)
runner = dict(type="EpochBasedRunner", max_epochs=20)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="MMDet3DWandbHook",
            init_kwargs=dict(
                project="centerpoint_od", name="zod_seq_full_val_multiclass_pcr200"
            ),
            log_checkpoint=False,
            log_checkpoint_metadata=False,
            num_eval_images=1,
            bbox_score_thr=0.1,
            visualize_3d=True,
            visualize_img=True,
            max_points=20000,
        ),
    ],
)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = "./work_dirs/centerpoint_zod"
load_from = None
resume_from = None
workflow = [("train", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
gpu_ids = range(0, 1)
