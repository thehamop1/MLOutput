┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --batch_size 28 --ckpt ../checkpoints/voxel_rcnn_car_8454.pth 
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/distributed/launch.py:188: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
2022-11-21 21:55:18,022   INFO  **********************Start logging**********************
2022-11-21 21:55:18,022   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 21:55:18,022   INFO  total_batch_size: 28
2022-11-21 21:55:18,022   INFO  cfg_file         cfgs/kitti_models/voxel_rcnn_car.yaml
2022-11-21 21:55:18,022   INFO  batch_size       14
2022-11-21 21:55:18,022   INFO  workers          4
2022-11-21 21:55:18,022   INFO  extra_tag        default
2022-11-21 21:55:18,022   INFO  ckpt             ../checkpoints/voxel_rcnn_car_8454.pth
2022-11-21 21:55:18,022   INFO  pretrained_model None
2022-11-21 21:55:18,022   INFO  launcher         pytorch
2022-11-21 21:55:18,022   INFO  tcp_port         18888
2022-11-21 21:55:18,022   INFO  local_rank       0
2022-11-21 21:55:18,022   INFO  set_cfgs         None
2022-11-21 21:55:18,022   INFO  max_waiting_mins 30
2022-11-21 21:55:18,022   INFO  start_epoch      0
2022-11-21 21:55:18,022   INFO  eval_tag         default
2022-11-21 21:55:18,022   INFO  eval_all         False
2022-11-21 21:55:18,022   INFO  ckpt_dir         None
2022-11-21 21:55:18,022   INFO  save_to_file     False
2022-11-21 21:55:18,022   INFO  infer_time       False
2022-11-21 21:55:18,022   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 21:55:18,022   INFO  cfg.LOCAL_RANK: 0
2022-11-21 21:55:18,022   INFO  cfg.CLASS_NAMES: ['Car']
2022-11-21 21:55:18,022   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 21:55:18,022   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 21:55:18,022   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 21:55:18,022   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 21:55:18,022   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': False}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 21:55:18,023   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 21:55:18,023   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 21:55:18,023   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:55:18,023   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:55:18,023   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}, {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.05, 0.05, 0.1], 'MAX_POINTS_PER_VOXEL': 5, 'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}}]
2022-11-21 21:55:18,023   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 21:55:18,023   INFO  
cfg.MODEL = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.NAME: VoxelRCNN
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.VFE = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.VFE.NAME: MeanVFE
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_3D.NAME: VoxelBackBone8x
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.MAP_TO_BEV = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.MAP_TO_BEV.NAME: HeightCompression
2022-11-21 21:55:18,023   INFO  cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES: 256
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.BACKBONE_2D = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.NAME: BaseBEVBackbone
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.LAYER_NUMS: [5, 5]
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.LAYER_STRIDES: [1, 2]
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.NUM_FILTERS: [64, 128]
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES: [1, 2]
2022-11-21 21:55:18,023   INFO  cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS: [128, 128]
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.DENSE_HEAD = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.NAME: AnchorHeadSingle
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.CLASS_AGNOSTIC: False
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.USE_DIRECTION_CLASSIFIER: True
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.DIR_OFFSET: 0.78539
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET: 0.0
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS: 2
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG: [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}]
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NAME: AxisAlignedTargetAssigner
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.POS_FRACTION: -1.0
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.SAMPLE_SIZE: 512
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NORM_BY_NUM_EXAMPLES: False
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.MATCH_HEIGHT: False
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:55:18,023   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:55:18,023   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.cls_weight: 1.0
2022-11-21 21:55:18,024   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.loc_weight: 2.0
2022-11-21 21:55:18,024   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dir_weight: 0.2
2022-11-21 21:55:18,024   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NAME: VoxelRCNNHead
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.SHARED_FC: [256, 256]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.3
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.USE_FAST_NMS: False
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.SCORE_THRESH: 0.0
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 2048
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.7
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.FEATURES_SOURCE: ['x_conv2', 'x_conv3', 'x_conv4']
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.PRE_MLP: True
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.GRID_SIZE: 6
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS = edict()
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2 = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2.MLPS: [[32, 32]]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2.QUERY_RANGES: [[4, 4, 4]]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2.POOL_RADIUS: [0.4]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2.NSAMPLE: [16]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv2.POOL_METHOD: max_pool
2022-11-21 21:55:18,024   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3 = edict()
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3.MLPS: [[32, 32]]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3.QUERY_RANGES: [[4, 4, 4]]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3.POOL_RADIUS: [0.8]
2022-11-21 21:55:18,024   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3.NSAMPLE: [16]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv3.POOL_METHOD: max_pool
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4 = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4.MLPS: [[32, 32]]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4.QUERY_RANGES: [[4, 4, 4]]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4.POOL_RADIUS: [1.6]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4.NSAMPLE: [16]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_LAYERS.x_conv4.POOL_METHOD: max_pool
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: roi_iou
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.75
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.25
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.55
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.GRID_3D_IOU_LOSS: False
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_iou3d_weight: 1.0
2022-11-21 21:55:18,025   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.3
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 21:55:18,025   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 21:55:18,025   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 21:55:18,025   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 2
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.LR: 0.01
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 21:55:18,026   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 21:55:18,026   INFO  cfg.TAG: voxel_rcnn_car
2022-11-21 21:55:18,026   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 21:55:18,027   INFO  Loading KITTI dataset
2022-11-21 21:55:18,117   INFO  Total samples for KITTI dataset: 3769
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-11-21 21:55:18,814   INFO  ==> Loading parameters from checkpoint ../checkpoints/voxel_rcnn_car_8454.pth to CPU
2022-11-21 21:55:18,840   INFO  ==> Done (loaded 257/257)
2022-11-21 21:55:18,855   INFO  *************** EPOCH 8454 EVALUATION *****************
eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 135/135 [00:58<00:00,  2.30it/s, recall_0.3=(6877, 6877) / 7052]
2022-11-21 21:56:24,612   INFO  *************** Performance of EPOCH 8454 *****************
2022-11-21 21:56:24,612   INFO  Generate label finished(sec_per_example: 0.0174 second).
2022-11-21 21:56:24,612   INFO  recall_roi_0.3: 0.977756
2022-11-21 21:56:24,612   INFO  recall_rcnn_0.3: 0.977548
2022-11-21 21:56:24,612   INFO  recall_roi_0.5: 0.971083
2022-11-21 21:56:24,612   INFO  recall_rcnn_0.5: 0.971361
2022-11-21 21:56:24,612   INFO  recall_roi_0.7: 0.825316
2022-11-21 21:56:24,612   INFO  recall_rcnn_0.7: 0.866259
2022-11-21 21:56:24,614   INFO  Average predicted number of objects(3769 samples): 6.025
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 15 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 56 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 15 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 12 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 15 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 56 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 21:56:39,818   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:97.8540, 89.7324, 89.3443
bev  AP:90.2107, 88.2890, 87.7703
3d   AP:89.1421, 84.5414, 78.6853
aos  AP:97.81, 89.64, 89.19
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:98.7723, 94.9265, 92.5585
bev  AP:93.5534, 91.1815, 88.9214
3d   AP:92.1587, 85.0154, 82.4832
aos  AP:98.74, 94.79, 92.37
Car AP@0.70, 0.50, 0.50:
bbox AP:97.8540, 89.7324, 89.3443
bev  AP:97.8917, 89.7398, 89.3670
3d   AP:97.8532, 89.7274, 89.3430
aos  AP:97.81, 89.64, 89.19
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:98.7723, 94.9265, 92.5585
bev  AP:98.7555, 94.9615, 94.6861
3d   AP:98.7242, 94.9308, 94.5958
aos  AP:98.74, 94.79, 92.37

2022-11-21 21:56:39,824   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/voxel_rcnn_car/default/eval/epoch_8454/val/default
2022-11-21 21:56:39,824   INFO  ****************Evaluation done.*****************