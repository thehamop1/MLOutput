┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 14  --ckpt ../checkpoints/pv_rcnn_8369.pth 
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
2022-11-21 21:11:14,857   INFO  **********************Start logging**********************
2022-11-21 21:11:14,857   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 21:11:14,857   INFO  total_batch_size: 14
2022-11-21 21:11:14,857   INFO  cfg_file         cfgs/kitti_models/pv_rcnn.yaml
2022-11-21 21:11:14,857   INFO  batch_size       7
2022-11-21 21:11:14,857   INFO  workers          4
2022-11-21 21:11:14,857   INFO  extra_tag        default
2022-11-21 21:11:14,857   INFO  ckpt             ../checkpoints/pv_rcnn_8369.pth
2022-11-21 21:11:14,857   INFO  pretrained_model None
2022-11-21 21:11:14,857   INFO  launcher         pytorch
2022-11-21 21:11:14,857   INFO  tcp_port         18888
2022-11-21 21:11:14,857   INFO  local_rank       0
2022-11-21 21:11:14,857   INFO  set_cfgs         None
2022-11-21 21:11:14,857   INFO  max_waiting_mins 30
2022-11-21 21:11:14,857   INFO  start_epoch      0
2022-11-21 21:11:14,857   INFO  eval_tag         default
2022-11-21 21:11:14,857   INFO  eval_all         False
2022-11-21 21:11:14,857   INFO  ckpt_dir         None
2022-11-21 21:11:14,857   INFO  save_to_file     False
2022-11-21 21:11:14,857   INFO  infer_time       False
2022-11-21 21:11:14,857   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 21:11:14,857   INFO  cfg.LOCAL_RANK: 0
2022-11-21 21:11:14,857   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-21 21:11:14,857   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 21:11:14,857   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 21:11:14,857   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 21:11:14,858   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 21:11:14,858   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 21:11:14,858   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:15', 'Pedestrian:10', 'Cyclist:10'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': False}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 21:11:14,858   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}, {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.05, 0.05, 0.1], 'MAX_POINTS_PER_VOXEL': 5, 'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}}]
2022-11-21 21:11:14,858   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 21:11:14,858   INFO  
cfg.MODEL = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.NAME: PVRCNN
2022-11-21 21:11:14,858   INFO  
cfg.MODEL.VFE = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.VFE.NAME: MeanVFE
2022-11-21 21:11:14,858   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_3D.NAME: VoxelBackBone8x
2022-11-21 21:11:14,858   INFO  
cfg.MODEL.MAP_TO_BEV = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.MAP_TO_BEV.NAME: HeightCompression
2022-11-21 21:11:14,858   INFO  cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES: 256
2022-11-21 21:11:14,858   INFO  
cfg.MODEL.BACKBONE_2D = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.NAME: BaseBEVBackbone
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.LAYER_NUMS: [5, 5]
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.LAYER_STRIDES: [1, 2]
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.NUM_FILTERS: [128, 256]
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES: [1, 2]
2022-11-21 21:11:14,858   INFO  cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS: [256, 256]
2022-11-21 21:11:14,858   INFO  
cfg.MODEL.DENSE_HEAD = edict()
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.NAME: AnchorHeadSingle
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.CLASS_AGNOSTIC: False
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.USE_DIRECTION_CLASSIFIER: True
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.DIR_OFFSET: 0.78539
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET: 0.0
2022-11-21 21:11:14,858   INFO  cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS: 2
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG: [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}, {'class_name': 'Pedestrian', 'anchor_sizes': [[0.8, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-0.6], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}, {'class_name': 'Cyclist', 'anchor_sizes': [[1.76, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-0.6], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}]
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NAME: AxisAlignedTargetAssigner
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.POS_FRACTION: -1.0
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.SAMPLE_SIZE: 512
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NORM_BY_NUM_EXAMPLES: False
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.MATCH_HEIGHT: False
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.cls_weight: 1.0
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.loc_weight: 2.0
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dir_weight: 0.2
2022-11-21 21:11:14,859   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.NAME: VoxelSetAbstraction
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.POINT_SOURCE: raw_points
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.NUM_KEYPOINTS: 2048
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.NUM_OUTPUT_FEATURES: 128
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SAMPLE_METHOD: FPS
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.FEATURES_SOURCE: ['bev', 'x_conv1', 'x_conv2', 'x_conv3', 'x_conv4', 'raw_points']
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE.SA_LAYER = edict()
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE.SA_LAYER.raw_points = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.raw_points.MLPS: [[16, 16], [16, 16]]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.raw_points.POOL_RADIUS: [0.4, 0.8]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.raw_points.NSAMPLE: [16, 16]
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE.SA_LAYER.x_conv1 = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv1.DOWNSAMPLE_FACTOR: 1
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv1.MLPS: [[16, 16], [16, 16]]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv1.POOL_RADIUS: [0.4, 0.8]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv1.NSAMPLE: [16, 16]
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE.SA_LAYER.x_conv2 = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv2.DOWNSAMPLE_FACTOR: 2
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv2.MLPS: [[32, 32], [32, 32]]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv2.POOL_RADIUS: [0.8, 1.2]
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv2.NSAMPLE: [16, 32]
2022-11-21 21:11:14,859   INFO  
cfg.MODEL.PFE.SA_LAYER.x_conv3 = edict()
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv3.DOWNSAMPLE_FACTOR: 4
2022-11-21 21:11:14,859   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv3.MLPS: [[64, 64], [64, 64]]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv3.POOL_RADIUS: [1.2, 2.4]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv3.NSAMPLE: [16, 32]
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.PFE.SA_LAYER.x_conv4 = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv4.DOWNSAMPLE_FACTOR: 8
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv4.MLPS: [[64, 64], [64, 64]]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv4.POOL_RADIUS: [2.4, 4.8]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.PFE.SA_LAYER.x_conv4.NSAMPLE: [16, 32]
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.POINT_HEAD = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.NAME: PointHeadSimple
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.CLS_FC: [256, 256]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC: True
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.USE_POINT_FEATURES_BEFORE_FUSION: True
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_REG: smooth-l1
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_cls_weight: 1.0
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NAME: PVRCNNHead
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.SHARED_FC: [256, 256]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.3
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 1024
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.7
2022-11-21 21:11:14,860   INFO  
cfg.MODEL.ROI_HEAD.ROI_GRID_POOL = edict()
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.GRID_SIZE: 6
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.MLPS: [[64, 64], [64, 64]]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_RADIUS: [0.8, 1.6]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.NSAMPLE: [16, 16]
2022-11-21 21:11:14,860   INFO  cfg.MODEL.ROI_HEAD.ROI_GRID_POOL.POOL_METHOD: max_pool
2022-11-21 21:11:14,861   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: roi_iou
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.75
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.25
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.55
2022-11-21 21:11:14,861   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-21 21:11:14,861   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-21 21:11:14,861   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 21:11:14,861   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 21:11:14,861   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 21:11:14,861   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 21:11:14,861   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 2
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.LR: 0.01
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 21:11:14,861   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 21:11:14,862   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 21:11:14,862   INFO  cfg.TAG: pv_rcnn
2022-11-21 21:11:14,862   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 21:11:14,862   INFO  Loading KITTI dataset
2022-11-21 21:11:14,951   INFO  Total samples for KITTI dataset: 3769
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-11-21 21:11:15,682   INFO  ==> Loading parameters from checkpoint ../checkpoints/pv_rcnn_8369.pth to CPU
2022-11-21 21:11:15,716   INFO  ==> Done (loaded 367/367)
2022-11-21 21:11:15,741   INFO  *************** EPOCH 8369 EVALUATION *****************
eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 270/270 [02:17<00:00,  1.97it/s, recall_0.3=(8340, 8347) / 8610]
2022-11-21 21:13:36,047   INFO  *************** Performance of EPOCH 8369 *****************
2022-11-21 21:13:36,047   INFO  Generate label finished(sec_per_example: 0.0372 second).
2022-11-21 21:13:36,047   INFO  recall_roi_0.3: 0.968451
2022-11-21 21:13:36,047   INFO  recall_rcnn_0.3: 0.968850
2022-11-21 21:13:36,047   INFO  recall_roi_0.5: 0.928474
2022-11-21 21:13:36,047   INFO  recall_rcnn_0.5: 0.932232
2022-11-21 21:13:36,047   INFO  recall_roi_0.7: 0.717312
2022-11-21 21:13:36,047   INFO  recall_rcnn_0.7: 0.751879
2022-11-21 21:13:36,049   INFO  Average predicted number of objects(3769 samples): 9.065
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 30 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 88 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 30 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 30 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 88 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 21:13:58,395   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:95.5723, 89.3166, 88.7600
bev  AP:89.7933, 87.3355, 85.7737
3d   AP:88.9697, 79.0463, 78.1505
aos  AP:95.55, 89.21, 88.58
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:97.3610, 93.6477, 91.6829
bev  AP:92.5622, 88.3843, 87.6385
3d   AP:91.1543, 82.5570, 79.9960
aos  AP:97.35, 93.50, 91.48
Car AP@0.70, 0.50, 0.50:
bbox AP:95.5723, 89.3166, 88.7600
bev  AP:95.5633, 89.3492, 88.8990
3d   AP:95.5157, 89.3271, 88.8452
aos  AP:95.55, 89.21, 88.58
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:97.3610, 93.6477, 91.6829
bev  AP:97.3976, 94.2300, 93.7638
3d   AP:97.3665, 94.1382, 93.5163
aos  AP:97.35, 93.50, 91.48
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:72.8559, 66.2567, 63.5693
bev  AP:66.2107, 59.1476, 54.8051
3d   AP:64.1591, 55.6683, 51.3199
aos  AP:67.46, 60.85, 57.94
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:73.4790, 67.2054, 63.7010
bev  AP:67.2185, 58.7996, 54.3169
3d   AP:63.7611, 54.9620, 50.0080
aos  AP:67.54, 61.17, 57.37
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:72.8559, 66.2567, 63.5693
bev  AP:77.0542, 72.4344, 69.1716
3d   AP:76.9698, 72.1528, 68.8695
aos  AP:67.46, 60.85, 57.94
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:73.4790, 67.2054, 63.7010
bev  AP:79.0767, 73.2859, 69.4113
3d   AP:78.9970, 73.0434, 69.1542
aos  AP:67.54, 61.17, 57.37
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.7089, 77.4644, 75.4108
bev  AP:88.6424, 73.1329, 69.9944
3d   AP:84.8182, 68.0993, 63.5723
aos  AP:89.62, 77.22, 75.12
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:95.2334, 80.4148, 76.0109
bev  AP:93.5624, 73.3727, 69.5484
3d   AP:87.5928, 68.0362, 63.5669
aos  AP:95.14, 80.15, 75.70
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.7089, 77.4644, 75.4108
bev  AP:88.8011, 77.9322, 72.7992
3d   AP:88.8011, 77.9322, 72.7992
aos  AP:89.62, 77.22, 75.12
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:95.2334, 80.4148, 76.0109
bev  AP:93.8794, 78.2105, 73.9705
3d   AP:93.8794, 78.2105, 73.9705
aos  AP:95.14, 80.15, 75.70

2022-11-21 21:13:58,399   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/pv_rcnn/default/eval/epoch_8369/val/default
2022-11-21 21:13:58,399   INFO  ****************Evaluation done.*****************