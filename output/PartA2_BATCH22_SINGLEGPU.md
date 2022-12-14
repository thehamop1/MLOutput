2022-11-15 22:12:31,032   INFO  **********************Start logging**********************
2022-11-15 22:12:31,032   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-15 22:12:31,032   INFO  cfg_file         cfgs/kitti_models/PartA2.yaml
2022-11-15 22:12:31,032   INFO  batch_size       22
2022-11-15 22:12:31,032   INFO  workers          4
2022-11-15 22:12:31,032   INFO  extra_tag        default
2022-11-15 22:12:31,032   INFO  ckpt             ../checkpoints/PartA2_7940.pth
2022-11-15 22:12:31,032   INFO  pretrained_model None
2022-11-15 22:12:31,032   INFO  launcher         none
2022-11-15 22:12:31,032   INFO  tcp_port         18888
2022-11-15 22:12:31,032   INFO  local_rank       0
2022-11-15 22:12:31,032   INFO  set_cfgs         None
2022-11-15 22:12:31,032   INFO  max_waiting_mins 30
2022-11-15 22:12:31,032   INFO  start_epoch      0
2022-11-15 22:12:31,032   INFO  eval_tag         default
2022-11-15 22:12:31,032   INFO  eval_all         False
2022-11-15 22:12:31,032   INFO  ckpt_dir         None
2022-11-15 22:12:31,032   INFO  save_to_file     False
2022-11-15 22:12:31,032   INFO  infer_time       False
2022-11-15 22:12:31,032   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-15 22:12:31,032   INFO  cfg.LOCAL_RANK: 0
2022-11-15 22:12:31,032   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-15 22:12:31,032   INFO  
cfg.DATA_CONFIG = edict()
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-15 22:12:31,032   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-15 22:12:31,032   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-15 22:12:31,032   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-15 22:12:31,032   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:20', 'Pedestrian:15', 'Cyclist:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': True}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-15 22:12:31,032   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-15 22:12:31,033   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-15 22:12:31,033   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-15 22:12:31,033   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-15 22:12:31,033   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}, {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.05, 0.05, 0.1], 'MAX_POINTS_PER_VOXEL': 5, 'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}}]
2022-11-15 22:12:31,033   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-15 22:12:31,033   INFO  
cfg.MODEL = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.NAME: PartA2Net
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.VFE = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.VFE.NAME: MeanVFE
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_3D.NAME: UNetV2
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.MAP_TO_BEV = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.MAP_TO_BEV.NAME: HeightCompression
2022-11-15 22:12:31,033   INFO  cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES: 256
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.BACKBONE_2D = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.NAME: BaseBEVBackbone
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.LAYER_NUMS: [5, 5]
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.LAYER_STRIDES: [1, 2]
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.NUM_FILTERS: [128, 256]
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES: [1, 2]
2022-11-15 22:12:31,033   INFO  cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS: [256, 256]
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.DENSE_HEAD = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.NAME: AnchorHeadSingle
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.CLASS_AGNOSTIC: False
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.USE_DIRECTION_CLASSIFIER: True
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.DIR_OFFSET: 0.78539
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET: 0.0
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS: 2
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG: [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}, {'class_name': 'Pedestrian', 'anchor_sizes': [[0.8, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}, {'class_name': 'Cyclist', 'anchor_sizes': [[1.76, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}]
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NAME: AxisAlignedTargetAssigner
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.POS_FRACTION: -1.0
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.SAMPLE_SIZE: 512
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NORM_BY_NUM_EXAMPLES: False
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.MATCH_HEIGHT: False
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.BOX_CODER: ResidualCoder
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG = edict()
2022-11-15 22:12:31,033   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.cls_weight: 1.0
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.loc_weight: 2.0
2022-11-15 22:12:31,033   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dir_weight: 0.2
2022-11-15 22:12:31,034   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.POINT_HEAD = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.NAME: PointIntraPartOffsetHead
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.CLS_FC: []
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.PART_FC: []
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC: True
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_REG: smooth-l1
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_cls_weight: 1.0
2022-11-15 22:12:31,034   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_part_weight: 1.0
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NAME: PartA2FCHead
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.SHARED_FC: [256, 256, 256]
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.3
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.SEG_MASK_SCORE_THRESH: 0.3
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 1024
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.7
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.POOL_SIZE: 12
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.NUM_FEATURES: 128
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL: 128
2022-11-15 22:12:31,034   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-15 22:12:31,034   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: roi_iou
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.75
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.25
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.65
2022-11-15 22:12:31,035   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-15 22:12:31,035   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-15 22:12:31,035   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-15 22:12:31,035   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-15 22:12:31,035   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-15 22:12:31,035   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-15 22:12:31,035   INFO  
cfg.OPTIMIZATION = edict()
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 4
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.LR: 0.01
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-15 22:12:31,035   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-15 22:12:31,036   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-15 22:12:31,036   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-15 22:12:31,036   INFO  cfg.TAG: PartA2
2022-11-15 22:12:31,036   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-15 22:12:31,036   INFO  Loading KITTI dataset
2022-11-15 22:12:31,124   INFO  Total samples for KITTI dataset: 3769
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-11-15 22:12:32,289   INFO  ==> Loading parameters from checkpoint ../checkpoints/PartA2_7940.pth to GPU
2022-11-15 22:12:32,394   INFO  ==> Done (loaded 333/333)
2022-11-15 22:12:32,499   INFO  *************** EPOCH 7940 EVALUATION *****************
eval: 100%|??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????| 172/172 [02:18<00:00,  1.25it/s, recall_0.3=(17033, 17037) / 17558]
2022-11-15 22:14:50,598   INFO  *************** Performance of EPOCH 7940 *****************
2022-11-15 22:14:50,598   INFO  Generate label finished(sec_per_example: 0.0366 second).
2022-11-15 22:14:50,598   INFO  recall_roi_0.3: 0.970099
2022-11-15 22:14:50,598   INFO  recall_rcnn_0.3: 0.970327
2022-11-15 22:14:50,598   INFO  recall_roi_0.5: 0.930117
2022-11-15 22:14:50,598   INFO  recall_rcnn_0.5: 0.935528
2022-11-15 22:14:50,598   INFO  recall_roi_0.7: 0.710901
2022-11-15 22:14:50,598   INFO  recall_rcnn_0.7: 0.746440
2022-11-15 22:14:50,599   INFO  Average predicted number of objects(3769 samples): 11.198
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 30 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 40 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 45 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 104 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 40 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 45 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 104 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-15 22:15:10,999   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:94.7748, 89.3073, 89.0406
bev  AP:90.2249, 87.9551, 87.5484
3d   AP:89.5582, 79.4064, 78.8429
aos  AP:94.71, 89.15, 88.82
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:97.8826, 93.7730, 91.7670
bev  AP:92.9012, 90.0635, 88.3510
3d   AP:92.1511, 82.9096, 81.9974
aos  AP:97.82, 93.58, 91.51
Car AP@0.70, 0.50, 0.50:
bbox AP:94.7748, 89.3073, 89.0406
bev  AP:94.7547, 89.3059, 89.0925
3d   AP:94.7077, 89.2728, 89.0355
aos  AP:94.71, 89.15, 88.82
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:97.8826, 93.7730, 91.7670
bev  AP:97.8110, 93.9588, 93.9089
3d   AP:97.7887, 93.8797, 93.7520
aos  AP:97.82, 93.58, 91.51
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:76.0807, 71.5235, 67.6182
bev  AP:70.5996, 64.0578, 59.9748
3d   AP:65.6892, 60.0500, 55.4494
aos  AP:74.00, 68.92, 64.76
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:76.5134, 72.2418, 68.8811
bev  AP:70.5310, 64.1972, 59.2461
3d   AP:66.8890, 59.6784, 54.6240
aos  AP:74.24, 69.40, 65.65
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:76.0807, 71.5235, 67.6182
bev  AP:78.6060, 75.2669, 72.7024
3d   AP:78.6658, 75.1838, 72.5445
aos  AP:74.00, 68.92, 64.76
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:76.5134, 72.2418, 68.8811
bev  AP:81.2031, 77.3362, 73.6710
3d   AP:81.2751, 77.1333, 73.4641
aos  AP:74.24, 69.40, 65.65
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.0268, 77.4112, 76.1091
bev  AP:86.9221, 73.3521, 70.7766
3d   AP:85.5048, 69.9044, 65.4865
aos  AP:88.87, 77.04, 75.68
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.3216, 80.2885, 77.5140
bev  AP:91.9560, 74.6358, 70.6339
3d   AP:90.3400, 70.1362, 66.9292
aos  AP:94.12, 79.86, 77.06
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.0268, 77.4112, 76.1091
bev  AP:87.4839, 77.8448, 73.2795
3d   AP:87.4839, 77.8427, 73.2524
aos  AP:88.87, 77.04, 75.68
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.3216, 80.2885, 77.5140
bev  AP:92.3894, 78.3787, 75.4342
3d   AP:92.3858, 78.3755, 75.4149
aos  AP:94.12, 79.86, 77.06

2022-11-15 22:15:11,003   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/PartA2/default/eval/epoch_7940/val/default
2022-11-15 22:15:11,003   INFO  ****************Evaluation done.*****************