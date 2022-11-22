┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 20 --ckpt ../checkpoints/pointrcnn_7870.pth 
2022-11-21 17:21:21,380   INFO  **********************Start logging**********************
2022-11-21 17:21:21,380   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 17:21:21,380   INFO  cfg_file         cfgs/kitti_models/pointrcnn.yaml
2022-11-21 17:21:21,380   INFO  batch_size       20
2022-11-21 17:21:21,380   INFO  workers          4
2022-11-21 17:21:21,380   INFO  extra_tag        default
2022-11-21 17:21:21,380   INFO  ckpt             ../checkpoints/pointrcnn_7870.pth
2022-11-21 17:21:21,380   INFO  pretrained_model None
2022-11-21 17:21:21,380   INFO  launcher         none
2022-11-21 17:21:21,380   INFO  tcp_port         18888
2022-11-21 17:21:21,380   INFO  local_rank       0
2022-11-21 17:21:21,380   INFO  set_cfgs         None
2022-11-21 17:21:21,381   INFO  max_waiting_mins 30
2022-11-21 17:21:21,381   INFO  start_epoch      0
2022-11-21 17:21:21,381   INFO  eval_tag         default
2022-11-21 17:21:21,381   INFO  eval_all         False
2022-11-21 17:21:21,381   INFO  ckpt_dir         None
2022-11-21 17:21:21,381   INFO  save_to_file     False
2022-11-21 17:21:21,381   INFO  infer_time       False
2022-11-21 17:21:21,381   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 17:21:21,381   INFO  cfg.LOCAL_RANK: 0
2022-11-21 17:21:21,381   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-21 17:21:21,381   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 17:21:21,381   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 17:21:21,381   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 17:21:21,381   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:20', 'Pedestrian:15', 'Cyclist:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': True}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 17:21:21,381   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'sample_points', 'NUM_POINTS': {'train': 16384, 'test': 16384}}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}]
2022-11-21 17:21:21,381   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 17:21:21,381   INFO  
cfg.MODEL = edict()
2022-11-21 17:21:21,381   INFO  cfg.MODEL.NAME: PointRCNN
2022-11-21 17:21:21,381   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 17:21:21,381   INFO  cfg.MODEL.BACKBONE_3D.NAME: PointNet2MSG
2022-11-21 17:21:21,381   INFO  
cfg.MODEL.BACKBONE_3D.SA_CONFIG = edict()
2022-11-21 17:21:21,381   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.NPOINTS: [4096, 1024, 256, 64]
2022-11-21 17:21:21,381   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.RADIUS: [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
2022-11-21 17:21:21,381   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.NSAMPLE: [[16, 32], [16, 32], [16, 32], [16, 32]]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.MLPS: [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.BACKBONE_3D.FP_MLPS: [[128, 128], [256, 256], [512, 512], [512, 512]]
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.POINT_HEAD = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.NAME: PointHeadBox
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.CLS_FC: [256, 256]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.REG_FC: [256, 256]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC: False
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.USE_POINT_FEATURES_BEFORE_FUSION: False
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER: PointResidualCoder
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.use_mean_size: True
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_REG: WeightedSmoothL1Loss
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_cls_weight: 1.0
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_box_weight: 1.0
2022-11-21 17:21:21,382   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.NAME: PointRCNNHead
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.ROI_HEAD.ROI_POINT_POOL = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.POOL_EXTRA_WIDTH: [0.0, 0.0, 0.0]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS: 512
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.DEPTH_NORMALIZER: 70.0
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.XYZ_UP_LAYER: [128, 128]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.0
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.USE_BN: False
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.ROI_HEAD.SA_CONFIG = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.NPOINTS: [128, 32, -1]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.RADIUS: [0.2, 0.4, 100]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.NSAMPLE: [16, 16, 16]
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.MLPS: [[128, 128, 128], [128, 128, 256], [256, 256, 512]]
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-21 17:21:21,382   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-21 17:21:21,382   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 9000
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.85
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: cls
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.6
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.45
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.55
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-21 17:21:21,383   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 17:21:21,383   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 17:21:21,383   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 17:21:21,383   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 17:21:21,383   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 2
2022-11-21 17:21:21,383   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.LR: 0.01
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 17:21:21,384   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 17:21:21,384   INFO  cfg.TAG: pointrcnn
2022-11-21 17:21:21,384   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 17:21:21,384   INFO  Loading KITTI dataset
2022-11-21 17:21:21,474   INFO  Total samples for KITTI dataset: 3769
2022-11-21 17:21:22,135   INFO  ==> Loading parameters from checkpoint ../checkpoints/pointrcnn_7870.pth to GPU
2022-11-21 17:21:22,150   INFO  ==> Done (loaded 309/309)
2022-11-21 17:21:22,162   INFO  *************** EPOCH 7870 EVALUATION *****************
eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [05:10<00:00,  1.65s/it, recall_0.3=(15772, 15785) / 17558]
2022-11-21 17:26:33,125   INFO  *************** Performance of EPOCH 7870 *****************
2022-11-21 17:26:33,125   INFO  Generate label finished(sec_per_example: 0.0825 second).
2022-11-21 17:26:33,125   INFO  recall_roi_0.3: 0.898280
2022-11-21 17:26:33,125   INFO  recall_rcnn_0.3: 0.899020
2022-11-21 17:26:33,125   INFO  recall_roi_0.5: 0.864734
2022-11-21 17:26:33,125   INFO  recall_rcnn_0.5: 0.870714
2022-11-21 17:26:33,125   INFO  recall_roi_0.7: 0.687094
2022-11-21 17:26:33,125   INFO  recall_rcnn_0.7: 0.733284
2022-11-21 17:26:33,126   INFO  Average predicted number of objects(3769 samples): 6.224
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 64 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 17:26:54,212   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7552, 89.7105, 89.1598
bev  AP:90.1749, 87.4286, 86.5097
3d   AP:88.7932, 78.7251, 77.7675
aos  AP:90.74, 89.61, 88.98
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.4310, 92.8890, 90.5282
bev  AP:93.3068, 89.0226, 86.8225
3d   AP:91.4116, 80.6158, 78.1439
aos  AP:96.41, 92.77, 90.35
Car AP@0.70, 0.50, 0.50:
bbox AP:90.7552, 89.7105, 89.1598
bev  AP:90.7075, 89.8271, 89.5323
3d   AP:90.7075, 89.7995, 89.4431
aos  AP:90.74, 89.61, 88.98
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:96.4310, 92.8890, 90.5282
bev  AP:96.4381, 95.1214, 92.8896
3d   AP:96.4193, 94.9351, 92.7565
aos  AP:96.41, 92.77, 90.35
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.2860, 66.2215, 62.1143
bev  AP:66.8641, 58.6812, 53.3778
3d   AP:61.2482, 54.0351, 49.6776
aos  AP:70.66, 63.36, 59.11
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.2899, 67.7732, 61.0256
bev  AP:66.0061, 57.9506, 52.3952
3d   AP:62.0759, 54.3830, 47.8597
aos  AP:71.44, 64.47, 57.84
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.2860, 66.2215, 62.1143
bev  AP:81.4931, 74.8693, 66.8467
3d   AP:81.4375, 74.6081, 66.7088
aos  AP:70.66, 63.36, 59.11
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:74.2899, 67.7732, 61.0256
bev  AP:82.0744, 76.0348, 68.7849
3d   AP:82.0175, 75.8580, 68.6315
aos  AP:71.44, 64.47, 57.84
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.3948, 76.9318, 75.3327
bev  AP:94.4121, 73.7166, 67.7811
3d   AP:86.8884, 72.2055, 66.7338
aos  AP:89.10, 76.38, 74.72
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:95.2864, 79.0900, 74.6229
bev  AP:94.8094, 74.3372, 69.9029
3d   AP:92.2013, 72.8828, 68.5147
aos  AP:94.96, 78.44, 74.03
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.3948, 76.9318, 75.3327
bev  AP:95.9022, 75.1657, 73.3661
3d   AP:95.9022, 75.1657, 73.3661
aos  AP:89.10, 76.38, 74.72
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:95.2864, 79.0900, 74.6229
bev  AP:96.2952, 77.1562, 72.6369
3d   AP:96.2952, 77.1562, 72.6369
aos  AP:94.96, 78.44, 74.03

2022-11-21 17:26:54,217   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/pointrcnn/default/eval/epoch_7870/val/default
2022-11-21 17:26:54,217   INFO  ****************Evaluation done.*****************