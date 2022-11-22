┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 20  --ckpt ../checkpoints/pointrcnn_7870.pth
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
2022-11-21 17:29:12,813   INFO  **********************Start logging**********************
2022-11-21 17:29:12,813   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 17:29:12,813   INFO  total_batch_size: 20
2022-11-21 17:29:12,813   INFO  cfg_file         cfgs/kitti_models/pointrcnn.yaml
2022-11-21 17:29:12,814   INFO  batch_size       10
2022-11-21 17:29:12,814   INFO  workers          4
2022-11-21 17:29:12,814   INFO  extra_tag        default
2022-11-21 17:29:12,814   INFO  ckpt             ../checkpoints/pointrcnn_7870.pth
2022-11-21 17:29:12,814   INFO  pretrained_model None
2022-11-21 17:29:12,814   INFO  launcher         pytorch
2022-11-21 17:29:12,814   INFO  tcp_port         18888
2022-11-21 17:29:12,814   INFO  local_rank       0
2022-11-21 17:29:12,814   INFO  set_cfgs         None
2022-11-21 17:29:12,814   INFO  max_waiting_mins 30
2022-11-21 17:29:12,814   INFO  start_epoch      0
2022-11-21 17:29:12,814   INFO  eval_tag         default
2022-11-21 17:29:12,814   INFO  eval_all         False
2022-11-21 17:29:12,814   INFO  ckpt_dir         None
2022-11-21 17:29:12,814   INFO  save_to_file     False
2022-11-21 17:29:12,814   INFO  infer_time       False
2022-11-21 17:29:12,814   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 17:29:12,814   INFO  cfg.LOCAL_RANK: 0
2022-11-21 17:29:12,814   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-21 17:29:12,814   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 17:29:12,814   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 17:29:12,814   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 17:29:12,814   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:20', 'Pedestrian:15', 'Cyclist:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': True}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 17:29:12,814   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'sample_points', 'NUM_POINTS': {'train': 16384, 'test': 16384}}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}]
2022-11-21 17:29:12,814   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 17:29:12,815   INFO  
cfg.MODEL = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.NAME: PointRCNN
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.NAME: PointNet2MSG
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.BACKBONE_3D.SA_CONFIG = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.NPOINTS: [4096, 1024, 256, 64]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.RADIUS: [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.NSAMPLE: [[16, 32], [16, 32], [16, 32], [16, 32]]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.SA_CONFIG.MLPS: [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.BACKBONE_3D.FP_MLPS: [[128, 128], [256, 256], [512, 512], [512, 512]]
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.POINT_HEAD = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.NAME: PointHeadBox
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.CLS_FC: [256, 256]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.REG_FC: [256, 256]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC: False
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.USE_POINT_FEATURES_BEFORE_FUSION: False
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER: PointResidualCoder
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.use_mean_size: True
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_REG: WeightedSmoothL1Loss
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_cls_weight: 1.0
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_box_weight: 1.0
2022-11-21 17:29:12,815   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.NAME: PointRCNNHead
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-21 17:29:12,815   INFO  
cfg.MODEL.ROI_HEAD.ROI_POINT_POOL = edict()
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.POOL_EXTRA_WIDTH: [0.0, 0.0, 0.0]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.NUM_SAMPLED_POINTS: 512
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.ROI_POINT_POOL.DEPTH_NORMALIZER: 70.0
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.XYZ_UP_LAYER: [128, 128]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.0
2022-11-21 17:29:12,815   INFO  cfg.MODEL.ROI_HEAD.USE_BN: False
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.SA_CONFIG = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.NPOINTS: [128, 32, -1]
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.RADIUS: [0.2, 0.4, 100]
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.NSAMPLE: [16, 16, 16]
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.SA_CONFIG.MLPS: [[128, 128, 128], [128, 128, 256], [256, 256, 512]]
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 9000
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.85
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: cls
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.6
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.45
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.55
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-21 17:29:12,816   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 17:29:12,816   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 17:29:12,817   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 17:29:12,817   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 17:29:12,817   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 2
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.LR: 0.01
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 17:29:12,817   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 17:29:12,817   INFO  cfg.TAG: pointrcnn
2022-11-21 17:29:12,817   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 17:29:12,818   INFO  Loading KITTI dataset
2022-11-21 17:29:12,906   INFO  Total samples for KITTI dataset: 3769
2022-11-21 17:29:13,540   INFO  ==> Loading parameters from checkpoint ../checkpoints/pointrcnn_7870.pth to CPU
2022-11-21 17:29:13,557   INFO  ==> Done (loaded 309/309)
2022-11-21 17:29:13,570   INFO  *************** EPOCH 7870 EVALUATION *****************
eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 189/189 [02:51<00:00,  1.10it/s, recall_0.3=(7743, 7747) / 8610]
2022-11-21 17:32:04,896   INFO  *************** Performance of EPOCH 7870 *****************
2022-11-21 17:32:04,896   INFO  Generate label finished(sec_per_example: 0.0454 second).
2022-11-21 17:32:04,896   INFO  recall_roi_0.3: 0.898633
2022-11-21 17:32:04,896   INFO  recall_rcnn_0.3: 0.898918
2022-11-21 17:32:04,896   INFO  recall_roi_0.5: 0.864294
2022-11-21 17:32:04,896   INFO  recall_rcnn_0.5: 0.870444
2022-11-21 17:32:04,897   INFO  recall_roi_0.7: 0.686446
2022-11-21 17:32:04,897   INFO  recall_rcnn_0.7: 0.735592
2022-11-21 17:32:04,898   INFO  Average predicted number of objects(3769 samples): 6.228
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 16 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 56 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 25 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 56 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 17:32:25,780   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7797, 89.7119, 89.1800
bev  AP:90.0494, 87.4828, 86.4805
3d   AP:88.6609, 78.6203, 77.7938
aos  AP:90.77, 89.63, 89.01
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:96.5074, 92.9188, 90.5288
bev  AP:93.2462, 89.0235, 86.8056
3d   AP:89.6542, 80.5246, 78.0665
aos  AP:96.49, 92.82, 90.36
Car AP@0.70, 0.50, 0.50:
bbox AP:90.7797, 89.7119, 89.1800
bev  AP:90.7336, 89.8390, 89.5504
3d   AP:90.7336, 89.8029, 89.4876
aos  AP:90.77, 89.63, 89.01
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:96.5074, 92.9188, 90.5288
bev  AP:96.5302, 95.1361, 92.9026
3d   AP:96.5102, 94.9846, 92.7911
aos  AP:96.49, 92.82, 90.36
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:75.3166, 71.1212, 63.3305
bev  AP:66.7934, 58.8341, 53.1197
3d   AP:62.0826, 54.5517, 49.9345
aos  AP:72.84, 67.79, 60.20
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:76.8182, 70.8655, 63.6607
bev  AP:66.0839, 58.0921, 51.3417
3d   AP:62.9127, 54.9382, 48.0997
aos  AP:74.08, 67.25, 60.16
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:75.3166, 71.1212, 63.3305
bev  AP:81.6625, 75.2302, 67.1499
3d   AP:81.5958, 75.1197, 67.0700
aos  AP:72.84, 67.79, 60.20
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:76.8182, 70.8655, 63.6607
bev  AP:82.8982, 76.7043, 69.0477
3d   AP:82.8217, 76.6031, 68.9406
aos  AP:74.08, 67.25, 60.16
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.1908, 76.3076, 74.4645
bev  AP:86.8837, 72.5273, 66.7896
3d   AP:85.7303, 70.7371, 64.5617
aos  AP:89.11, 75.92, 74.04
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.7430, 76.8896, 73.9009
bev  AP:91.9279, 73.0649, 68.7534
3d   AP:90.4812, 69.9132, 65.4396
aos  AP:94.65, 76.47, 73.46
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.1908, 76.3076, 74.4645
bev  AP:94.8311, 74.4487, 72.3127
3d   AP:94.8311, 74.4487, 72.3127
aos  AP:89.11, 75.92, 74.04
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.7430, 76.8896, 73.9009
bev  AP:95.6446, 74.8492, 71.6276
3d   AP:95.6446, 74.8492, 71.6276
aos  AP:94.65, 76.47, 73.46

2022-11-21 17:32:25,784   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/pointrcnn/default/eval/epoch_7870/val/default
2022-11-21 17:32:25,784   INFO  ****************Evaluation done.*****************