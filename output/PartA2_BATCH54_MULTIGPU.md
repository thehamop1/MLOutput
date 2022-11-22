┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 54 --ckpt ../checkpoints/PartA2_free_7872.pth
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
2022-11-21 16:55:09,108   INFO  **********************Start logging**********************
2022-11-21 16:55:09,109   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 16:55:09,109   INFO  total_batch_size: 54
2022-11-21 16:55:09,109   INFO  cfg_file         cfgs/kitti_models/PartA2_free.yaml
2022-11-21 16:55:09,109   INFO  batch_size       27
2022-11-21 16:55:09,109   INFO  workers          4
2022-11-21 16:55:09,109   INFO  extra_tag        default
2022-11-21 16:55:09,109   INFO  ckpt             ../checkpoints/PartA2_free_7872.pth
2022-11-21 16:55:09,109   INFO  pretrained_model None
2022-11-21 16:55:09,109   INFO  launcher         pytorch
2022-11-21 16:55:09,109   INFO  tcp_port         18888
2022-11-21 16:55:09,109   INFO  local_rank       0
2022-11-21 16:55:09,109   INFO  set_cfgs         None
2022-11-21 16:55:09,109   INFO  max_waiting_mins 30
2022-11-21 16:55:09,109   INFO  start_epoch      0
2022-11-21 16:55:09,109   INFO  eval_tag         default
2022-11-21 16:55:09,109   INFO  eval_all         False
2022-11-21 16:55:09,109   INFO  ckpt_dir         None
2022-11-21 16:55:09,109   INFO  save_to_file     False
2022-11-21 16:55:09,109   INFO  infer_time       False
2022-11-21 16:55:09,109   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 16:55:09,109   INFO  cfg.LOCAL_RANK: 0
2022-11-21 16:55:09,109   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-21 16:55:09,109   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 16:55:09,110   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 16:55:09,110   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 16:55:09,110   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:20', 'Pedestrian:15', 'Cyclist:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': True}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 16:55:09,110   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}, {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.05, 0.05, 0.1], 'MAX_POINTS_PER_VOXEL': 5, 'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}}]
2022-11-21 16:55:09,110   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 16:55:09,110   INFO  
cfg.MODEL = edict()
2022-11-21 16:55:09,110   INFO  cfg.MODEL.NAME: PointRCNN
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.VFE = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.VFE.NAME: MeanVFE
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.BACKBONE_3D.NAME: UNetV2
2022-11-21 16:55:09,111   INFO  cfg.MODEL.BACKBONE_3D.RETURN_ENCODED_TENSOR: False
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.POINT_HEAD = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.NAME: PointIntraPartOffsetHead
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.CLS_FC: [128, 128]
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.PART_FC: [128, 128]
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.REG_FC: [128, 128]
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.CLASS_AGNOSTIC: False
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.USE_POINT_FEATURES_BEFORE_FUSION: False
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER: PointResidualCoder
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.use_mean_size: True
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.TARGET_CONFIG.BOX_CODER_CONFIG.mean_size: [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_REG: WeightedSmoothL1Loss
2022-11-21 16:55:09,111   INFO  
cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_cls_weight: 1.0
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_box_weight: 1.0
2022-11-21 16:55:09,111   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.point_part_weight: 1.0
2022-11-21 16:55:09,112   INFO  cfg.MODEL.POINT_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 16:55:09,112   INFO  
cfg.MODEL.ROI_HEAD = edict()
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NAME: PartA2FCHead
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.CLASS_AGNOSTIC: True
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.SHARED_FC: [256, 256, 256]
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.CLS_FC: [256, 256]
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.REG_FC: [256, 256]
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.DP_RATIO: 0.3
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.DISABLE_PART: True
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.SEG_MASK_SCORE_THRESH: 0.0
2022-11-21 16:55:09,112   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG = edict()
2022-11-21 16:55:09,112   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN = edict()
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_TYPE: nms_gpu
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.MULTI_CLASSES_NMS: False
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_PRE_MAXSIZE: 9000
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_POST_MAXSIZE: 512
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TRAIN.NMS_THRESH: 0.8
2022-11-21 16:55:09,112   INFO  
cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST = edict()
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_TYPE: nms_gpu
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.MULTI_CLASSES_NMS: False
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_PRE_MAXSIZE: 9000
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_POST_MAXSIZE: 100
2022-11-21 16:55:09,112   INFO  cfg.MODEL.ROI_HEAD.NMS_CONFIG.TEST.NMS_THRESH: 0.85
2022-11-21 16:55:09,112   INFO  
cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL = edict()
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.POOL_SIZE: 12
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.NUM_FEATURES: 128
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL: 128
2022-11-21 16:55:09,113   INFO  
cfg.MODEL.ROI_HEAD.TARGET_CONFIG = edict()
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.ROI_PER_IMAGE: 128
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.FG_RATIO: 0.5
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.SAMPLE_ROI_BY_EACH_CLASS: True
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_SCORE_TYPE: roi_iou
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_FG_THRESH: 0.75
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH: 0.25
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.CLS_BG_THRESH_LO: 0.1
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.HARD_BG_RATIO: 0.8
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.TARGET_CONFIG.REG_FG_THRESH: 0.65
2022-11-21 16:55:09,113   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG = edict()
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CLS_LOSS: BinaryCrossEntropy
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.REG_LOSS: smooth-l1
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION: True
2022-11-21 16:55:09,113   INFO  
cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_cls_weight: 1.0
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_reg_weight: 1.0
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.rcnn_corner_weight: 1.0
2022-11-21 16:55:09,113   INFO  cfg.MODEL.ROI_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 16:55:09,113   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 16:55:09,114   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.1
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 16:55:09,114   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 16:55:09,114   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 4
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.LR: 0.003
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 16:55:09,114   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 16:55:09,115   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 16:55:09,115   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 16:55:09,115   INFO  cfg.TAG: PartA2_free
2022-11-21 16:55:09,115   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 16:55:09,116   INFO  Loading KITTI dataset
2022-11-21 16:55:09,206   INFO  Total samples for KITTI dataset: 3769
2022-11-21 16:55:10,320   INFO  ==> Loading parameters from checkpoint ../checkpoints/PartA2_free_7872.pth to CPU
2022-11-21 16:55:10,435   INFO  ==> Done (loaded 275/275)
2022-11-21 16:55:10,529   INFO  *************** EPOCH 7872 EVALUATION *****************
eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [02:45<00:00,  2.36s/it, recall_0.3=(8194, 8196) / 8610]
2022-11-21 16:57:55,977   INFO  *************** Performance of EPOCH 7872 *****************
2022-11-21 16:57:55,977   INFO  Generate label finished(sec_per_example: 0.0439 second).
2022-11-21 16:57:55,977   INFO  recall_roi_0.3: 0.950740
2022-11-21 16:57:55,977   INFO  recall_rcnn_0.3: 0.950911
2022-11-21 16:57:55,977   INFO  recall_roi_0.5: 0.908713
2022-11-21 16:57:55,977   INFO  recall_rcnn_0.5: 0.911617
2022-11-21 16:57:55,977   INFO  recall_roi_0.7: 0.702335
2022-11-21 16:57:55,977   INFO  recall_rcnn_0.7: 0.735080
2022-11-21 16:57:55,979   INFO  Average predicted number of objects(3769 samples): 10.231
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 24 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 30 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 96 will likely result in GPU under-utilization due to low occupancy.
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
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 20 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 96 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 16:58:16,216   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7064, 89.1032, 88.7695
bev  AP:90.0969, 86.7886, 84.5979
3d   AP:89.1192, 78.7253, 77.9830
aos  AP:90.68, 89.00, 88.64
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.9393, 91.7174, 89.6301
bev  AP:92.8410, 88.1545, 86.1620
3d   AP:91.6840, 80.3118, 78.1013
aos  AP:95.90, 91.61, 89.50
Car AP@0.70, 0.50, 0.50:
bbox AP:90.7064, 89.1032, 88.7695
bev  AP:90.6940, 89.3182, 89.1733
3d   AP:90.6907, 89.2836, 89.1190
aos  AP:90.68, 89.00, 88.64
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.9393, 91.7174, 89.6301
bev  AP:95.9646, 94.0124, 93.6626
3d   AP:95.9422, 93.8628, 91.9144
aos  AP:95.90, 91.61, 89.50
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:80.2816, 75.6527, 71.7540
bev  AP:74.8368, 69.3140, 64.5455
3d   AP:70.3169, 65.9922, 60.4849
aos  AP:78.94, 74.03, 69.88
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:80.8329, 76.9968, 73.3469
bev  AP:75.4837, 69.8351, 64.4979
3d   AP:72.3607, 66.4047, 60.0729
aos  AP:79.32, 75.20, 71.20
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:80.2816, 75.6527, 71.7540
bev  AP:83.7051, 82.0240, 79.0633
3d   AP:83.7003, 81.9823, 78.9359
aos  AP:78.94, 74.03, 69.88
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:80.8329, 76.9968, 73.3469
bev  AP:87.3005, 84.9858, 80.3568
3d   AP:87.2962, 84.9335, 79.6124
aos  AP:79.32, 75.20, 71.20
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.4204, 81.0870, 77.1343
bev  AP:88.7514, 76.2628, 73.6792
3d   AP:87.6469, 74.2926, 69.9137
aos  AP:89.31, 80.57, 76.60
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.0643, 82.9229, 79.0304
bev  AP:93.2320, 78.4984, 73.9288
3d   AP:91.9237, 75.3263, 70.5746
aos  AP:93.90, 82.36, 78.47
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.4204, 81.0870, 77.1343
bev  AP:88.8460, 78.6822, 74.6989
3d   AP:88.8460, 78.6822, 74.6989
aos  AP:89.31, 80.57, 76.60
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.0643, 82.9229, 79.0304
bev  AP:93.3543, 80.3345, 76.5469
3d   AP:93.3543, 80.3345, 76.5469
aos  AP:93.90, 82.36, 78.47

2022-11-21 16:58:16,216   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/PartA2_free/default/eval/epoch_7872/val/default
2022-11-21 16:58:16,216   INFO  ****************Evaluation done.*****************