┌─[cristian][Desktop][±][master ↓4 U:4 ?:2 ✗][~/.../OpenPCDet/tools]
└─▪  python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/second.yaml --batch_size 34 --ckpt ../checkpoints/second_7862.pth 
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
2022-11-21 21:33:59,241   INFO  **********************Start logging**********************
2022-11-21 21:33:59,241   INFO  CUDA_VISIBLE_DEVICES=ALL
2022-11-21 21:33:59,241   INFO  total_batch_size: 34
2022-11-21 21:33:59,241   INFO  cfg_file         cfgs/kitti_models/second.yaml
2022-11-21 21:33:59,241   INFO  batch_size       17
2022-11-21 21:33:59,241   INFO  workers          4
2022-11-21 21:33:59,241   INFO  extra_tag        default
2022-11-21 21:33:59,241   INFO  ckpt             ../checkpoints/second_7862.pth
2022-11-21 21:33:59,241   INFO  pretrained_model None
2022-11-21 21:33:59,241   INFO  launcher         pytorch
2022-11-21 21:33:59,241   INFO  tcp_port         18888
2022-11-21 21:33:59,241   INFO  local_rank       0
2022-11-21 21:33:59,241   INFO  set_cfgs         None
2022-11-21 21:33:59,241   INFO  max_waiting_mins 30
2022-11-21 21:33:59,241   INFO  start_epoch      0
2022-11-21 21:33:59,241   INFO  eval_tag         default
2022-11-21 21:33:59,242   INFO  eval_all         False
2022-11-21 21:33:59,242   INFO  ckpt_dir         None
2022-11-21 21:33:59,242   INFO  save_to_file     False
2022-11-21 21:33:59,242   INFO  infer_time       False
2022-11-21 21:33:59,242   INFO  cfg.ROOT_DIR: /home/cristian/Github/OpenPCDet
2022-11-21 21:33:59,242   INFO  cfg.LOCAL_RANK: 0
2022-11-21 21:33:59,242   INFO  cfg.CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
2022-11-21 21:33:59,242   INFO  
cfg.DATA_CONFIG = edict()
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATASET: KittiDataset
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_PATH: ../data/kitti
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
2022-11-21 21:33:59,242   INFO  
cfg.DATA_CONFIG.DATA_SPLIT = edict()
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_SPLIT.train: train
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_SPLIT.test: val
2022-11-21 21:33:59,242   INFO  
cfg.DATA_CONFIG.INFO_PATH = edict()
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.INFO_PATH.train: ['kitti_infos_train.pkl']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.INFO_PATH.test: ['kitti_infos_val.pkl']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.GET_ITEM_LIST: ['points']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.FOV_POINTS_ONLY: True
2022-11-21 21:33:59,242   INFO  
cfg.DATA_CONFIG.DATA_AUGMENTOR = edict()
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST: ['placeholder']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST: [{'NAME': 'gt_sampling', 'USE_ROAD_PLANE': True, 'DB_INFO_PATH': ['kitti_dbinfos_train.pkl'], 'PREPARE': {'filter_by_min_points': ['Car:5', 'Pedestrian:5', 'Cyclist:5'], 'filter_by_difficulty': [-1]}, 'SAMPLE_GROUPS': ['Car:20', 'Pedestrian:15', 'Cyclist:15'], 'NUM_POINT_FEATURES': 4, 'DATABASE_WITH_FAKELIDAR': False, 'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0], 'LIMIT_WHOLE_SCENE': True}, {'NAME': 'random_world_flip', 'ALONG_AXIS_LIST': ['x']}, {'NAME': 'random_world_rotation', 'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]}, {'NAME': 'random_world_scaling', 'WORLD_SCALE_RANGE': [0.95, 1.05]}]
2022-11-21 21:33:59,242   INFO  
cfg.DATA_CONFIG.POINT_FEATURE_ENCODING = edict()
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.encoding_type: absolute_coordinates_encoding
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.used_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list: ['x', 'y', 'z', 'intensity']
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG.DATA_PROCESSOR: [{'NAME': 'mask_points_and_boxes_outside_range', 'REMOVE_OUTSIDE_BOXES': True}, {'NAME': 'shuffle_points', 'SHUFFLE_ENABLED': {'train': True, 'test': False}}, {'NAME': 'transform_points_to_voxels', 'VOXEL_SIZE': [0.05, 0.05, 0.1], 'MAX_POINTS_PER_VOXEL': 5, 'MAX_NUMBER_OF_VOXELS': {'train': 16000, 'test': 40000}}]
2022-11-21 21:33:59,242   INFO  cfg.DATA_CONFIG._BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
2022-11-21 21:33:59,242   INFO  
cfg.MODEL = edict()
2022-11-21 21:33:59,242   INFO  cfg.MODEL.NAME: SECONDNet
2022-11-21 21:33:59,242   INFO  
cfg.MODEL.VFE = edict()
2022-11-21 21:33:59,242   INFO  cfg.MODEL.VFE.NAME: MeanVFE
2022-11-21 21:33:59,242   INFO  
cfg.MODEL.BACKBONE_3D = edict()
2022-11-21 21:33:59,242   INFO  cfg.MODEL.BACKBONE_3D.NAME: VoxelBackBone8x
2022-11-21 21:33:59,242   INFO  
cfg.MODEL.MAP_TO_BEV = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.MAP_TO_BEV.NAME: HeightCompression
2022-11-21 21:33:59,243   INFO  cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES: 256
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.BACKBONE_2D = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.NAME: BaseBEVBackbone
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.LAYER_NUMS: [5, 5]
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.LAYER_STRIDES: [1, 2]
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.NUM_FILTERS: [128, 256]
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.UPSAMPLE_STRIDES: [1, 2]
2022-11-21 21:33:59,243   INFO  cfg.MODEL.BACKBONE_2D.NUM_UPSAMPLE_FILTERS: [256, 256]
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.DENSE_HEAD = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.NAME: AnchorHeadSingle
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.CLASS_AGNOSTIC: False
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.USE_DIRECTION_CLASSIFIER: True
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.DIR_OFFSET: 0.78539
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.DIR_LIMIT_OFFSET: 0.0
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.NUM_DIR_BINS: 2
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG: [{'class_name': 'Car', 'anchor_sizes': [[3.9, 1.6, 1.56]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-1.78], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.6, 'unmatched_threshold': 0.45}, {'class_name': 'Pedestrian', 'anchor_sizes': [[0.8, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-0.6], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}, {'class_name': 'Cyclist', 'anchor_sizes': [[1.76, 0.6, 1.73]], 'anchor_rotations': [0, 1.57], 'anchor_bottom_heights': [-0.6], 'align_center': False, 'feature_map_stride': 8, 'matched_threshold': 0.5, 'unmatched_threshold': 0.35}]
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NAME: AxisAlignedTargetAssigner
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.POS_FRACTION: -1.0
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.SAMPLE_SIZE: 512
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NORM_BY_NUM_EXAMPLES: False
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.MATCH_HEIGHT: False
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.BOX_CODER: ResidualCoder
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG = edict()
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.cls_weight: 1.0
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.loc_weight: 2.0
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.dir_weight: 0.2
2022-11-21 21:33:59,243   INFO  cfg.MODEL.DENSE_HEAD.LOSS_CONFIG.LOSS_WEIGHTS.code_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.POST_PROCESSING = edict()
2022-11-21 21:33:59,243   INFO  cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
2022-11-21 21:33:59,243   INFO  cfg.MODEL.POST_PROCESSING.SCORE_THRESH: 0.1
2022-11-21 21:33:59,243   INFO  cfg.MODEL.POST_PROCESSING.OUTPUT_RAW_SCORE: False
2022-11-21 21:33:59,243   INFO  cfg.MODEL.POST_PROCESSING.EVAL_METRIC: kitti
2022-11-21 21:33:59,243   INFO  
cfg.MODEL.POST_PROCESSING.NMS_CONFIG = edict()
2022-11-21 21:33:59,244   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.MULTI_CLASSES_NMS: False
2022-11-21 21:33:59,244   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_TYPE: nms_gpu
2022-11-21 21:33:59,244   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH: 0.01
2022-11-21 21:33:59,244   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_PRE_MAXSIZE: 4096
2022-11-21 21:33:59,244   INFO  cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_POST_MAXSIZE: 500
2022-11-21 21:33:59,244   INFO  
cfg.OPTIMIZATION = edict()
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU: 4
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.NUM_EPOCHS: 80
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.OPTIMIZER: adam_onecycle
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.LR: 0.003
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.WEIGHT_DECAY: 0.01
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.MOMENTUM: 0.9
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.MOMS: [0.95, 0.85]
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.PCT_START: 0.4
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.DIV_FACTOR: 10
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.DECAY_STEP_LIST: [35, 45]
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.LR_DECAY: 0.1
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.LR_CLIP: 1e-07
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.LR_WARMUP: False
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.WARMUP_EPOCH: 1
2022-11-21 21:33:59,244   INFO  cfg.OPTIMIZATION.GRAD_NORM_CLIP: 10
2022-11-21 21:33:59,244   INFO  cfg.TAG: second
2022-11-21 21:33:59,244   INFO  cfg.EXP_GROUP_PATH: kitti_models
2022-11-21 21:33:59,245   INFO  Loading KITTI dataset
2022-11-21 21:33:59,333   INFO  Total samples for KITTI dataset: 3769
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
2022-11-21 21:33:59,981   INFO  ==> Loading parameters from checkpoint ../checkpoints/second_7862.pth to CPU
2022-11-21 21:33:59,995   INFO  ==> Done (loaded 163/163)
2022-11-21 21:34:00,006   INFO  *************** EPOCH 7862 EVALUATION *****************
eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 111/111 [00:50<00:00,  2.21it/s, recall_0.3=(0, 8170) / 8610]
2022-11-21 21:35:02,762   INFO  *************** Performance of EPOCH 7862 *****************
2022-11-21 21:35:02,762   INFO  Generate label finished(sec_per_example: 0.0166 second).
2022-11-21 21:35:02,762   INFO  recall_roi_0.3: 0.000000
2022-11-21 21:35:02,762   INFO  recall_rcnn_0.3: 0.949146
2022-11-21 21:35:02,762   INFO  recall_roi_0.5: 0.000000
2022-11-21 21:35:02,762   INFO  recall_rcnn_0.5: 0.891059
2022-11-21 21:35:02,762   INFO  recall_roi_0.7: 0.000000
2022-11-21 21:35:02,762   INFO  recall_rcnn_0.7: 0.665034
2022-11-21 21:35:02,764   INFO  Average predicted number of objects(3769 samples): 14.227
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 40 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 45 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 50 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/core/typed_passes.py:330: NumbaPerformanceWarning: 
The keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.

To find out why, try turning on parallel diagnostics, see https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics for help.

File "../pcdet/datasets/kitti/kitti_object_eval_python/eval.py", line 122:
@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
^

  state.func_ir.loc))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 40 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 45 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 32 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 50 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 36 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 35 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
/home/cristian/anaconda3/envs/openpcd/lib/python3.7/site-packages/numba/cuda/dispatcher.py:488: NumbaPerformanceWarning: Grid size 28 will likely result in GPU under-utilization due to low occupancy.
  warn(NumbaPerformanceWarning(msg))
2022-11-21 21:35:24,401   INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:90.7803, 89.8999, 89.0433
bev  AP:90.0097, 87.9282, 86.4528
3d   AP:88.6137, 78.6245, 77.2243
aos  AP:90.76, 89.77, 88.82
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.6261, 94.1728, 91.7683
bev  AP:92.4184, 88.5586, 87.6479
3d   AP:90.5534, 81.6116, 78.6108
aos  AP:95.59, 94.01, 91.52
Car AP@0.70, 0.50, 0.50:
bbox AP:90.7803, 89.8999, 89.0433
bev  AP:90.7940, 90.1441, 89.5173
3d   AP:90.7940, 90.0886, 89.4014
aos  AP:90.76, 89.77, 88.82
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.6261, 94.1728, 91.7683
bev  AP:95.6751, 94.8476, 94.2478
3d   AP:95.6623, 94.7450, 94.0537
aos  AP:95.59, 94.01, 91.52
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:68.7347, 66.3279, 63.2579
bev  AP:61.9979, 56.6604, 53.8126
3d   AP:56.5544, 52.9835, 47.7343
aos  AP:64.66, 61.76, 58.45
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:69.5228, 66.4894, 62.8409
bev  AP:60.7329, 56.5680, 52.1375
3d   AP:55.9413, 51.1434, 46.1661
aos  AP:64.91, 61.28, 57.51
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:68.7347, 66.3279, 63.2579
bev  AP:75.3555, 73.8473, 69.7569
3d   AP:75.3437, 73.7890, 69.6610
aos  AP:64.66, 61.76, 58.45
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:69.5228, 66.4894, 62.8409
bev  AP:76.3102, 74.7104, 70.8022
3d   AP:76.3025, 74.5821, 70.7215
aos  AP:64.91, 61.28, 57.51
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:87.5600, 77.0909, 74.3813
bev  AP:84.0183, 70.7012, 65.4772
3d   AP:80.5862, 67.1589, 63.1087
aos  AP:87.42, 76.74, 74.00
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:91.4087, 79.0226, 75.4018
bev  AP:88.0461, 71.1604, 66.8949
3d   AP:82.9640, 66.7401, 62.7843
aos  AP:91.25, 78.61, 74.96
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:87.5600, 77.0909, 74.3813
bev  AP:86.0192, 76.8489, 72.3789
3d   AP:86.0192, 76.8489, 72.3770
aos  AP:87.42, 76.74, 74.00
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:91.4087, 79.0226, 75.4018
bev  AP:90.2814, 77.2718, 73.5534
3d   AP:90.2814, 77.2718, 73.5507
aos  AP:91.25, 78.61, 74.96

2022-11-21 21:35:24,405   INFO  Result is save to /home/cristian/Github/OpenPCDet/output/kitti_models/second/default/eval/epoch_7862/val/default
2022-11-21 21:35:24,405   INFO  ****************Evaluation done.*****************