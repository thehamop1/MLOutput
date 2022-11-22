# PartA2 (Non-Free + single GPU TITAN)
## Run Command
python test.py --cfg_file cfgs/kitti_models/PartA2.yaml --batch_size 22 --ckpt ../checkpoints/PartA2_7940.pth

## Output
output/PartA2_BATCH22_SINGLEGPU.md

## Results

# PartA2 (Non-Free + Multi GPU)
## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch  --cfg_file cfgs/kitti_models/PartA2.yaml --batch_size 26 --ckpt ../checkpoints/PartA2_7940.pth

## Output
output/PartA2_BATCH26_MULTIGPU.md

## Results

# PartA2 (Non-Free + Multi GPU)
## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch  --cfg_file cfgs/kitti_models/PartA2.yaml --batch_size 22 --ckpt ../checkpoints/PartA2_7940.pth

## Output
output/PartA2_BATCH22_MULTIGPU.md

## Results

# PartA2 (Free + Single GPU TITAN)
## Run Command
python test.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 32 --ckpt ../checkpoints/PartA2_free_7872.pth 

## Output
output/PartA2_free_BATCH32_SINGLEGPU.md

## Results

# PartA2 (Free + Single GPU TITAN)
## Run Command
python test.py --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 22 --ckpt ../checkpoints/PartA2_free_7872.pth 

## Output
output/PartA2_free_BATCH22_SINGLEGPU.md

## Results

# PartA2 (Free + Multi GPU)
## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 32 --ckpt ../checkpoints/PartA2_free_7872.pth

## Output
output/PartA2_free_BATCH32_MULTIGPU.md

## Results

# PartA2 (Free + Multi GPU)
## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 22 --ckpt ../checkpoints/PartA2_free_7872.pth

## Output
output/PartA2_free_BATCH22_MULTIGPU.md

## Results

# PartA2 (Free + Multi GPU)
## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_free.yaml --batch_size 54 --ckpt ../checkpoints/PartA2_free_7872.pt

## Output
output/PartA2_free_BATCH54_MULTIGPU.md

## Results

# PointRCNN (Single GPU Titan)

## Run Command
python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 20  --ckpt ../checkpoints/pointrcnn_7870.pth

## Output
output/PointRCNN_BATCH20_SINGLEGPU.md

## Results

# PointRCNN (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 20  --ckpt ../checkpoints/pointrcnn_7870.pth
## Output
output/PointRCNN_BATCH20_MULTIGPU.md

## Results

# PointRCNN (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pointrcnn.yaml --batch_size 28  --ckpt ../checkpoints/pointrcnn_7870.pth
## Output
output/PointRCNN_BATCH28_MULTIGPU.md

## Results

# PV-RCNN (Single GPU Titan)

## Run Command
python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 14  --ckpt ../checkpoints/pv_rcnn_8369.pth

## Output
output/PV_RCNN_BATCH14_SINGLEGPU.md

## Results

# PV-RCNN (Mutli-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth 

## Output
output/PV_RCNN_BATCH16_MULTIGPU.md

## Results

# PV-RCNN (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 16 --ckpt ../checkpoints/pv_rcnn_8369.pth 

## Output
output/PV_RCNN_BATCH16_MULTIGPU.md

## Results

# SECOND (Single GPU Titan)

## Run Command
python test.py --cfg_file cfgs/kitti_models/second.yaml --batch_size 34  --ckpt ../checkpoints/second_7862.pth

## Output
output/SECOND_BATCH34_SINGLEGPU.md

## Results

# SECOND (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/second.yaml --batch_size 34 --ckpt ../checkpoints/second_7862.pth 

## Output
output/SECOND_BATCH34_SINGLEGPU.md

## Results

# SECOND (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/second.yaml --batch_size 46 --ckpt ../checkpoints/second_7862.pth 

## Output
output/SECOND_BATCH46_SINGLEGPU.md

## Results

# VOXEL-RCNN (CAR) (Single GPU Titan)

## Run Command
python test.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --batch_size 28  --ckpt ../checkpoints/voxel_rcnn_car_8454.pth

## Output
output/VOXEL_RCNN_CAR_BATCH28_SINGLEGPU.md 

## Results

# VOXEL-RCNN (CAR) (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --batch_size 28 --ckpt ../checkpoints/voxel_rcnn_car_8454.pth


## Output
output/VOXEL_RCNN_CAR_BATCH28_MULTIGPU.md 

## Results

# VOXEL-RCNN (CAR) (Multi-GPU)

## Run Command
python -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --batch_size 34 --ckpt ../checkpoints/voxel_rcnn_car_8454.pth


## Output
output/VOXEL_RCNN_CAR_BATCH34_MULTIGPU.md 

## Results
