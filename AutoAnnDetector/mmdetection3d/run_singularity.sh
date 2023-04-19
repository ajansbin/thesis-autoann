#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task 16
#SBATCH --time 168:00:00
#SBATCH --mem-per-gpu 64G
#SBATCH --output /workspaces/%u/mmdetection3d/logs/%j.out
#SBATCH --partition zprod
#

singularity exec --nv --bind /workspaces/$USER/mmdetection3d:/mmdetection3d \
  --bind /datasets/zod/zodv2:/mmdetection3d/data/zod \
  --bind /staging/agp/masterthesis/2023autoann/storage/:/storage \
  --pwd /mmdetection3d/ \
  --env PYTHONPATH=/mmdetection3d/ \
  /workspaces/$USER/mmdetection3d/mmdet3d.sif \
  bash $@
#
#EOF
