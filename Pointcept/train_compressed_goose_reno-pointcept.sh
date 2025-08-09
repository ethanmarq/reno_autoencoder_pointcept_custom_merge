#!/bin/bash

#SBATCH --job-name train_reno_pointcept
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node a100:2
#SBATCH --mem 64gb
#SBATCH --time 30:00:00
  
cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept

### Linking Files
#Q_lvl="Q_64"

#cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept/data/goose
#ln -s /scratch/marque6/Goose-Pontcept/labels_challenge labels_challenge
#ln -s /scratch/marque6/Goose-Pontcept/lidar lidar

#cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept/data/goose/compressed_lidar
#ln -s /scratch/marque6/Goose-Pontcept/reno_compression_results/train/${Q_lvl}/compressed train
#ln -s /scratch/marque6/Goose-Pontcept/reno_compression_results/trainEx/${Q_lvl}/compressed trainEx
#ln -s /scratch/marque6/Goose-Pontcept/reno_compression_results/val/${Q_lvl}/compressed val
#ln -s /scratch/marque6/Goose-Pontcept/reno_compression_results/valEx/${Q_lvl}/compressed valEx

#cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept
###

source ../activate_pixi.sh

wandb login

sh ./scripts/train.sh -g 2 -d goose -c semseg-compressed-merged-base -n reno-ptv3_compressed_aug9_num-work8_2
# Change batch_size in config to match number of GPUs