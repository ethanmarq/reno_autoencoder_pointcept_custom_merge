#!/bin/bash

#SBATCH --job-name train_reno_pointcept
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 64gb
#SBATCH --time 30:00:00
  
#cd ~/jun25_mamba_goosepptv3/Pointcept/data
#source activate jun25_mamba_goosepptv3
cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept

###
#cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept/data

#unlink goose

## We will be using pure files 
#COMPRESSOR="reno"
#Q_lvl="Q_64"
#ln -s /scratch/marque6/goose-pointcept-decomp-bin/${COMPRESSOR}/${Q_lvl} goose

##cd ~/jun25_mamba_goosepptv3/Pointcept
#cd /home/marque6/000_MLBD/2025_Summer/merged_PointTransformerV3/Pointcept
###

source ../activate_pixi.sh
#source activate pointcept-torch2.5.0-cu12.4

wandb login	

sh ./scripts/train.sh -g 1 -d goose -c semseg-merged-v3m1-0-base -n reno_pptv3_uncompressed_jul28