#!/bin/bash
#SBATCH -A ykz@v100
#SBATCH --array=0-10
#SBATCH --job-name=name_of_the_job         
#SBATCH --output=name_of_the_job%j.out     
#SBATCH --error=name_of_the_job%j.out     
#SBATCH --constraint v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread        
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=20

module purge
module load python cudnn/8.5.0.96-11.7-cuda

python /gpfsdswork/projects/rech/ykz/ulm75uc/VMIM-vs-MSE-/script/train_estimator_array.py --total_steps=9000  --resnet='resnet18' --loss='train_compressor_mse' --seed=$SLURM_ARRAY_TASK_ID --filename=job_$SLURM_ARRAY_TASK_ID
