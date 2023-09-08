#!/bin/bash
#SBATCH -A ykz@v100
#SBATCH --job-name=name_of_the_job  
#SBATCH --output=name_of_the_job%j.out     
#SBATCH --error=name_of_the_job%j.out     
#SBATCH --array=21-31
#SBATCH --constraint v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread        
#SBATCH -t 20:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=20

module purge
module load tensorflow-gpu/py3/2.7.0

python hmc_bhm.py  --seed=$SLURM_ARRAY_TASK_ID --filename=job_$SLURM_ARRAY_TASK_ID
