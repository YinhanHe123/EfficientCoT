#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=7:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=1   # number of CPU cores per task
#SBATCH --gres=gpu:RTX2080   # gpu devices per node
#SBATCH --partition=interactive   # partition
#SBATCH -J "log_analysis"   # job name
#SBATCH --mail-user=nee7ne@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --constraint=rivanna   # cluster
#SBATCH --account=UVADM_PAID   # allocation name