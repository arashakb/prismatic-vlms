#! /bin/bash 
#SBATCH --partition=ce-mri
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --time=3-00:00:00
#SBATCH --output=logs_sbatch/%x_%j.out
#SBATCH --error=logs_sbatch/%x_%j.err
#SBATCH --exclusive


# Load necessary modules, if any
# module load cuda/11.8  # Example: load CUDA
module load gcc/11.1.0
module load nccl/2.8.3-1-cuda10.2-gcc7.3
module load cuda/12.1
module load ffmpeg/20190305
module load libtool/2.4.6 

# Activate conda env or any other setup
# source ~/miniconda3/etc/profile.d/conda.sh
conda activate prism

# Run your command
./run.sh

