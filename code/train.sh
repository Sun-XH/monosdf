#!/bin/bash
#SBATCH --gres=gpu:1         # Number of GPUs (per node)
#SBATCH --mem=32G               # memory (per node)
#SBATCH --time=2-15:00            # time (DD-HH:MM)
#SBATCH --cpus-per-task=6         # Number of CPUs (per task)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xsa55@sfu.ca
#SBATCH --nodelist=cs-venus-05
#SBATCH --partition=long
#SBATCH --output=/localscratch/xsa55/monosdf/exps/output/multiscan_4.out
#SBATCH -J multiscan_monosdf_4

echo 'ENV Start'

source ~/.zshrc

module load LIB/CUDA/11.1
module load LIB/CUDNN/8.0.5-CUDA11.1

conda activate monosdf-new

cd /localscratch/xsa55/monosdf/code
echo 'Job Start'
THRUST_IGNORE_CUB_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/multiscan_mlp.conf  --scan_id 1
