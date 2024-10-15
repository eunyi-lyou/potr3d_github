#!/bin/bash

#SBATCH --job-name=T_a2_vid3dhp       # Submit a job named "example"
#SBATCH --partition=a3000        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:1             # Use 1 GPU
#SBATCH --time=7-00:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=40000              # cpu memory size
#SBATCH --cpus-per-task=6        # cpu 개수
#SBATCH --output=slurm_log_out.txt         # 스크립트 실행 결과 std output을 저장할 파일 이름

ml purge
ml load cuda/11.0                # 필요한 쿠다 버전 로드
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate potr3d             # Activate your conda environment

srun python run/train.py --cfg configs/vid3dhp/vid3dhp.yaml --gpus 1
#python run/eval.py --cfg configs/panoptic/panoptic.yaml --gpus 1


# Node1 : 30000 / 12
# Node2 : 25000 / 8
# Node3 : 12000 / 8
