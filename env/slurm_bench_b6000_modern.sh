#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -p leinegpu
#SBATCH --job-name="hicgan_bench_modern"
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:b6000:1
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Does a Blackwell-capable CUDA runtime make the b6000 faster for this model?
#
# The training environment (TF 2.15 / CUDA 12.0 / cuDNN 8) gives 1.26 it/s on a
# b6000 at batch 50, against 1.45 on an A100, with fp32 and mixed precision
# byte-identical. cuDNN 8 has no Blackwell kernels, so this is the expected
# result rather than a surprise.
#
# This runs the same synthetic-data benchmark under BOTH environments on the
# SAME card, so the two numbers are directly comparable. Nothing is read from
# disk, so the input pipeline cannot confound it.
#
# Build the modern environment first, on a login node:
#     bash env/setup_modern_tf.sh
set -e
echo "Job ran on: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
HICGAN="${SLURM_SUBMIT_DIR:-$PWD}"
BENCH="$HICGAN/env/bench_gpu_synthetic.py"
[[ -f "$BENCH" ]] || { echo "missing $BENCH" >&2; exit 1; }

echo
echo "==================== OLD STACK: TF 2.15 / CUDA 12.0 / cuDNN 8"
conda init >/dev/null 2>&1 || true
hash -r
source activate hicgan
python -c "import tensorflow as tf,os;b=tf.sysconfig.get_build_info();print('TF',tf.__version__,'CUDA',b.get('cuda_version'),'cuDNN',b.get('cudnn_version'))"
python "$BENCH" --window 512 --batch 50 --steps 25 --hicgan "$HICGAN" || echo "old stack failed"
conda deactivate || true

echo
echo "==================== NEW STACK: TF 2.21 / CUDA 12.9 / cuDNN 9.24"
source "$HICGAN/env/activate_modern_tf.sh"
python -c "import tensorflow as tf,os;b=tf.sysconfig.get_build_info();print('TF',tf.__version__,'CUDA',b.get('cuda_version'),'cuDNN',b.get('cudnn_version'))"
python "$BENCH" --window 512 --batch 50 --steps 25 --hicgan "$HICGAN" || echo "new stack failed"

echo
echo "Compare the samples/s and the mixed/float32 ratio between the two blocks."
echo "If the new stack is materially faster, or mixed precision finally helps,"
echo "the training scripts should switch to it. If both are ~63 samples/s the"
echo "card is simply not faster than an A100 for this model."
