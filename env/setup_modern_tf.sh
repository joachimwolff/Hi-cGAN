#!/bin/bash
# Build a TensorFlow environment new enough to have Blackwell-capable CUDA.
#
# The training environment is TF 2.15 / CUDA 12.0 / cuDNN 8. cuDNN 8 predates
# Blackwell entirely, which is why an RTX PRO 6000 runs the model no faster than
# an A100 and why --mixedPrecision changes nothing on it: the fp16 tensor-core
# paths never engage.
#
# What this installs, and the honest limitation:
#   TF 2.21 wheels are still built for sm_60/70/80/89 + compute_90 PTX. There is
#   NO native sm_100 or sm_120 in any TensorFlow pip wheel, so TF's own kernels
#   still reach Blackwell through PTX JIT. What DOES change is the bundled
#   runtime: CUDA 12.9 and cuDNN 9.24, both of which have native Blackwell
#   kernels. Convolutions and GEMMs are the bulk of this model's work and go
#   through cuDNN/cuBLAS, so the gain could be large -- but it is unmeasured
#   until someone runs slurm_bench_b6000_modern.sh on a b6000 node.
#
# Verified on this workstation (RTX 4090, sm_89, driver 580.173.02): the code
# runs unchanged under BOTH Keras 3 and legacy Keras 2. Construction, a train
# step, model save, load_model and tf.train.Checkpoint all pass. No source
# change was needed for the migration.
set -e
VENV="${1:-$HOME/venvs/hicgan-tf-modern}"
python3.12 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "tensorflow[and-cuda]" tf-keras \
    cooler numpy pandas scipy matplotlib pydot tqdm h5py \
    pyBigWig hicmatrix scikit-learn hicrep
echo
echo "installed into $VENV"
"$VENV/bin/python" - <<'PY'
import tensorflow as tf
b = tf.sysconfig.get_build_info()
print("TF", tf.__version__, "built against CUDA", b.get("cuda_version"),
      "cuDNN", b.get("cudnn_version"))
print("built for:", b.get("cuda_compute_capabilities"))
print("NOTE: no sm_100/sm_120 -> TF's own kernels JIT from compute_90 PTX on Blackwell")
PY
echo
echo "Source env/activate_modern_tf.sh before running anything: TensorFlow does"
echo "not find the pip-installed CUDA libraries without LD_LIBRARY_PATH."
