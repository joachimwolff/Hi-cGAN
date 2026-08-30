# Running Hi-cGAN on Blackwell (RTX PRO 6000, `gpu:b6000`)

## The problem

Two training jobs on b6000 cards, identical except for precision, ran at exactly
the same speed and were **slower than an A100**:

| stack | GPU | precision | it/s at batch 50 |
|---|---|---|---|
| TF 2.15 / CUDA 12.0 / cuDNN 8 | A100-80g | float32 | 1.45 |
| TF 2.15 / CUDA 12.0 / cuDNN 8 | b6000 | float32 | 1.26 |
| TF 2.15 / CUDA 12.0 / cuDNN 8 | b6000 | mixed_float16 | 1.26 |

Mixed precision was genuinely active (`global policy = mixed_float16` in the
log) and the card was correctly detected as an RTX PRO 6000 Blackwell with
97,887 MiB. The GPUs showed 100 % utilisation throughout.

## The cause

    TF 2.15.0, CUDA 12.0, cuDNN 8
    built for: sm_60 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90 compute_90

Blackwell is `sm_120` (RTX PRO 6000, GB202) or `sm_100` (datacenter B100/B200).
**Neither is in that list.** Every kernel therefore reaches the card through
forward-compatible PTX JIT from `compute_90`, and cuDNN 8 has no Blackwell
kernels at all. 100 % utilisation is consistent with this: utilisation counts
whether kernels are resident, not whether they are efficient. The fp16
tensor-core paths never engage, so `--mixedPrecision` costs a cast and buys
nothing.

## What a newer stack does and does not fix

`env/setup_modern_tf.sh` installs TF 2.21 with CUDA 12.9 and cuDNN 9.24.

**It does not add native TensorFlow kernels for Blackwell.** Even TF 2.21 wheels
are built for `sm_60/70/80/89 + compute_90`; no TensorFlow pip wheel targets
sm_100 or sm_120. TF's own ops still JIT.

**It does replace the maths libraries with Blackwell-aware ones.** CUDA 12.9 and
cuDNN 9.24 both have native Blackwell kernels, and convolutions and GEMMs -
which are the bulk of this model's work - go through cuDNN and cuBLAS rather
than through TensorFlow's own kernels. Whether that is enough is **unmeasured**;
`env/slurm_bench_b6000_modern.sh` runs both stacks on the same card to find out.

If it is not enough, the remaining options are the NVIDIA NGC TensorFlow
container, which is built for current architectures, or a source build with
CUDA 12.8+ and `sm_120` in the target list.

## Code compatibility: nothing had to change

Tested on this workstation (RTX 4090, sm_89, driver 580.173.02) under TF 2.21,
in **both** Keras 3 and legacy Keras 2 (`TF_USE_LEGACY_KERAS=1`):

| stage | Keras 2 | Keras 3 |
|---|---|---|
| model construction | OK | OK |
| `distributed_train_step` | OK | OK |
| `model.save(..., save_format="keras")` | OK | OK |
| `load_model(custom_objects, safe_mode=False)` | OK | OK |
| `tf.train.Checkpoint` save/restore | OK | OK |

`save_format="keras"` was expected to fail under Keras 3 and does not. The single
custom layer, `CustomReshapeLayer`, already defines `get_config` and round-trips.
**No source file was modified on this branch** - the migration is entirely an
environment change.

## Does the new stack help on hardware that IS supported?

No. On the RTX 4090 (sm_89, native in both stacks), window 512, batch 32:

| stack | float32 | mixed | speedup |
|---|---|---|---|
| TF 2.15 / CUDA 12.0 / cuDNN 8 | 57.2 samples/s | 88.3 | 1.54x |
| TF 2.21 / CUDA 12.9 / cuDNN 9.24 | 56.2 samples/s | 76.1 | 1.35x |

float32 is unchanged and mixed precision is *worse*. So there is no reason to
migrate for its own sake. The only argument for the new stack is Blackwell, and
that argument is untested until the benchmark runs on a b6000.

## Usage

    bash env/setup_modern_tf.sh              # once, on a login node
    source env/activate_modern_tf.sh         # before every use
    sbatch env/slurm_bench_b6000_modern.sh   # measure both stacks on one card

`activate_modern_tf.sh` is required, not optional: TF 2.21 installed by pip does
not find its own bundled CUDA libraries and reports an empty GPU list until the
`nvidia-*` package lib directories are on `LD_LIBRARY_PATH`.
