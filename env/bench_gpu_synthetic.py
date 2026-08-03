#!/usr/bin/env python3
"""Pure GPU throughput for the Hi-cGAN step: synthetic data, no file I/O.

Separates two explanations for fp32 and mixed precision giving identical
throughput on the cluster: either the GPU is saturated and precision does not
help, or the GPU is idle waiting for data and neither does.

Feeds the real model random tensors of the real shape from memory, so nothing is
read from disk. If this reports far more samples/s than the training job
achieves, the job is input-bound and no GPU or precision change will help it.
"""
import os, sys, time, argparse
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=512)
ap.add_argument("--batch", type=int, default=50)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--hicgan", default=None, help="path to the Hi-cGAN source tree")
a = ap.parse_args()
if a.hicgan:
    sys.path.insert(0, a.hicgan)
from hicgan.lib import hicGAN

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", [g.name for g in gpus] or "NONE")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def run(label, mixed):
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if mixed else "float32")
    scope = tf.distribute.OneDeviceStrategy(
        device="/GPU:0" if gpus else "/CPU:0")
    with scope.scope():
        m = hicGAN.HiCGAN(log_dir="/tmp", number_factors=1, loss_weight_pixel=100,
                          loss_weight_adversarial=1.0, loss_weight_discriminator=0.5,
                          loss_type_pixel="L2", loss_weight_tv=1e-12,
                          input_size=a.window, learning_rate_generator=2e-5,
                          learning_rate_discriminator=1e-6, adam_beta_1=0.5,
                          plot_type="png", plot_frequency=10, scope=scope,
                          mixed_precision=mixed, restore_checkpoint=False)
    f = np.random.rand(a.batch, 3 * a.window, 1, 1).astype("float32")
    t = np.random.rand(a.batch, a.window, a.window, 1).astype("float32")
    data = ({"factorData": tf.constant(f)}, {"out_matrixData": tf.constant(t)})
    t0 = None
    for i in range(a.steps + a.warmup):
        if i == a.warmup:
            t0 = time.perf_counter()
        m.distributed_train_step(data)
    dt = time.perf_counter() - t0
    sps = a.steps * a.batch / dt
    print(f"  {label:<10} {dt/a.steps:7.3f} s/step   {sps:7.1f} samples/s   "
          f"{sps/a.batch:5.2f} it/s at batch {a.batch}", flush=True)
    return sps


print(f"\nwindow {a.window}, batch {a.batch}, {a.steps} timed steps, no file I/O\n")
f32 = run("float32", False)
f16 = run("mixed", True)
print(f"\nmixed / float32 = {f16/f32:.2f}x")
print("\nThe running training job achieves 1.26 it/s at batch 50.")
print(f"This benchmark: {f32/a.batch:.2f} it/s float32, {f16/a.batch:.2f} it/s mixed.")
print("If those are much higher, the job is input-bound and neither a faster GPU")
print("nor mixed precision can help it; fix the data pipeline instead.")
