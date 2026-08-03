#!/usr/bin/env python3
"""Raw matmul and convolution throughput. No Hi-cGAN, no data pipeline.

Written because "the new generation is not one bit faster" is hard to believe,
and the PTX-JIT explanation for it was inferred from build flags rather than
measured. This measures the hardware directly, so it cannot be confounded by the
model, the input pipeline or the training loop.

The decisive number is the fp16/fp32 ratio for the matmul. Tensor cores make
fp16 several times faster than fp32 on every card since Volta. If a card reports
fp16 ~= fp32, its tensor cores are not being used, which is exactly what PTX-JIT
from an older compute capability would cause.

Reference peaks: A100-80g about 19.5 TFLOPS fp32 and 312 TFLOPS fp16 tensor;
RTX PRO 6000 Blackwell substantially above both. Anything within a factor of two
of the fp32 figure for fp16 means the tensor cores are idle.
"""
import argparse, time
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=8192)
ap.add_argument("--iters", type=int, default=30)
a = ap.parse_args()

b = tf.sysconfig.get_build_info()
print(f"TF {tf.__version__}  built against CUDA {b.get('cuda_version')} "
      f"cuDNN {b.get('cudnn_version')}")
print(f"built for: {b.get('cuda_compute_capabilities')}")
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", [g.name for g in gpus] or "NONE")
if not gpus:
    raise SystemExit("no GPU visible")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
d = tf.config.experimental.get_device_details(gpus[0])
print(f"device: {d.get('device_name')}  compute capability: "
      f"{d.get('compute_capability')}")


def matmul(dtype, n, iters):
    x = tf.random.normal((n, n), dtype=tf.float32)
    x = tf.cast(x, dtype)
    y = tf.cast(tf.random.normal((n, n), dtype=tf.float32), dtype)
    f = tf.function(lambda a_, b_: tf.linalg.matmul(a_, b_))
    for _ in range(5):
        r = f(x, y)
    _ = r.numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(x, y)
    _ = r.numpy()
    dt = time.perf_counter() - t0
    return 2.0 * n**3 * iters / dt / 1e12


def conv(dtype, iters):
    x = tf.cast(tf.random.normal((16, 512, 512, 32)), dtype)
    k = tf.cast(tf.random.normal((3, 3, 32, 64)), dtype)
    f = tf.function(lambda a_, b_: tf.nn.conv2d(a_, b_, strides=1, padding="SAME"))
    for _ in range(5):
        r = f(x, k)
    _ = r.numpy()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(x, k)
    _ = r.numpy()
    dt = time.perf_counter() - t0
    flops = 2.0 * 16 * 512 * 512 * 3 * 3 * 32 * 64
    return flops * iters / dt / 1e12


print(f"\nmatmul {a.size}x{a.size}, {a.iters} iterations")
m32 = matmul(tf.float32, a.size, a.iters)
m16 = matmul(tf.float16, a.size, a.iters)
print(f"  float32 {m32:8.1f} TFLOPS")
print(f"  float16 {m16:8.1f} TFLOPS")
print(f"  fp16/fp32 = {m16/m32:.2f}x   <-- tensor cores idle if this is near 1")

print(f"\nconv2d 16x512x512x32 -> 64ch, 3x3")
c32 = conv(tf.float32, a.iters)
c16 = conv(tf.float16, a.iters)
print(f"  float32 {c32:8.1f} TFLOPS")
print(f"  float16 {c16:8.1f} TFLOPS")
print(f"  fp16/fp32 = {c16/c32:.2f}x")
