# Slim runtime image for the HashToken next-gen miner.
#
# ~180 MB of content against 8,200 MB for vastai/base-image:cuda-*-cudnn-devel
# and 4,130 MB for cupy/cupy. On a slow host that is the difference between a
# 20-minute boot and well under a minute -- and boot time is not cosmetic: an
# interruptible rig was preempted mid-`pip install` on 2026-08-05 because the
# 8.2 GB pull took longer than the bid could be held.
#
#   base python:3.12-slim              40 MB
#   cupy-cuda13x + numpy + requests    87 MB
#   cuda-toolkit[cudart,nvrtc]==13.*   54 MB
#
# NO CUDA TOOLKIT, NO nvcc. Measured on an RTX 5090: the NVRTC fallback in
# _compile_kernel() is within 0.03% of nvcc (5,813 vs 5,803 MH/s), which is what
# makes dropping ~8 GB possible. libcuda.so and nvidia-smi come from the HOST via
# the NVIDIA container runtime, not from this image.
#
# cuda13x, not cuda12x: every RTX 5090 offer on vast reports cuda_max_good
# 13.0-13.2. Installing the 12x wheel over a preinstalled 13x one produces the
# "multiple CuPy packages" collision and undefined behaviour.
#
# The MINER SOURCE is deliberately NOT baked in -- rig_boot fetches it at boot.
# That keeps this tag frozen so host image caches accumulate, while miner changes
# ship without a rebuild.
FROM python:3.12-slim

# ca-certificates for HTTPS. curl is NOT required: the boot script uses python's
# urllib, which is already here, so we skip ~10 MB of apt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pinned so the tag is reproducible and host caches stay valid.
# cuda-toolkit is installed with ONLY cudart and nvrtc -- cublas, cufft, curand,
# cusolver and cusparse are several hundred MB and this kernel uses none of them.
RUN pip install --no-cache-dir \
        "cupy-cuda13x==14.1.1" \
        "numpy==2.2.6" \
        "requests==2.34.2" \
        "cuda-toolkit[cudart,nvrtc]==13.*"

# CuPy JIT-compiles the kernel on first use; give it a writable cache.
ENV CUPY_CACHE_DIR=/opt/cupy-cache
RUN mkdir -p /opt/cupy-cache && chmod 777 /opt/cupy-cache

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
