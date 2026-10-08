#!/usr/bin/env bash
# Environment fixes needed to run Triton / inductor on this machine (containerized GPU
# environment).
#
#   source tools/gpu_env.sh
#
# Background: inside the container, /usr/lib/x86_64-linux-gnu/libcuda.so points at a
# **0-byte stub**, while the real driver lives in /usr/local/nvidia/lib64/. torch loads
# libcuda via ctypes with the default RTLD_LOCAL, so the symbols never enter the global
# scope — as a result Triton's driver extension (cuda_utils) fails to load with
# `undefined symbol: cuModuleGetFunction`, and torch.compile's inductor backend is
# likewise unusable. LD_PRELOAD puts the real driver into the global scope.
#
# Note: PYTHONNOUSERSITE=1 avoids a broken torch under ~/.local (missing
# libnvshmem_host.so.3), which would otherwise shadow the working copy under /usr/local.

export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so.1${LD_PRELOAD:+:$LD_PRELOAD}
export PYTHONNOUSERSITE=1
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
