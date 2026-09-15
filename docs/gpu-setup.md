# GPU setup

The current accelerator path uses CuPy through SigPy for single-GPU 3-D NUFFT
operations.

## Install

Verify that the GPU is visible in the same shell, container, or scheduler job
that will run the toolbox:

```bash
nvidia-smi
ls /dev/nvidiactl /dev/nvidia0
```

Then create the environment and install the CUDA 12 extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpu]"
```

Verify CuPy:

```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount()); cp.show_config()"
```

Use `compute.device_id: 0` for the first visible GPU. Device numbering follows
the environment visible inside the process, including `CUDA_VISIBLE_DEVICES`.

Do not install another `cupy`, `cupy-cuda11x`, or `cupy-cuda13x` distribution
in the same environment. A warning about unavailable or incompatible NCCL does
not affect the current single-GPU implementation; NCCL is needed for
multi-GPU collectives, which the toolbox does not currently use.

The GPU implementation keeps static trajectories, phase ramps, DCF, and coil
ROIs resident where practical. Completed k-space batches and final combined
images are returned to CPU for checkpointed storage.

