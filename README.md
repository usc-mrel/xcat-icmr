# XCAT-iCMR

XCAT-iCMR is a modular simulation package for dynamic interventional
cardiovascular MRI using XCAT phantoms.

The project is currently a scaffold. Planned stages include phantom
generation, MR contrast, Gd balloon simulation, k-space encoding,
undersampling, and noise.

## Requirements

- Python 3.10 or newer.
- The XCAT executable and Pulseq inputs referenced by the simulation YAML.
- For GPU NUFFT: an NVIDIA GPU with a working driver. `nvidia-smi` must work
  in the same shell or container that will run XCAT-iCMR.

CUDA is optional. CPU and GPU installations are kept separate so the same
project can be installed on workstations, clusters, and CPU-only systems.

## CPU development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Use `compute.device_id: -1` in the YAML for CPU execution.

## GPU development setup

First verify that the GPU is visible in the environment where the simulation
will run:

```bash
nvidia-smi
ls /dev/nvidiactl /dev/nvidia0
```

If either command fails, fix the host, scheduler allocation, or container GPU
pass-through before installing Python GPU packages. CuPy cannot repair missing
NVIDIA device access.

Create the environment and install the CUDA 12 GPU extra:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,gpu]"
```

The `gpu` extra installs the official CUDA 12 CuPy wheel together with its CUDA
component dependencies. A compatible NVIDIA driver is still required on the
host. Do not install another `cupy`, `cupy-cuda11x`, or `cupy-cuda13x` package
in the same environment.

Verify the installation:

```bash
python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount()); cp.show_config()"
```

CuPy may warn about an unavailable or incompatible NCCL library on systems
with a separate system CUDA installation. NCCL is not required by the current
single-GPU SigPy NUFFT implementation; it is needed only if multi-GPU
collective operations are added later.

Use `compute.device_id: 0` for the first visible GPU, `1` for the second, and
so on. In a scheduler or container, these IDs refer to the GPUs visible inside
that job or container.

To reproduce the environment later, create a fresh virtual environment and
repeat the appropriate editable install command above. Package version ranges
are declared in `pyproject.toml`; the CUDA major version is intentionally an
explicit installation choice.

## Configuration and validation

Create a simulation configuration from the provided template:

```bash
cp configs/simulation.template.yaml configs/my_simulation.yaml
```

Validate it without running XCAT or generating simulation data:

```bash
xcat-icmr validate configs/my_simulation.yaml
```

The validator checks the YAML schema, relationships between sections, and
required external files.

Inspect the Pulseq signature, resolved MATLAB metadata, and oriented
trajectory without running the simulation:

```bash
xcat-icmr inspect-sequence configs/my_simulation.yaml
```

An existing MATLAB v7.3 `par` file can be checked field-by-field:

```bash
xcat-icmr inspect-sequence configs/my_simulation.yaml \
  --matlab-reference /path/to/par_reference.mat
```
