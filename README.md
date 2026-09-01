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

## Reusable simulation artifacts

Expensive products are stored under `outputs/cache` using IDs derived from
the inputs that affect their numerical content. Changing `run.id`, the
run-specific directory under the same `outputs` root, undersampling, noise, or
Gd-balloon settings does not erase or replace a compatible tissue cache. A
changed XCAT, contrast, sequence, trajectory, coil, or NUFFT input resolves to
a different cache ID, so older results remain available.

The cache is split into three dependency levels:

- `labels/<id>`: one `uint16` XCAT tissue-label cycle.
- `contrast/<id>`: one real `float32` high-resolution bSSFP tissue cycle.
- `tissue_kspace/<id>`: per-frame `complex64` fully sampled tissue k-space,
  shared trajectory metadata, and one `complex64` 4-D low-resolution
  reference for the complete group.

Inspect the IDs and whether each level is missing, partial, or complete:

```bash
xcat-icmr inspect-cache configs/my_simulation.yaml
```

Outputs created by the earlier run-scoped layout can be adopted without
rerunning XCAT or contrast generation. Labels are converted to `uint16` one
frame at a time. Compatible contrast files are hard-linked when the filesystem
supports it, avoiding a second copy:

```bash
xcat-icmr adopt-legacy-cache configs/my_simulation.yaml
```

Generate or resume the fully sampled tissue cache:

```bash
xcat-icmr generate-tissue-kspace-cache configs/my_simulation.yaml
```

The tissue cache honors the two temporal grids in the YAML. For example:

```yaml
timeline:
  xcat_time_step_s: 0.005
  kspace_time_step_s: 0.010
  xcat_to_kspace: average
```

This averages XCAT frames 1-2 before generating k-space frame 1, frames 3-4
before frame 2, and so on. Averaging occurs on the cropped high-resolution PCS
grid before orientation and padding, and one multicoil NUFFT is performed per
aggregated frame. `center` selects the central XCAT frame of each time window.
`trajectory-aware` is reserved for future sample-time-aware encoding and
currently raises a not-implemented error. The complete motion cycle must divide
evenly into the selected k-space time step.

Each frame MAT file contains only `kspace`, ordered as
`[sample, arm, coil]`. Shared coordinates, DCF, timing, orientation, and NUFFT
settings are in `tissue_kspace_metadata.mat`. The low-resolution reference is
stored once as `fullysampled_reference_4d.h5`; its `image` dataset is ordered
`[logical_x, logical_y, logical_z, time]`. It can be read in MATLAB with
`h5read(filename, '/image')` or in Python with `h5py.File(filename)['image']`.
