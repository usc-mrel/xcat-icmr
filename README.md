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

The normal user workflow does not require a separate validation or cache
inspection command. Run the complete cache-aware simulation pipeline with:

```bash
xcat-icmr simulate configs/my_simulation.yaml
```

Validation is performed automatically before any generation starts. Existing
compatible labels, tissue k-space, references, and acquisitions are reused.
To inspect the complete resolved plan without generating data, use:

```bash
xcat-icmr simulate configs/my_simulation.yaml --dry-run
```

The standalone validator checks the YAML schema, relationships between
sections, and required external files and remains available for diagnostics.
The individual generation commands below are advanced stage-level controls.

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
- `contrast/<id>`: optional real `float32` high-resolution debug contrast.
- `tissue_kspace/<id>`: one chunked `complex64` HDF5 file per 5 ms cardiac
  phase, ordered `[sample, trajectory_TR, coil]`. High-resolution contrast is
  calculated one phase at a time and discarded after encoding.
- `dynamic_acquisition/<id>`: the persisted combined tissue-plus-Gd multicoil
  stream, ordered `[sample, global_TR, coil]`, plus its image-only fully sampled
  tissue-plus-Gd reference when requested.

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

Validate timing, TR snapping, and the complete user-provided view order without
generating data:

```bash
xcat-icmr plan-acquisition configs/my_simulation.yaml
```

Preflight storage, then generate or resume the reusable full-trajectory tissue
library:

```bash
xcat-icmr generate-tissue-kspace-library configs/my_simulation.yaml --dry-run
xcat-icmr generate-tissue-kspace-library configs/my_simulation.yaml
```

The generator refuses to start if the configured large-cache flag is disabled
or free space is insufficient. Existing valid phase files are reused. The
current 1500-sample, 1232-arm, 16-coil, 200-phase SPI library is approximately
44 GiB in raw complex64 storage.

On GPU, the trajectory, RF-centering phase ramp, and all normalized coil-map
ROIs remain device-resident across cardiac phases. Contrast is uploaded once
per phase and coils are encoded in batches of four. Only completed k-space
batches are returned to CPU for checkpointed HDF5 storage. The adjoint stage
similarly retains its trajectory, DCF, and low-resolution coil maps on GPU and
downloads only the final coil-combined image.

Reconstruct the actual cached k-space into a resumable, coil-combined 4-D
adjoint reference before proceeding to the moving-Gd simulation:

```bash
# Quick validation after generating phase 1
xcat-icmr generate-tissue-adjoint-reference \
  configs/my_simulation.yaml --start-frame 1 --end-frame 1

# Complete 200-phase reference
xcat-icmr generate-tissue-adjoint-reference configs/my_simulation.yaml
```

The output is `tissue_fullysampled_adjoint_reference_4d.h5`, with complex64
dataset `image[logical_x, logical_y, logical_z, cardiac_phase]`. The supplied
DCF is applied before each coil adjoint and normalized sensitivity maps are
combined as `sum(conj(S) * adjoint_coil)` without a second sensitivity-power
division. Completed reference phases are reused.

Generate the stored TR-level tissue-plus-Gd acquisition and the separate
image-only fully sampled reference:

```bash
xcat-icmr generate-dynamic-acquisition configs/my_simulation.yaml --dry-run
xcat-icmr generate-dynamic-acquisition configs/my_simulation.yaml
xcat-icmr generate-dynamic-fullysampled-reference configs/my_simulation.yaml
```

Before committing to the full experiment, generate exactly one repeatable
view-order cycle and its frame-wise combined adjoint:

```bash
xcat-icmr generate-dynamic-acquisition configs/my_simulation.yaml \
  --view-order-cycles 1 \
  --save-adjoint-debug
```

This bounded output is stored below the dynamic-acquisition cache's `debug/`
directory and never marks the complete experiment as finished. It stores only
combined tissue-plus-Gd multicoil k-space and the combined frame-wise adjoint;
separate tissue-only and Gd-only k-space copies are not retained.

The canonical acquisition frame duration is defined under `acquisition`. Later
undersampled experiments group an integer number of these source frames:

```yaml
acquisition:
  frame_duration_s: 0.055

undersampling:
  enabled: true
  # Generate the unique floor/ceil integer multiples around this request.
  # With 55 ms source frames: 300 ms -> 275 ms and 330 ms.
  target_frame_duration_s: 0.300
```

Generate the bounded Step-4 validation for one complete view-order cycle:

```bash
xcat-icmr generate-undersampled-debug configs/my_simulation.yaml
```

The combined multicoil k-space is exposed through a small HDF5 virtual dataset,
so the canonical TR stream is not duplicated. The command saves a temporally
matched fully sampled reference and a DCF-weighted, coil-combined adjoint for
comparison. Human-readable records are organized by control-point filename,
velocity, grouping factor, and actual temporal resolution under the run output;
content-addressed cache IDs remain the authoritative numerical identity.

After approving that bounded debug, generate both complete bracketed
acquisitions and their fully sampled ground-truth profiles, without running a
reconstruction:

```bash
xcat-icmr generate-undersampled-acquisition configs/my_simulation.yaml
```

Each `grouped_multicoil_kspace.h5` is a versioned, self-describing
reconstruction input. Alongside the virtual complex64 k-space it stores the
exact SigPy trajectory coordinates, normalized DCF, effective low-resolution
coil maps, frame boundaries, target grid, orientation transforms, NUFFT
settings, sequence signature, and simulation provenance. A reconstruction
therefore does not need to reopen or reinterpret the simulation YAML, and it
validates the acquisition automatically. For optional diagnostics, inspect it
with:

```bash
xcat-icmr inspect-acquisition /path/to/grouped_multicoil_kspace.h5
```

The k-space remains virtual during normal use so the 275 and 330 ms groupings
do not duplicate the canonical multicoil samples. The canonical acquisition
must remain in the output cache; a future explicit bundling command can create
a portable materialized copy when required.

Reconstruction is configured separately from simulation. Copy the shipped
template, select one or more simulation YAML files and generated frame
durations, and add one or more reconstruction settings. The cache IDs and
HDF5 paths are resolved internally:

```yaml
inputs:
  simulations:
    - config: simulation.template.yaml
      frame_duration_s: 0.330
```

Advanced users may instead use `inputs.acquisitions` with explicit
`grouped_multicoil_kspace.h5` paths.

```bash
cp configs/recon_config.template.yaml configs/my_recon.yaml
xcat-icmr reconstruct configs/my_recon.yaml --dry-run
xcat-icmr reconstruct configs/my_recon.yaml
```

The dry run validates every acquisition and expands every reconstruction
setting over every input. Acquisition and reconstruction names are generated
from embedded provenance and validated parameters. Results will be placed in
readable paths such as
`reconstructions/causal-irls/o3_cg3_lt0p005_ls0p0005__8fff70bd`; the suffix
prevents collisions. Optional PCA or fixed-weight ROVIR coil compression is
controlled under `preprocessing.coil_compression`. Disabled compression does
not affect the
reconstruction identity. The first backend is causal Fair-L1 IRLS, adapted
from the established NIH SPI reconstruction. It reads the trajectory, DCF,
sensitivity maps, frame boundaries, and NUFFT kernel settings directly from
the acquisition HDF5. It writes a resumable `reconstruction.h5`, per-frame
costs and timings, and `result.json`. Completed outputs are reused; pass
`--overwrite` to replace the matching output.

Compare a completed reconstruction with the fully sampled reference recorded
in its source acquisition. Passing the reconstruction config assesses every
completed job selected by that config, so users do not need to copy generated
output paths or cache IDs:

```bash
xcat-icmr assess-image-quality configs/my_recon.yaml
```

Direct reconstruction directories, `result.json`, and `reconstruction.h5`
paths are also accepted. In a batch, incomplete jobs are reported and skipped;
the command continues assessing completed jobs and returns a nonzero status if
anything was skipped or failed. `assess-reconstruction` remains available as
a compatibility alias.

The assessment applies one global magnitude scale and retains three primary
outputs: `metrics.json`, a six-panel `summary_line_profiles.png` with tracking,
apparent-CNR, localization-error, and FWHM diagnostics, and
`curved_profile_comparison.png`. The curved comparison shows the ground truth,
the reconstruction, and a false-color fusion in which ground truth is magenta,
reconstruction is green, and agreement is gray/white. It does not create a
full 4-D difference volume or separate frame/montage diagnostics.

The metrics retain ground-truth-assisted path localization, apparent CNR, and
FWHM while also reporting a causal 3-D tracker. Causal errors are decomposed
into parallel, two normal-direction, perpendicular-magnitude, and total 3-D
components. Apparent CNR is reported separately at the known and causally
tracked positions. Parallel FWHM is measured along the curved path;
perpendicular FWHM is retained for both local normal directions and summarized
by their median.

Planned paired reconstruction assessment will associate every reconstruction
with this matching fully sampled reference and report SSIM, RMSE, correlation,
apparent CNR, balloon visibility, tracking error, and FWHM both along and
perpendicular to the catheter trajectory. This IQA stage follows validation of
the grouped acquisition and is not yet implemented.

The acquisition view-order file contains a nonempty integer list of zero-based
trajectory-TR indices; it does not require plane or interleave metadata.
Repeated and omitted indices are valid, and the list cycles automatically over
the experiment. For every simulation TR, the simulator gathers one trajectory
TR from the matching periodic tissue phase, evaluates the moving balloon as a
sparse partial-volume object, encodes only that selected Gd trajectory TR, and
saves their additive sum. Incomplete final image frames are dropped.

The reference still honors the two tissue-reference grids in the YAML. For
example:

```yaml
timeline:
  xcat_time_step_s: 0.005
  reference_time_step_s: 0.010
  xcat_to_reference: average
```

This averages XCAT frames 1-2 before generating reference frame 1, frames 3-4
before reference frame 2, and so on. Averaging occurs on the cropped
high-resolution PCS grid before orientation and padding, and one multicoil
forward/adjoint NUFFT is performed per aggregated frame. `center` selects the
central XCAT frame of each time window.
`trajectory-aware` is reserved for future sample-time-aware encoding and
currently raises a not-implemented error. The complete motion cycle must divide
evenly into the selected reference time step. This tissue-reference interval
is independent of the Pulseq TR; dynamic acquisition uses exact Pulseq sample
timestamps.

The dynamic fully sampled reference averages the tissue states and sparse Gd
positions within each acquisition frame before producing the final image; this
models balloon motion blur. Its `image` dataset is ordered
`[logical_x, logical_y, logical_z,time]`. Temporary full-trajectory reference
k-space is discarded. The generator reuses the complete approved tissue
adjoint reference and keeps the full trajectory and normalized coil maps
resident on the selected GPU while encoding successive Gd frames. Completed
output frames are resumable and a complete output is returned as a cache hit.
The HDF5 image can be read in MATLAB with
`h5read(filename, '/image')` or in Python with `h5py.File(filename)['image']`.

Generate a curved-line intensity profile from that fully sampled reference:

```bash
xcat-icmr generate-curved-line-profile configs/my_simulation.yaml
```

The analysis samples the configured catheter path at fixed arc-length spacing
and records the centerline, mean, and maximum magnitude inside a transverse
tube at every time frame. It writes a MATLAB file, a time-distance heatmap, a
geometry overlay, and metadata below the reference's
`analysis/curved_line_profile/` directory. Set
`analysis.curved_line_profile.enabled: true` to run it automatically after
`generate-dynamic-fullysampled-reference`. Use `--input FILE.h5` for a
compatible alternate reference and `--overwrite` when the analysis inputs or
settings have changed.
