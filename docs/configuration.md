# Configuration

XCAT-iCMR separates the simulated experiment from reconstruction choices.
Start from the supplied files rather than editing them in place:

```bash
cp configs/simulation.template.yaml configs/my_simulation.yaml
cp configs/recon_sweep.example.yaml configs/my_recon.yaml
```

## Simulation configuration

The main sections of `simulation.template.yaml` are:

| Section | Purpose |
|---|---|
| `run`, `compute` | Output root, run label, CPU/GPU device, and precision |
| `scanner` | Field strength and optional physical effects |
| `phantom`, `motion` | XCAT inputs, patient position, crop, cardiac and respiratory motion |
| `timeline` | XCAT and tissue-reference temporal sampling |
| `sequence` | Pulseq file, trajectory metadata, contrast, and RF profile |
| `coils`, `encoding` | Sensitivity maps, logical axis order, FOV, NUFFT parameters, and DCF |
| `intervention` | Catheter control points, velocity, size, dilution, and Gd signal model |
| `acquisition` | Canonical frame duration and chronological TR view order |
| `undersampling`, `noise` | Output temporal grouping and optional noise model |
| `outputs`, `analysis` | Retained artifacts, NRRD exports, and curved-profile settings |

Relative paths are resolved from the YAML file. `compute.device_id: -1` selects
CPU; `0`, `1`, and so forth select visible GPUs.

### Time scales

The time settings describe different stages and should not be conflated:

- `timeline.xcat_time_step_s`: interval between XCAT anatomical states.
- `timeline.reference_time_step_s`: interval used for periodic tissue encoding.
- `acquisition.frame_duration_s`: canonical fully sampled acquisition block.
- `undersampling.target_frame_duration_s`: requested experimental temporal
  resolution.

For the supplied SPI example these are 5 ms, 5 ms, 55 ms, and 300 ms. The
undersampling stage generates the unique integer multiples bracketing the
request: 5 × 55 ms = 275 ms and 6 × 55 ms = 330 ms.

`timeline.xcat_to_reference` controls how several XCAT states contribute to a
coarser tissue reference. `average` averages the states; `center` selects the
central state. `trajectory-aware` is reserved and raises a not-implemented
error.

### Motion and duration

Supported motion modes are `no-motion`, `breath-hold`, and `free-breathing`.
Respiratory frequency is used only for free breathing. When total duration is
`auto`, the catheter path length and configured velocity determine the virtual
experiment duration.

The catheter path is a markups JSON in the same physical frame used by the
configured phantom. The simulator interpolates it by arc length and evaluates
a sparse, partial-volume balloon at the acquisition times.

## Reconstruction configuration

A reconstruction file selects generated acquisitions without exposing cache
IDs:

```yaml
inputs:
  simulations:
    - config: simulation.template.yaml
      frame_duration_s: 0.275
    - config: simulation.template.yaml
      frame_duration_s: 0.330
```

Every entry under `reconstructions` is applied to every selected acquisition.
This makes compact, explicit hyperparameter sweeps possible:

```yaml
reconstructions:
  - method: causal-irls
    parameters:
      outer_iterations: 3
      cg_iterations: 3
      spatial_regularization: 0.0005
      temporal_regularization: 0.005
      fair_l1_delta: 0.001
      regularization_scale: 1.0
```

Optional PCA or fixed-weight ROVIR coil compression is configured once under
`preprocessing.coil_compression`. Output names are derived automatically from
the validated acquisition and reconstruction parameters.

