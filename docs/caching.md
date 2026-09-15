# Caching and outputs

XCAT-iCMR separates content-addressed numerical caches from readable
experiment results.

## Cache levels

| Cache | Contents | Typical type/layout |
|---|---|---|
| `labels/<id>` | One periodic XCAT tissue cycle | `uint16` MAT volumes |
| `contrast/<id>` | Optional retained high-resolution contrast | real `float32` |
| `tissue_kspace/<id>` | Periodic tissue encoding | per-phase `complex64` HDF5, `[sample, trajectory_TR, coil]` |
| `dynamic_acquisition/<id>` | Tissue-plus-Gd canonical stream and fully sampled reference | `complex64` k-space and 4-D image |
| `undersampled_acquisition/<id>` | Temporally grouped reconstruction input | self-describing HDF5 |

An identity contains the inputs that affect that artifact. For example, an
XCAT change creates a new label identity, while changing only reconstruction
iterations does not. Compatible entries are reused and incomplete entries are
resumed where the stage supports checkpoints.

The current reference SPI example contains 200 tissue phases. Its full
1500-sample × 1232-arm × 16-coil periodic tissue library occupies roughly
44 GiB as `complex64`; users should preflight large runs and avoid unnecessary
copies.

## Human-readable results

Run outputs organize acquisitions by control-point filename, velocity, and
actual temporal resolution. Reconstruction directories add the method and a
readable parameter recipe, with a short digest only for collision prevention.
Users normally refer to simulation and reconstruction YAML files rather than
these paths.

## Inspection

The top-level pipeline performs required validation automatically. These
commands are optional diagnostics:

```bash
xcat-icmr simulate configs/my_simulation.yaml --dry-run
xcat-icmr inspect-cache configs/my_simulation.yaml
xcat-icmr inspect-acquisition /path/to/grouped_multicoil_kspace.h5
```

The undersampled HDF5 can use virtual datasets to avoid duplicating the
canonical k-space stream. Keep its referenced cache entry when moving results,
unless a future materialization step is used to make it standalone.

