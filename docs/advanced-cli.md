# Advanced command-line interface

The normal workflow needs only `simulate`, `reconstruct`, and
`assess-image-quality`. The following stage-level commands are useful for
development and diagnosis.

## Configuration and sequence

```bash
xcat-icmr validate configs/my_simulation.yaml
xcat-icmr inspect-sequence configs/my_simulation.yaml
xcat-icmr plan-acquisition configs/my_simulation.yaml
```

Use `inspect-sequence --matlab-reference FILE.mat` to compare resolved sequence
metadata with a MATLAB reference.

## Cache and acquisition

```bash
xcat-icmr inspect-cache configs/my_simulation.yaml
xcat-icmr inspect-acquisition /path/to/grouped_multicoil_kspace.h5
xcat-icmr adopt-legacy-cache configs/my_simulation.yaml
```

## Individual generation stages

```bash
xcat-icmr generate-dynamic-cycle configs/my_simulation.yaml
xcat-icmr generate-tissue-kspace-library configs/my_simulation.yaml --dry-run
xcat-icmr generate-tissue-kspace-library configs/my_simulation.yaml
xcat-icmr generate-tissue-adjoint-reference configs/my_simulation.yaml
xcat-icmr generate-dynamic-acquisition configs/my_simulation.yaml --dry-run
xcat-icmr generate-dynamic-acquisition configs/my_simulation.yaml
xcat-icmr generate-dynamic-fullysampled-reference configs/my_simulation.yaml
xcat-icmr generate-undersampled-debug configs/my_simulation.yaml
xcat-icmr generate-undersampled-acquisition configs/my_simulation.yaml
```

Existing valid frames and cache entries are reused unless the command provides
and receives `--overwrite`. Check command-specific options with:

```bash
xcat-icmr COMMAND --help
```

## Curved profile

```bash
xcat-icmr generate-curved-line-profile configs/my_simulation.yaml
```

By default this uses the matching fully sampled dynamic reference. `--input`
selects another compatible HDF5 image and `--overwrite` recomputes the profile.

