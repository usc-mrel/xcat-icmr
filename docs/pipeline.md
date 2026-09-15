# Pipeline and data products

The normal command is:

```bash
xcat-icmr simulate configs/my_simulation.yaml
```

It executes the following dependency graph and reuses complete compatible
stages.

## 1. Anatomy and motion

XCAT produces a periodic series of cropped `uint16` tissue-label volumes in
patient coordinates. The patient position defines the patient-to-device
mapping; sequence orientation then defines the logical encoding axes.

## 2. Tissue contrast and encoding

The tissue library maps labels to bSSFP signal using sequence TE, TR, and flip
angle. The Pulseq RF profile is evaluated along its logical slab direction.
The configured RF center shift is implemented as the corresponding k-space
phase ramp; coil maps are not spatially shifted.

Each contrast state is multiplied by normalized coil sensitivities and encoded
with the 3-D SigPy NUFFT. High-resolution contrast is transient unless its
debug output is explicitly enabled.

## 3. Moving Gd balloon

Control points are interpolated at constant velocity along physical arc length.
At each acquisition time, the Gd balloon is rasterized sparsely with
partial-volume weights. Its bSSFP signal is added to the tissue signal and only
the trajectory views required at that time are encoded.

The stored canonical stream is combined tissue-plus-Gd multicoil k-space; the
toolbox does not retain redundant tissue-only and Gd-only dynamic streams.

## 4. Fully sampled reference and undersampling

The fully sampled reference is produced by forward encoding and DCF-weighted
adjoint NUFFT on the configured reconstruction grid. Balloon positions inside
each reference frame are averaged to model temporal motion blur.

The chronological view-order file is a list of zero-based trajectory-TR
indices. It may repeat or omit indices and cycles automatically if the virtual
experiment is longer than one complete acquisition. Canonical frames are
grouped into integer multiples to create the requested undersampled temporal
resolutions; incomplete final groups are dropped.

## 5. Reconstruction

```bash
xcat-icmr reconstruct configs/my_recon.yaml
```

The reconstruction HDF5 input is self-describing: it contains or references
the complex64 k-space, SigPy coordinates, DCF, effective coil maps, frame
boundaries, FOV/grid, orientation, sequence signature, and simulation
provenance. The initial backend is causal Fair-L1 IRLS with optional fixed coil
compression and resumable checkpoints.

## 6. Image quality assessment

```bash
xcat-icmr assess-image-quality configs/my_recon.yaml
```

Each completed reconstruction is paired automatically with its fully sampled
reference. Assessment reports volume correlation and NRMSE, time–arc-length
SSIM and correlation, ground-truth- and tracker-guided apparent CNR, causal 3-D
tracking errors, and parallel/perpendicular FWHM. Its primary products are
`metrics.json`, `summary_line_profiles.png`, and
`curved_profile_comparison.png`.

