"""Single-pass validation and comparison of XCAT and MATLAB label volumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from xcat_icmr.phantom.binary import XcatBinaryVolume
from xcat_icmr.tissue import XcatLabel


class XcatLabelComparisonError(Exception):
    """Raised when label validation or MATLAB comparison cannot proceed."""


@dataclass(frozen=True)
class XcatLabelComparison:
    """Voxelwise comparison with validation collected in the same pass."""

    binary_path: Path
    matlab_path: Path
    logical_shape: tuple[int, int, int]
    matlab_stored_shape: tuple[int, int, int]
    voxel_count: int
    unique_labels: tuple[int, ...]
    mismatch_count: int
    max_abs_error: float

    @property
    def passed(self) -> bool:
        return self.mismatch_count == 0


def _validated_integer_labels(
    values: np.ndarray,
    *,
    maximum_label: int,
    first_slice: int,
) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise XcatLabelComparisonError(
            f"XCAT labels contain non-finite values near slice {first_slice}"
        )
    if not np.all(values == np.floor(values)):
        invalid = values[values != np.floor(values)]
        raise XcatLabelComparisonError(
            "XCAT labels must be integer-valued; found "
            f"{float(invalid.flat[0]):g} near slice {first_slice}"
        )
    integers = values.astype(np.int16, copy=False)
    valid = (integers >= 0) & (integers <= maximum_label)
    if not np.all(valid):
        invalid = np.unique(integers[~valid])
        raise XcatLabelComparisonError(
            f"XCAT labels outside 0--{maximum_label}: "
            + ", ".join(str(int(value)) for value in invalid[:8])
        )
    return integers


def validate_xcat_labels(
    volume: XcatBinaryVolume,
    *,
    chunk_slices: int = 8,
) -> tuple[int, ...]:
    """Validate a cropped label volume without loading it all into memory."""

    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    maximum_label = max(int(label) for label in XcatLabel)
    unique_labels: set[int] = set()
    for start in range(0, volume.cropped_shape[2], chunk_slices):
        stop = min(start + chunk_slices, volume.cropped_shape[2])
        integer_chunk = _validated_integer_labels(
            np.asarray(volume.cropped[:, :, start:stop]),
            maximum_label=maximum_label,
            first_slice=start,
        )
        unique_labels.update(int(value) for value in np.unique(integer_chunk))
    return tuple(sorted(unique_labels))


def compare_xcat_labels_to_matlab(
    volume: XcatBinaryVolume,
    matlab_path: str | Path,
    *,
    dataset_name: str = "P",
    chunk_slices: int = 8,
) -> XcatLabelComparison:
    """Validate cropped XCAT labels while comparing with MATLAB v7.3 ``P``."""

    if chunk_slices <= 0:
        raise ValueError("chunk_slices must be positive")
    reference_path = Path(matlab_path).expanduser().resolve(strict=False)
    if not reference_path.is_file():
        raise XcatLabelComparisonError(
            f"MATLAB label reference does not exist: {reference_path}"
        )

    try:
        handle = h5py.File(reference_path, "r")
    except OSError as exc:
        raise XcatLabelComparisonError(
            f"could not open MATLAB v7.3 reference {reference_path}: {exc}"
        ) from exc

    maximum_label = max(int(label) for label in XcatLabel)
    unique_labels: set[int] = set()
    mismatch_count = 0
    max_abs_error = 0.0
    with handle:
        if dataset_name not in handle:
            raise XcatLabelComparisonError(
                f"MATLAB reference is missing dataset {dataset_name!r}"
            )
        dataset = handle[dataset_name]
        if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
            raise XcatLabelComparisonError(
                f"MATLAB dataset {dataset_name!r} must be three-dimensional"
            )

        expected_stored_shape = tuple(reversed(volume.cropped_shape))
        if dataset.shape != expected_stored_shape:
            raise XcatLabelComparisonError(
                "MATLAB stored shape does not match the reversed Python "
                f"logical shape: {dataset.shape} != {expected_stored_shape}"
            )

        for start in range(0, volume.cropped_shape[2], chunk_slices):
            stop = min(start + chunk_slices, volume.cropped_shape[2])
            python_chunk = np.asarray(volume.cropped[:, :, start:stop])
            integer_chunk = _validated_integer_labels(
                python_chunk,
                maximum_label=maximum_label,
                first_slice=start,
            )
            unique_labels.update(int(value) for value in np.unique(integer_chunk))

            # MATLAB v7.3 stores array axes in reverse order for HDF5 access.
            matlab_chunk = np.asarray(dataset[start:stop, :, :]).transpose(2, 1, 0)
            if not np.all(np.isfinite(matlab_chunk)):
                raise XcatLabelComparisonError(
                    f"MATLAB labels contain non-finite values near slice {start}"
                )
            difference = np.abs(
                python_chunk.astype(np.float64, copy=False)
                - matlab_chunk.astype(np.float64, copy=False)
            )
            mismatch_count += int(np.count_nonzero(difference))
            if difference.size:
                max_abs_error = max(
                    max_abs_error, float(np.max(difference))
                )

        stored_shape = tuple(dataset.shape)

    return XcatLabelComparison(
        binary_path=volume.path,
        matlab_path=reference_path,
        logical_shape=volume.cropped_shape,
        matlab_stored_shape=stored_shape,
        voxel_count=int(np.prod(volume.cropped_shape, dtype=np.int64)),
        unique_labels=tuple(sorted(unique_labels)),
        mismatch_count=mismatch_count,
        max_abs_error=max_abs_error,
    )


def format_xcat_label_comparison(report: XcatLabelComparison) -> str:
    """Format XCAT label validation and MATLAB comparison results."""

    return "\n".join(
        (
            "XCAT label comparison",
            f"Binary:               {report.binary_path}",
            f"MATLAB reference:     {report.matlab_path}",
            f"Python logical shape: {report.logical_shape}",
            f"MATLAB stored shape:  {report.matlab_stored_shape}",
            f"Voxels compared:      {report.voxel_count:,}",
            (
                "Validated labels:     "
                + ", ".join(str(label) for label in report.unique_labels)
            ),
            f"Mismatch voxels:      {report.mismatch_count:,}",
            f"Maximum |Δ|:          {report.max_abs_error:g}",
            f"Overall:              {'PASS' if report.passed else 'FAIL'}",
        )
    )
