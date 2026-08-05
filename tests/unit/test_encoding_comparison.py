from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from xcat_icmr.encoding import (
    compare_device_references,
    validate_image_reference,
)


def test_device_comparison_records_small_numerical_difference(tmp_path: Path) -> None:
    kspace = np.ones((3, 2), dtype=np.complex64)
    adjoint = np.ones((4, 4, 4), dtype=np.complex64)
    cpu = tmp_path / "cpu.mat"
    gpu = tmp_path / "gpu.mat"
    savemat(cpu, {"kspace": kspace, "adjoint": adjoint, "elapsed_s": [[2.0]]})
    savemat(
        gpu,
        {
            "kspace": kspace * (1 + 1e-5),
            "adjoint": adjoint * (1 + 1e-5),
            "elapsed_s": [[0.5]],
        },
    )

    report = compare_device_references(cpu, gpu, tmp_path / "parity.mat")

    assert report.passed
    assert report.speedup == 4.0
    assert loadmat(report.output_path)["passed"].item() == 1


def test_image_reference_validation_preserves_orientation(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[1:6, 2:5, 3:7] = 1
    image[1:3, 2:4, 3:5] = 2
    gt = tmp_path / "gt.mat"
    reference = tmp_path / "reference.mat"
    savemat(gt, {"image": image})
    savemat(reference, {"adjoint_rss": image, "fov_mm": [[8, 8, 8]]})

    report = validate_image_reference(
        gt, reference, tmp_path / "image_validation.mat"
    )

    assert report.correlation > 0.999
    assert report.intended_orientation_is_best
    np.testing.assert_allclose(report.center_offset_voxels, 0, atol=1e-6)
