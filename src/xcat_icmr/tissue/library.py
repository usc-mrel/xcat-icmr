"""Built-in tissue libraries translated from MATLAB ``config.m``."""

from __future__ import annotations

from xcat_icmr.tissue.models import (
    TissueGroup,
    TissueLibrary,
    TissueProperties,
    XcatLabel,
)


LEGACY_MATLAB_055T_NAME = "legacy-matlab-0p55t"


def _properties(
    t1_ms: float, t2_ms: float, proton_density_percent: float
) -> TissueProperties:
    return TissueProperties(
        t1_ms=t1_ms,
        t2_ms=t2_ms,
        proton_density_percent=proton_density_percent,
    )


# This is intentionally explicit. It is the Python reference translation of
# assign_tissue_mask_(0.55) in config.m, including its current choices such as
# grouping pancreas with kidney and leaving many known anatomy labels unassigned.
LEGACY_MATLAB_055T = TissueLibrary(
    name=LEGACY_MATLAB_055T_NAME,
    field_strength_t=0.55,
    groups=(
        TissueGroup(
            "Water",
            (
                XcatLabel.STOMACH_CONTENTS,
                XcatLabel.ESOPHAGUS_CONTENTS,
                XcatLabel.GALL_BLADDER,
            ),
            _properties(2200.0, 200.0, 100.0),
        ),
        TissueGroup(
            "Kidney",
            (
                XcatLabel.PANCREAS,
                XcatLabel.RIGHT_KIDNEY_CORTEX,
                XcatLabel.LEFT_KIDNEY_CORTEX,
                XcatLabel.RIGHT_KIDNEY_MEDULLA,
                XcatLabel.LEFT_KIDNEY_MEDULLA,
                XcatLabel.RIGHT_RENAL_PELVIS,
                XcatLabel.LEFT_RENAL_PELVIS,
            ),
            _properties(651.0, 101.0, 70.0),
        ),
        TissueGroup(
            "Muscle",
            (
                XcatLabel.LEFT_ATRIUM_MYOCARDIUM,
                XcatLabel.RIGHT_ATRIUM_MYOCARDIUM,
                XcatLabel.LEFT_VENTRICLE_MYOCARDIUM,
                XcatLabel.RIGHT_VENTRICLE_MYOCARDIUM,
                XcatLabel.MUSCLE,
                XcatLabel.STOMACH_WALL,
                XcatLabel.SMALL_INTESTINE,
                XcatLabel.ASCENDING_LARGE_INTESTINE,
                XcatLabel.TRANSVERSE_LARGE_INTESTINE,
                XcatLabel.DESCENDING_LARGE_INTESTINE,
            ),
            _properties(701.0, 58.0, 80.0),
        ),
        TissueGroup(
            "Fat",
            (
                XcatLabel.PERICARDIUM,
                XcatLabel.BODY,
                XcatLabel.ADRENAL,
            ),
            _properties(187.0, 93.0, 70.0),
        ),
        TissueGroup(
            "Blood",
            (
                XcatLabel.LEFT_VENTRICLE_CHAMBER,
                XcatLabel.RIGHT_VENTRICLE_CHAMBER,
                XcatLabel.ARTERY,
                XcatLabel.VEIN,
                XcatLabel.LEFT_ATRIUM_CHAMBER,
                XcatLabel.RIGHT_ATRIUM_CHAMBER,
            ),
            _properties(1122.0, 263.0, 95.0),
        ),
        TissueGroup(
            "Liver",
            (XcatLabel.LIVER,),
            _properties(339.0, 66.0, 90.0),
        ),
        TissueGroup(
            "Bone",
            (XcatLabel.RIB, XcatLabel.CORTICAL_BONE, XcatLabel.SPINE),
            _properties(250.0, 20.0, 12.0),
        ),
        TissueGroup(
            "Spleen",
            (XcatLabel.SPLEEN,),
            # MATLAB: 2 * (B0 * 42.577e6)^0.33 with B0 = 0.55 T.
            _properties(540.7262486498952, 62.0, 70.0),
        ),
        TissueGroup(
            "BoneMarrow",
            (XcatLabel.BONE_MARROW,),
            _properties(187.0, 93.0, 35.0),
        ),
        TissueGroup(
            "Air",
            (
                XcatLabel.OUTSIDE,
                XcatLabel.SINUS,
                XcatLabel.RIGHT_LUNG,
                XcatLabel.LEFT_LUNG,
                XcatLabel.INTESTINE_CONTENTS,
                XcatLabel.TRACHEA_BRONCHI,
                XcatLabel.AIRWAY_TREE,
            ),
            _properties(0.0, 0.0, 0.0),
        ),
    ),
)


_BUILTIN_LIBRARIES = {LEGACY_MATLAB_055T.name: LEGACY_MATLAB_055T}


def get_tissue_library(name: str) -> TissueLibrary:
    """Resolve a built-in tissue library by its YAML-facing name."""

    try:
        return _BUILTIN_LIBRARIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BUILTIN_LIBRARIES))
        raise KeyError(
            f"unknown tissue library {name!r}; available: {available}"
        ) from exc
