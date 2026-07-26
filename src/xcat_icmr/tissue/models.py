"""Typed anatomical labels and MR tissue-property models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class XcatLabel(IntEnum):
    """Activity labels emitted by the XCAT configuration used by MATLAB."""

    OUTSIDE = 0
    LEFT_VENTRICLE_MYOCARDIUM = 1
    RIGHT_VENTRICLE_MYOCARDIUM = 2
    LEFT_ATRIUM_MYOCARDIUM = 3
    RIGHT_ATRIUM_MYOCARDIUM = 4
    LEFT_VENTRICLE_CHAMBER = 5
    RIGHT_VENTRICLE_CHAMBER = 6
    LEFT_ATRIUM_CHAMBER = 7
    RIGHT_ATRIUM_CHAMBER = 8
    BODY = 9
    MUSCLE = 10
    BRAIN = 11
    SINUS = 12
    LIVER = 13
    GALL_BLADDER = 14
    RIGHT_LUNG = 15
    LEFT_LUNG = 16
    ESOPHAGUS = 17
    ESOPHAGUS_CONTENTS = 18
    LARYNGOPHARYNX = 19
    STOMACH_WALL = 20
    STOMACH_CONTENTS = 21
    PANCREAS = 22
    RIGHT_KIDNEY_CORTEX = 23
    RIGHT_KIDNEY_MEDULLA = 24
    LEFT_KIDNEY_CORTEX = 25
    LEFT_KIDNEY_MEDULLA = 26
    ADRENAL = 27
    RIGHT_RENAL_PELVIS = 28
    LEFT_RENAL_PELVIS = 29
    SPLEEN = 30
    RIB = 31
    CORTICAL_BONE = 32
    SPINE = 33
    SPINAL_CORD = 34
    BONE_MARROW = 35
    ARTERY = 36
    VEIN = 37
    BLADDER = 38
    PROSTATE = 39
    ASCENDING_LARGE_INTESTINE = 40
    TRANSVERSE_LARGE_INTESTINE = 41
    DESCENDING_LARGE_INTESTINE = 42
    SMALL_INTESTINE = 43
    RECTUM = 44
    SEMINAL_VESICLE = 45
    VAS_DEFERENS = 46
    TESTICULAR = 47
    EPIDIDYMIS = 48
    EJACULATORY_DUCT = 49
    PERICARDIUM = 50
    CARTILAGE = 51
    INTESTINE_CONTENTS = 52
    URETER = 53
    URETHRA = 54
    LYMPH_NORMAL = 55
    LYMPH_ABNORMAL = 56
    TRACHEA_BRONCHI = 57
    AIRWAY_TREE = 58
    UTERUS = 59
    VAGINA = 60
    RIGHT_OVARY = 61
    LEFT_OVARY = 62
    FALLOPIAN_TUBES = 63
    PARATHYROID = 64
    THYROID = 65
    THYMUS = 66
    SALIVARY = 67
    PITUITARY = 68
    EYE = 69
    EYE_LENS = 70
    HEART_LESION = 71


@dataclass(frozen=True)
class TissueProperties:
    """MR properties of a tissue group in the MATLAB units."""

    t1_ms: float
    t2_ms: float
    proton_density_percent: float

    def __post_init__(self) -> None:
        for name, value in (
            ("t1_ms", self.t1_ms),
            ("t2_ms", self.t2_ms),
            ("proton_density_percent", self.proton_density_percent),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TissueGroup:
    """One set of XCAT anatomical labels sharing the same MR properties."""

    name: str
    labels: tuple[XcatLabel, ...]
    properties: TissueProperties

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tissue group name cannot be empty")
        if not self.labels:
            raise ValueError(f"tissue group {self.name!r} has no labels")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"tissue group {self.name!r} contains duplicate labels")


@dataclass(frozen=True)
class TissueLibrary:
    """Named collection of non-overlapping tissue groups."""

    name: str
    field_strength_t: float
    groups: tuple[TissueGroup, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tissue library name cannot be empty")
        if self.field_strength_t <= 0:
            raise ValueError("field_strength_t must be positive")

        owners: dict[XcatLabel, str] = {}
        for group in self.groups:
            for label in group.labels:
                if label in owners:
                    raise ValueError(
                        f"XCAT label {int(label)} belongs to both "
                        f"{owners[label]!r} and {group.name!r}"
                    )
                owners[label] = group.name

    @property
    def mapped_labels(self) -> frozenset[XcatLabel]:
        """Labels to which the MATLAB configuration assigns a tissue group."""

        return frozenset(label for group in self.groups for label in group.labels)

    def group_for_label(self, label: int | XcatLabel) -> TissueGroup | None:
        """Return the tissue group for a known label, or None if unassigned."""

        xcat_label = XcatLabel(label)
        return next(
            (group for group in self.groups if xcat_label in group.labels),
            None,
        )
