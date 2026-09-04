"""XCAT anatomy labels, tissue libraries, and quantitative property mapping."""

from xcat_icmr.tissue.library import (
    LEGACY_MATLAB_055T,
    LEGACY_MATLAB_055T_NAME,
    get_tissue_library,
)
from xcat_icmr.tissue.mapping import (
    LabelMappingError,
    TissueParameterVolumes,
    map_labels_to_tissue_properties,
)
from xcat_icmr.tissue.models import (
    TissueGroup,
    TissueLibrary,
    TissueProperties,
    XcatLabel,
)

__all__ = [
    "LEGACY_MATLAB_055T",
    "LEGACY_MATLAB_055T_NAME",
    "LabelMappingError",
    "TissueGroup",
    "TissueLibrary",
    "TissueParameterVolumes",
    "TissueProperties",
    "XcatLabel",
    "get_tissue_library",
    "map_labels_to_tissue_properties",
]
