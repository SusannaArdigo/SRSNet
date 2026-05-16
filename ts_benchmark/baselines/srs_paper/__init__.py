from ts_benchmark.baselines.srs_paper.ablations import (
    SRSNet_NoAF,
    SRSNet_NoDR,
    SRSNet_NoSP,
    SRSNet_NoSRS,
)
from ts_benchmark.baselines.srs_paper.extensions import (
    SRSNet_HypernetAF,
    SRSNet_PSRS,
    SRSNet_PSRS_HypernetAF,
    SRSNet_RandomSP,
    SRSNet_RandomSPNoShuffle,
    SRSNet_RandomSPRandomShuffle,
    SRSNet_RandomSP_HypernetAF,
    SRSNet_RandomSRS,  # backwards-compat alias of SRSNet_RandomSP
    SRSNet_TASP,
    SRSNet_TASP_HypernetAF,
)
from ts_benchmark.baselines.srs_paper.plugins import (
    SRSPlusCrossformer,
    SRSPlusPatchMLP,
    SRSPlusPatchTST,
    SRSPlusxPatch,
)

__all__ = [
    "SRSNet_NoAF",
    "SRSNet_NoDR",
    "SRSNet_NoSP",
    "SRSNet_NoSRS",
    # Random selectivity controls
    "SRSNet_RandomSP",
    "SRSNet_RandomSPNoShuffle",
    "SRSNet_RandomSPRandomShuffle",
    # Constructive extensions
    "SRSNet_TASP",
    "SRSNet_HypernetAF",
    "SRSNet_PSRS",
    # Factorial combinations
    "SRSNet_TASP_HypernetAF",
    "SRSNet_RandomSP_HypernetAF",
    "SRSNet_PSRS_HypernetAF",
    "SRSNet_RandomSRS",
    "SRSPlusCrossformer",
    "SRSPlusPatchMLP",
    "SRSPlusPatchTST",
    "SRSPlusxPatch",
]
