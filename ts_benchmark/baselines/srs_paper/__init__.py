from ts_benchmark.baselines.srs_paper.ablations import (
    SRSNet_NoAF,
    SRSNet_NoDR,
    SRSNet_NoSP,
    SRSNet_NoSRS,
)
from ts_benchmark.baselines.srs_paper.extensions import (
    SRSNet_HypernetAF,
    SRSNet_HypernetAF_TransformerEncoder,
    SRSNet_RandomSP,
    SRSNet_RandomSPNoShuffle,
    SRSNet_RandomSPRandomShuffle,
    SRSNet_RandomSP_HypernetAF,
    SRSNet_RandomSP_NoShuffle_HypernetAF,
    SRSNet_RandomSRS,  # backwards-compat alias of SRSNet_RandomSP
    SRSNet_TASP,
    SRSNet_TASP_HypernetAF,
    SRSNet_TASP_NoShuffle_HypernetAF,
    SRSNet_TASP_TransformerEncoder,
    SRSNet_TransformerEncoder,
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
    # Constructive extensions (1-axis)
    "SRSNet_TASP",
    "SRSNet_HypernetAF",
    # Factorial combinations (2-axis: Select x Fusion)
    "SRSNet_TASP_HypernetAF",
    "SRSNet_RandomSP_HypernetAF",
    # 3-axis combos (Select x Identity-Shuffle x Hypernet-Fusion)
    "SRSNet_TASP_NoShuffle_HypernetAF",
    "SRSNet_RandomSP_NoShuffle_HypernetAF",
    # Backbone extension (Transformer Encoder)
    "SRSNet_TransformerEncoder",
    "SRSNet_TASP_TransformerEncoder",
    "SRSNet_HypernetAF_TransformerEncoder",
    "SRSNet_RandomSRS",
    "SRSPlusCrossformer",
    "SRSPlusPatchMLP",
    "SRSPlusPatchTST",
    "SRSPlusxPatch",
]
