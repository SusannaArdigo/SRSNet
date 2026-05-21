"""TFB wrappers for the paper's ablation study (Tab.4).

Four variants, each swapping a single SRS component:
  - SRSNet_NoSRS   : drop the whole SRS module (adjacent patching only)
  - SRSNet_NoSP    : skip _select (keep _shuffle)
  - SRSNet_NoDR    : keep _select, identity _shuffle
  - SRSNet_NoAF    : drop learned alpha, fuse 50/50

Each wrapper plugs the matching layer class from layers.py into a
SRSNetModel subclass + a DeepForecastingModelBase subclass so TFB can
pick it up by name (srs_paper.SRSNet_NoXX).
"""

from ts_benchmark.baselines.srs_paper.layers import (
    SRSNoAdaptiveFusion,
    SRSNoDynamicReassembly,
    SRSNoSRS,
    SRSNoSelectivePatching,
)
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet


class _AblationSRSNetModel(SRSNetModel):
    """SRSNetModel base that swaps the patch_embedding for an ablation layer."""

    layer_cls = None  # override in subclass

    def __init__(self, config):
        super().__init__(config)                                # builds revin, head, default patch_embedding
        # Overwrite the default SRS with the ablation variant (same constructor signature).
        self.patch_embedding = self.layer_cls(
            config.d_model,
            self.patch_len,
            self.stride,
            self.seq_len,
            config.dropout,
            config.hidden_size,
            config.alpha,
            config.pos,
        )


class SRSNet_NoSRS_Model(_AblationSRSNetModel):
    layer_cls = SRSNoSRS                                        # adjacent only, no SRS


class SRSNet_NoSP_Model(_AblationSRSNetModel):
    layer_cls = SRSNoSelectivePatching                          # _shuffle only


class SRSNet_NoDR_Model(_AblationSRSNetModel):
    layer_cls = SRSNoDynamicReassembly                          # _select only


class SRSNet_NoAF_Model(_AblationSRSNetModel):
    layer_cls = SRSNoAdaptiveFusion                             # fixed 50/50 fusion


class _AblationSRSNet(SRSNet):
    """SRSNet TFB wrapper that reports a distinct model_name + ablation model_cls."""

    variant_name = None
    model_cls = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def model_name(self):
        """String TFB uses to label this variant in result CSVs."""
        return self.variant_name

    def _init_model(self):
        """Instantiate the underlying ablation nn.Module."""
        return self.model_cls(self.config)


class SRSNet_NoSRS(_AblationSRSNet):
    variant_name = "SRSNet_NoSRS"
    model_cls = SRSNet_NoSRS_Model


class SRSNet_NoSP(_AblationSRSNet):
    variant_name = "SRSNet_NoSP"
    model_cls = SRSNet_NoSP_Model


class SRSNet_NoDR(_AblationSRSNet):
    variant_name = "SRSNet_NoDR"
    model_cls = SRSNet_NoDR_Model


class SRSNet_NoAF(_AblationSRSNet):
    variant_name = "SRSNet_NoAF"
    model_cls = SRSNet_NoAF_Model


# Inject the default hyper-params into each wrapper so TFB's CLI override
# logic (--model-hyper-params JSON) merges on top of the same defaults
# as the vanilla SRSNet.
for _cls in (SRSNet_NoSRS, SRSNet_NoSP, SRSNet_NoDR, SRSNet_NoAF):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS
