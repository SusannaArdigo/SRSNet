from ts_benchmark.baselines.srs_paper.layers import (
    SRSNoAdaptiveFusion,
    SRSNoDynamicReassembly,
    SRSNoSRS,
    SRSNoSelectivePatching,
)
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet


class _AblationSRSNetModel(SRSNetModel):

    layer_cls = None 

    def __init__(self, config):
        super().__init__(config)                                
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

    variant_name = None
    model_cls = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def model_name(self):
        return self.variant_name

    def _init_model(self):
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


for _cls in (SRSNet_NoSRS, SRSNet_NoSP, SRSNet_NoDR, SRSNet_NoAF):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS
