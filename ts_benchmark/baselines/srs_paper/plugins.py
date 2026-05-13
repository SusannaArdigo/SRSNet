import torch
from torch import nn

from ts_benchmark.baselines.patchmlp.patchmlp import MODEL_HYPER_PARAMS as PATCHMLP_PARAMS
from ts_benchmark.baselines.patchmlp.patchmlp import PatchMLP
from ts_benchmark.baselines.patchmlp.models.patchmlp_model import PatchMLPModel
from ts_benchmark.baselines.srs_paper.layers import SRSAsPatchEmbedding, patch_count
from ts_benchmark.baselines.time_series_library.adapters_for_transformers import (
    TransformerAdapter,
)
from ts_benchmark.baselines.time_series_library.models.Crossformer import Crossformer
from ts_benchmark.baselines.time_series_library.models.PatchTST import PatchTST
from ts_benchmark.baselines.xpatch.layers.network import Network
from ts_benchmark.baselines.xpatch.xpatch import MODEL_HYPER_PARAMS as XPATCH_PARAMS
from ts_benchmark.baselines.xpatch.xpatch import xPatch
from ts_benchmark.baselines.xpatch.models.xPatch import xPatchModel


def _srs_kwargs(config, default_hidden=128, default_alpha=2.0, default_dropout=None):
    return {
        "hidden_size": getattr(config, "srs_hidden_size", default_hidden),
        "alpha": getattr(config, "srs_alpha", default_alpha),
        "dropout": getattr(config, "srs_dropout", default_dropout if default_dropout is not None else config.dropout),
        "pos": getattr(config, "srs_pos", True),
    }


class SRSPlusPatchTST(TransformerAdapter):
    def __init__(self, **kwargs):
        super().__init__("SRS+PatchTST", PatchTST, **kwargs)

    def _init_model(self):
        model = PatchTST(self.config)
        kwargs = _srs_kwargs(self.config, default_dropout=self.config.dropout)
        model.patch_embedding = SRSAsPatchEmbedding(
            self.config.d_model,
            self.config.patch_len,
            self.config.stride,
            self.config.stride,
            kwargs["dropout"],
            kwargs["hidden_size"],
            kwargs["alpha"],
            kwargs["pos"],
            seq_len=self.config.seq_len,
        )
        return model


class SRSPlusCrossformer(TransformerAdapter):
    def __init__(self, **kwargs):
        super().__init__("SRS+Crossformer", Crossformer, **kwargs)

    def _init_model(self):
        model = Crossformer(self.config)
        kwargs = _srs_kwargs(self.config, default_dropout=0)
        model.enc_value_embedding = SRSAsPatchEmbedding(
            self.config.d_model,
            self.config.seg_len,
            self.config.seg_len,
            model.pad_in_len - self.config.seq_len,
            kwargs["dropout"],
            kwargs["hidden_size"],
            kwargs["alpha"],
            False,
            seq_len=self.config.seq_len,
        )
        return model


class _SRSEmbLayer(nn.Module):
    def __init__(self, patch_len, patch_step, seq_len, d_model, config):
        super().__init__()
        patch_num = patch_count(seq_len, patch_len, patch_step)
        inner = max(1, d_model // patch_num)
        kwargs = _srs_kwargs(config, default_hidden=256, default_dropout=0)
        self.srs = SRSAsPatchEmbedding(
            inner,
            patch_len,
            patch_step,
            0,
            kwargs["dropout"],
            kwargs["hidden_size"],
            kwargs["alpha"],
            kwargs["pos"],
            seq_len=seq_len,
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.proj = nn.Linear(inner * patch_num, d_model)

    def forward(self, x):
        patches, n_vars = self.srs(x)
        patches = patches.reshape(x.shape[0], n_vars, patches.shape[-2], patches.shape[-1])
        return self.proj(self.flatten(patches))


class _SRSMultiScaleEmb(nn.Module):
    def __init__(self, seq_len, d_model, config, patch_len=(48, 24, 12, 6)):
        super().__init__()
        patch_step = patch_len
        quarter = d_model // 4
        self.layers = nn.ModuleList(
            [
                _SRSEmbLayer(patch_len[0], patch_step[0] // 2, seq_len, quarter, config),
                _SRSEmbLayer(patch_len[1], patch_step[1] // 2, seq_len, quarter, config),
                _SRSEmbLayer(patch_len[2], patch_step[2] // 2, seq_len, quarter, config),
                _SRSEmbLayer(patch_len[3], patch_step[3] // 2, seq_len, quarter, config),
            ]
        )

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], dim=-1)


class _SRSPatchMLPModel(PatchMLPModel):
    def __init__(self, config):
        super().__init__(config)
        self.emb = _SRSMultiScaleEmb(config.seq_len, config.d_model, config)


class SRSPlusPatchMLP(PatchMLP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def model_name(self):
        return "SRS+PatchMLP"

    def _init_model(self):
        return _SRSPatchMLPModel(self.config)


class _SRSxPatchNetwork(Network):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, config):
        super().__init__(seq_len, pred_len, patch_len, stride, padding_patch)
        padding = stride if padding_patch == "end" else 0
        kwargs = _srs_kwargs(config, default_hidden=64, default_dropout=0)
        self.srs_emb = SRSAsPatchEmbedding(
            self.dim,
            patch_len,
            stride,
            padding,
            kwargs["dropout"],
            kwargs["hidden_size"],
            kwargs["alpha"],
            kwargs["pos"],
            seq_len=seq_len,
        )

    def forward(self, s, t):
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)
        bsz, channels, input_len = s.shape
        s_patches, _ = self.srs_emb(s)
        t = torch.reshape(t, (bsz * channels, input_len))

        s = self.gelu1(s_patches)
        s = self.bn1(s)
        res = s
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)
        res = self.fc2(res)
        s = s + res
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)

        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)
        x = torch.reshape(x, (bsz, channels, self.pred_len))
        return x.permute(0, 2, 1)


class _SRSxPatchModel(xPatchModel):
    def __init__(self, config):
        super().__init__(config)
        self.net = _SRSxPatchNetwork(
            config.seq_len,
            config.pred_len,
            config.patch_len,
            config.stride,
            config.padding_patch,
            config,
        )


class SRSPlusxPatch(xPatch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def model_name(self):
        return "SRS+xPatch"

    def _init_model(self):
        return _SRSxPatchModel(self.config)


for _cls, _params in (
    (SRSPlusPatchMLP, PATCHMLP_PARAMS),
    (SRSPlusxPatch, XPATCH_PARAMS),
):
    _cls.MODEL_HYPER_PARAMS = _params
