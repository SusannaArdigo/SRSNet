"""SRSNet extensions proposed by reproduction study.

Implements 4 new modules that address Future Work and Limitations
documented in the SRSNet paper (Sec.6 of arxiv 2510.14510):

A. SRSWithGatedAF       -- Gated Adaptive Fusion (replaces free alpha
                           parameter with a mini-MLP that produces a
                           content-dependent fusion weight).
                           Addresses L4 (Initialization) + FW#3
                           (more efficient update mechanism for alpha).

F. SRSRandom            -- Random patch sampling instead of the
                           learned Selective Patching scorer.  Acts as
                           a strict baseline for the paper claim that
                           "selectivity matters".

B. SRSNet_DDA           -- SRSNet wrapper that consumes a data-driven
                           alpha value precomputed by FFT seasonality
                           statistics (see tools/compute_dda_alpha.py).
                           Addresses L4 (alpha initialization).

D. SRSMultiScale        -- Three parallel SRS modules with
                           patch_len in {8, 24, 48}, aligned via
                           adaptive average pooling on patch_num and
                           fused with a linear projection.
                           Addresses FW#1 (environment-aware /
                           multi-resolution selective patching).

The corresponding SRSNet wrappers (SRSNet_GAF, SRSNet_RandomSRS,
SRSNet_DDA, SRSNet_MSP) plug into the standard
ts_benchmark.baselines.deep_forecasting_model_base.DeepForecastingModelBase
pipeline so that paper_repro.py only has to override --model-name.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ts_benchmark.baselines.srsnet.layers.SRS import SRS, PositionalEmbedding
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet


# ---------------------------------------------------------------------------
# A. Gated Adaptive Fusion (GAF)
# ---------------------------------------------------------------------------
class SRSWithGatedAF(SRS):
    """SRS with data-dependent Adaptive Fusion.

    Replaces ``alpha`` (a free [patch_num, d_model] parameter shared
    across the batch) with a small MLP that consumes the concatenated
    original/reconstructed embeddings and produces a per-sample fusion
    weight.  This addresses the paper's own observation that "the
    initialization of the weights alpha seems to be important"
    (Section 6, Potential limitations, bullet 4).
    """

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)
        # Gating MLP: [E_orig | E_rec] -> per-patch-per-channel weight in R^d_model.
        gate_hidden = max(d_model // 4, 16)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, d_model),
        )
        # Keep self.alpha around for state_dict compatibility, but it is
        # no longer used in forward().

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)

        rec_repr_space = self._rec_view(x)
        original_repr_space = self._origin_view(x)

        e_orig = self.value_embedding_org(original_repr_space)
        e_rec = self.value_embedding_rec(rec_repr_space)

        # Data-dependent gating, applied per (batch*nvars, patch, d_model).
        gate_in = torch.cat([e_orig, e_rec], dim=-1)
        weight = torch.sigmoid(self.gate(gate_in))

        embedding = weight * e_orig + (1.0 - weight) * e_rec
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


# ---------------------------------------------------------------------------
# F. Random selective patching baseline
# ---------------------------------------------------------------------------
class SRSRandom(SRS):
    """SRS with random patch sampling instead of scored selection.

    Keeps the rest of the architecture untouched (Dynamic Reassembly,
    Adaptive Fusion, embeddings).  Acts as a critical baseline: if
    Random-SRS matches Selective-SRS, the "selectivity is important"
    claim of the paper is empirically refuted.
    """

    def _select(self, x_rec):
        # x_rec shape: [batch, n_vars, seq_len - patch_size + 1, patch_size]
        batch, n_vars, candidate_num, patch_size = x_rec.shape
        # Sample random candidate indices for each selected slot.
        random_indices = torch.randint(
            0,
            candidate_num,
            (batch, n_vars, 1, self.patch_num),
            device=x_rec.device,
        )
        # Expand to [batch, n_vars, patch_size, patch_num] then permute to
        # gather along the candidate dim (-2) as expected by parent code.
        random_idx_expanded = (
            random_indices.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)
        )
        # [batch, n_vars, patch_num, patch_size]
        random_patches = torch.gather(input=x_rec, dim=-2, index=random_idx_expanded)
        return random_patches


# ---------------------------------------------------------------------------
# D. Multi-scale Selective Patching (MSP)
# ---------------------------------------------------------------------------
class SRSMultiScale(nn.Module):
    """Three parallel SRS modules with different patch lengths.

    Each scale produces an embedding [bs*nvars, patch_num_i, d_model].
    Patch dimensions are aligned to a reference scale via adaptive
    average pooling, then concatenated along the feature dim and fused
    with a linear layer.  Output shape matches a single SRS at the
    reference scale, so the downstream FlattenHead is unchanged.
    """

    SCALES = (8, 24, 48)
    REF_SCALE_IDX = 1  # patch_len = 24, matches the paper default.

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__()
        # ``patch_len`` from the config is the reference patch length
        # (24).  We ignore it as an explicit value and use SCALES.
        del patch_len, stride
        ref_len = self.SCALES[self.REF_SCALE_IDX]
        self.ref_patch_num = (
            math.ceil((seq_len - ref_len) / ref_len) + 1
        )
        self.scales_list = list(self.SCALES)

        # One SRS per scale.  Each manages its own padding/scorer/alpha.
        self.srs_modules = nn.ModuleList(
            [
                SRS(
                    d_model=d_model,
                    patch_len=scale,
                    stride=scale,
                    seq_len=seq_len,
                    dropout=dropout,
                    hidden_size=hidden_size,
                    alpha=alpha,
                    pos=pos,
                )
                for scale in self.SCALES
            ]
        )

        # Fusion across scales: [bs*nvars, ref_patch_num, len(SCALES)*d_model]
        # -> [bs*nvars, ref_patch_num, d_model].
        self.fuse = nn.Linear(len(self.SCALES) * d_model, d_model)
        self.fuse_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = None
        aligned = []
        for srs in self.srs_modules:
            emb, n_vars = srs(x)  # [bs*nvars, patch_num_i, d_model]
            # Align patch_num to reference via adaptive pool over patches.
            emb_t = emb.transpose(-1, -2)
            emb_aligned_t = F.adaptive_avg_pool1d(emb_t, self.ref_patch_num)
            aligned.append(emb_aligned_t.transpose(-1, -2))
        fused = torch.cat(aligned, dim=-1)
        fused = self.fuse(fused)
        return self.fuse_dropout(fused), n_vars


# ---------------------------------------------------------------------------
# Model wrappers (SRSNetModel subclasses)
# ---------------------------------------------------------------------------
class _CustomEmbeddingSRSNetModel(SRSNetModel):
    """SRSNetModel that swaps the patch_embedding for a custom module."""

    embedding_cls = None  # override in subclass

    def __init__(self, config):
        super().__init__(config)
        if self.embedding_cls is None:
            raise NotImplementedError("Subclass must set embedding_cls.")
        self.patch_embedding = self.embedding_cls(
            config.d_model,
            self.patch_len,
            self.stride,
            self.seq_len,
            config.dropout,
            config.hidden_size,
            config.alpha,
            config.pos,
        )


class SRSNet_GAF_Model(_CustomEmbeddingSRSNetModel):
    embedding_cls = SRSWithGatedAF


class SRSNet_RandomSRS_Model(_CustomEmbeddingSRSNetModel):
    embedding_cls = SRSRandom


class SRSNet_MSP_Model(_CustomEmbeddingSRSNetModel):
    embedding_cls = SRSMultiScale


class SRSNet_DDA_Model(SRSNetModel):
    """DDA uses the standard SRSNetModel; alpha is overridden by CLI."""

    pass


# ---------------------------------------------------------------------------
# DeepForecastingModelBase wrappers (registered in __init__.py)
# ---------------------------------------------------------------------------
class _ExtensionSRSNet(SRSNet):
    """SRSNet variant that reports a distinct model_name and model class."""

    variant_name = None
    model_cls = None

    @property
    def model_name(self):
        return self.variant_name

    def _init_model(self):
        return self.model_cls(self.config)


class SRSNet_GAF(_ExtensionSRSNet):
    variant_name = "SRSNet_GAF"
    model_cls = SRSNet_GAF_Model


class SRSNet_RandomSRS(_ExtensionSRSNet):
    variant_name = "SRSNet_RandomSRS"
    model_cls = SRSNet_RandomSRS_Model


class SRSNet_DDA(_ExtensionSRSNet):
    variant_name = "SRSNet_DDA"
    model_cls = SRSNet_DDA_Model


class SRSNet_MSP(_ExtensionSRSNet):
    variant_name = "SRSNet_MSP"
    model_cls = SRSNet_MSP_Model


for _cls in (SRSNet_GAF, SRSNet_RandomSRS, SRSNet_DDA, SRSNet_MSP):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS


__all__ = [
    "SRSWithGatedAF",
    "SRSRandom",
    "SRSMultiScale",
    "SRSNet_GAF",
    "SRSNet_RandomSRS",
    "SRSNet_DDA",
    "SRSNet_MSP",
    "SRSNet_GAF_Model",
    "SRSNet_RandomSRS_Model",
    "SRSNet_DDA_Model",
    "SRSNet_MSP_Model",
]
