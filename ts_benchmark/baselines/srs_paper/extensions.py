"""Selectivity-controls extensions for SRSNet reproduction.

This module replaces the earlier "broad extension set" (DDA / MSP / GAF /
RandomSRS) with a focused selectivity-controls study, per the plan in
``scripts/repro/selectivity_extension_plan.md``.

Rationale (summary -- read the plan for the full discussion):

    * The earlier extensions were either capacity-confounded (GAF, MSP),
      effectively no-ops in the sigmoid-saturated regime (DDA), or imprecisely
      named and entangled with both selection and shuffling (RandomSRS).
    * The selectivity-controls study tests one sharp question instead:
        How much does SRSNet's performance depend on the **learned** patch
        selector, compared with random patch choices?
    * Two retrained controls are added.  Both share the SRSNet architecture
      except for the SRS ``_select`` (and, in the third variant, the SRS
      ``_shuffle``):

        * ``SRSRandomSP``                  -- random ``_select`` only,
                                              learned ``_shuffle`` kept.
        * ``SRSRandomSPNoShuffle``         -- random ``_select`` + identity
                                              shuffle (deterministic).  This
                                              is the **default** second
                                              variant per review note #3 of
                                              the plan.
        * ``SRSRandomSPRandomShuffle``     -- random ``_select`` + random
                                              ``_shuffle``.  Optional, kept
                                              for ablating both stages at
                                              once.

For naming hygiene (review note #6) the old ``SRSNet_RandomSRS`` wrapper is
**aliased** to ``SRSNet_RandomSP`` so old call-sites do not silently break,
but new code should use the precise name.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ts_benchmark.baselines.srsnet.layers.SRS import SRS
from ts_benchmark.baselines.srsnet.models.srsnet_model import SRSNetModel
from ts_benchmark.baselines.srsnet.srsnet import MODEL_HYPER_PARAMS, SRSNet


# ---------------------------------------------------------------------------
# Random selective-patching layers
# ---------------------------------------------------------------------------
class SRSRandomSP(SRS):
    """SRS variant that selects candidate windows uniformly at random.

    Everything else (``_shuffle``, adaptive fusion, embeddings, dropout,
    positional handling) is inherited unchanged from the parent ``SRS``.
    The only intervention is in ``_select``: instead of scoring each
    candidate window through the learned ``scorer_select`` MLP, we draw
    a uniform random index for each output slot.

    Two consequences worth noting:

    * The learned ``scorer_select`` parameters still exist (so the
      state-dict shape matches a vanilla SRSNet), but they receive zero
      gradient through the forward pass.  This is intentional -- the goal
      is a tight ``selectivity claim'' control, not a parameter-count
      ablation.
    * The original ``_select`` multiplies each selected patch by a unit-
      scaled ``max_scores`` factor.  Random selection has no scores, so we
      drop that multiplication.  This matches the sketch in the plan.
    """

    def _select(self, x_rec):
        # x_rec shape: [batch, n_vars, candidate_num, patch_size]
        batch, n_vars, candidate_num, patch_size = x_rec.shape
        idx = torch.randint(
            low=0,
            high=candidate_num,
            size=(batch, n_vars, 1, self.patch_num),
            device=x_rec.device,
        )
        # Broadcast indices over the patch_size dim so torch.gather can
        # collect the entire patch window for each chosen candidate.
        gather_idx = idx.repeat(1, 1, patch_size, 1).permute(0, 1, 3, 2)
        return torch.gather(x_rec, dim=-2, index=gather_idx)


class SRSRandomSPNoShuffle(SRSRandomSP):
    """Random ``_select`` + identity ``_shuffle`` (deterministic shuffle).

    Per review note #3 of the plan, identity shuffle is the recommended
    default for the second variant because it introduces no extra source
    of stochasticity and isolates "no learning in ``_shuffle``" cleanly.
    """

    def _shuffle(self, selected_patches):
        return selected_patches


class SRSRandomSPRandomShuffle(SRSRandomSP):
    """Random ``_select`` + random ``_shuffle`` (full stochastic baseline).

    Kept as an optional third variant for users who want to ablate both
    selection and shuffling at the same time.  Has higher per-seed
    variance than ``SRSRandomSPNoShuffle`` because both stages are now
    sampled fresh each forward pass.
    """

    def _shuffle(self, selected_patches):
        batch, n_vars, patch_num, patch_size = selected_patches.shape
        scores = torch.rand(
            batch, n_vars, patch_num, 1, device=selected_patches.device
        )
        order = torch.argsort(scores, dim=-2, descending=True)
        gather_idx = order.repeat(1, 1, 1, patch_size)
        return torch.gather(selected_patches, dim=-2, index=gather_idx)


# ---------------------------------------------------------------------------
# Constructive extensions: TASP, Hypernet-AF, PS-SRS
# ---------------------------------------------------------------------------
class SRSTimeAware(SRS):
    """TASP: scorer over engineered interpretable features.

    Addresses paper FW#1 (environment-aware mechanism) + L3 (interpretability).
    Replaces the [patch_len -> hidden -> patch_num] MLP scorer with a
    [4 -> 16 -> patch_num] MLP whose inputs are 4 per-window summary
    statistics: dominant FFT magnitude, lag-1 autocorrelation, variance,
    and trend slope.  Total scorer parameters drop sharply, so any
    improvement cannot be attributed to extra capacity.
    """

    N_FEATURES = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the learned scorer over raw patch values with a tiny
        # MLP over engineered per-window features.  The shuffle scorer
        # is left alone so any change in MSE is attributable to the
        # selection scorer alone.
        hidden = max(self.N_FEATURES * 4, 16)
        self.scorer_select = nn.Sequential(
            nn.Linear(self.N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.patch_num),
        )

    @staticmethod
    def _window_features(x_rec):
        """x_rec: [B, C, candidate_num, patch_size] -> [B, C, candidate_num, 4]."""
        # Dominant non-DC FFT magnitude per window.
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        if spec.shape[-1] > 1:
            dominant = spec[..., 1:].max(dim=-1).values
        else:
            dominant = spec[..., 0]
        # Demean for autocorrelation and trend-slope computation.
        mean = x_rec.mean(dim=-1, keepdim=True)
        var = x_rec.var(dim=-1, unbiased=False)
        x_c = x_rec - mean
        # Lag-1 autocorrelation, biased toward zero for very short windows.
        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom
        # Trend slope: least-squares fit residual.
        t = torch.arange(x_rec.shape[-1], device=x_rec.device, dtype=x_rec.dtype)
        t_c = t - t.mean()
        slope_denom = (t_c * t_c).sum() + 1e-8
        slope = ((x_rec - mean) * t_c).sum(dim=-1) / slope_denom
        feats = torch.stack([dominant, ac1, var, slope], dim=-1)
        return feats

    def _select(self, x_rec):
        # x_rec: [B, C, candidate_num, patch_size]
        feats = self._window_features(x_rec)
        scores = self.scorer_select(feats)  # [B, C, candidate_num, patch_num]
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)
        max_scores[non_zero_mask] *= inv
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches
        return selected_patches


class SRSHypernetAF(SRS):
    """Hypernet-AF: data-dependent alpha via a tiny hypernet.

    Addresses paper FW#3 (efficient alpha update mechanism) + L4
    (initialization).  Replaces the free [patch_num, d_model] alpha
    parameter with a small hypernet that consumes a batch-level context
    vector and outputs the per-cell alpha.  Zero-init of the hypernet
    weights plus a constant bias of 3.5 makes step 0 behave identically
    to the paper's default (sigmoid(3.5) ~ 0.97), so any deviation must
    come from learned context-dependence rather than capacity at init.
    """

    HYPER_HIDDEN = 8
    INIT_ALPHA = 3.5

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)
        # Remove the free alpha parameter; the hypernet replaces it.
        # We still keep the buffer name for state-dict shape compatibility
        # with vanilla SRSNet, but it will not be used in forward().
        if hasattr(self, "alpha"):
            self.register_buffer("_unused_alpha", torch.zeros_like(self.alpha.data))
            del self.alpha
        self.context_proj = nn.Linear(d_model, self.HYPER_HIDDEN)
        self.hyper = nn.Linear(self.HYPER_HIDDEN, self.patch_num * d_model)
        # Vanilla-preserving init: zero weights + bias = INIT_ALPHA on the
        # output so sigmoid(alpha_t=0) ~ 0.97, matching the paper baseline.
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.zeros_(self.hyper.weight)
        nn.init.constant_(self.hyper.bias, self.INIT_ALPHA)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        rec_repr_space = self._rec_view(x)
        original_repr_space = self._origin_view(x)
        e_orig = self.value_embedding_org(original_repr_space)
        e_rec = self.value_embedding_rec(rec_repr_space)
        # Context: mean across batch*nvars and across patches of e_orig.
        # Shape after the SRS forward of the parent: [bs*nvars, patch_num, d_model].
        ctx = e_orig.mean(dim=0).mean(dim=0)  # [d_model]
        h = torch.relu(self.context_proj(ctx))
        alpha_dyn = self.hyper(h).view(self.patch_num, e_orig.shape[-1])
        weight = torch.sigmoid(alpha_dyn)
        embedding = weight * e_orig + (1.0 - weight) * e_rec
        if self.pos:
            embedding = embedding + self.position_embedding(original_repr_space)
        return self.dropout(embedding), n_vars


class SRSPatternSupervised(SRS):
    """PS-SRS: scorer trained with an auxiliary pattern-descriptor loss.

    Addresses paper FW#4 (supervised module for sample-wise patterns) +
    L3 (interpretability).  Adds an auxiliary head over the scorer's
    first hidden layer that predicts a small set of engineered
    descriptors (FFT magnitude, lag-1 autocorrelation, variance).  The
    auxiliary loss is exposed via ``last_aux_loss`` so the SRSNet
    wrapper can return it in ``out_loss["additional_loss"]``.  When the
    layer is in ``eval()`` mode the auxiliary computation is skipped.
    """

    N_DESCRIPTORS = 3
    LAMBDA_AUX = 1e-2

    def __init__(self, d_model, patch_len, stride, seq_len, dropout,
                 hidden_size, alpha=2.0, pos=True):
        super().__init__(d_model, patch_len, stride, seq_len, dropout,
                         hidden_size, alpha, pos)
        # The scorer in the parent is nn.Sequential(Linear, ReLU, Linear).
        # We tap into the first hidden activation (after the ReLU) and
        # predict a per-window descriptor vector from it.
        hidden_dim = self.scorer_select[0].out_features
        self.descriptor_head = nn.Linear(hidden_dim, self.N_DESCRIPTORS)
        self.last_aux_loss = torch.tensor(0.0)

    @staticmethod
    def _ground_truth_descriptors(x_rec):
        """Compute the engineered descriptors used as auxiliary targets."""
        spec = torch.fft.rfft(x_rec, dim=-1).abs()
        if spec.shape[-1] > 1:
            dominant = spec[..., 1:].max(dim=-1).values
        else:
            dominant = spec[..., 0]
        mean = x_rec.mean(dim=-1, keepdim=True)
        var = x_rec.var(dim=-1, unbiased=False)
        x_c = x_rec - mean
        denom = x_c.pow(2).sum(dim=-1) + 1e-8
        ac1 = (x_c[..., 1:] * x_c[..., :-1]).sum(dim=-1) / denom
        return torch.stack([dominant, ac1, var], dim=-1)

    def _select(self, x_rec):
        # Compute scorer activations once, then reuse for both selection
        # and the auxiliary loss.
        first_layer = self.scorer_select[0]
        relu = self.scorer_select[1]
        second_layer = self.scorer_select[2]
        h = relu(first_layer(x_rec))
        scores = second_layer(h)

        if self.training:
            with torch.no_grad():
                gt = self._ground_truth_descriptors(x_rec)
            pred = self.descriptor_head(h)
            self.last_aux_loss = F.mse_loss(pred, gt) * self.LAMBDA_AUX
        else:
            self.last_aux_loss = torch.tensor(0.0, device=x_rec.device)

        # Mirror the parent's selection logic from this point on.
        indices = torch.argmax(scores, dim=-2, keepdim=True)
        max_scores = torch.gather(input=scores, dim=-2, index=indices)
        non_zero_mask = max_scores != 0
        inv = (1 / max_scores[non_zero_mask]).detach()
        x_rec_indices = indices.repeat(1, 1, self.patch_len, 1).permute(0, 1, 3, 2)
        selected_patches = torch.gather(input=x_rec, index=x_rec_indices, dim=-2)
        max_scores[non_zero_mask] *= inv
        selected_patches = max_scores.permute(0, 1, 3, 2) * selected_patches
        return selected_patches


# ---------------------------------------------------------------------------
# SRSNetModel subclasses (swap the patch_embedding for the random variant)
# ---------------------------------------------------------------------------
class _SelectivityControlsSRSNetModel(SRSNetModel):
    """SRSNetModel that swaps the patch_embedding for a control layer."""

    embedding_cls = None  # override in subclass

    def __init__(self, config):
        super().__init__(config)
        if self.embedding_cls is None:
            raise NotImplementedError(
                "Subclass must set embedding_cls to an SRS variant."
            )
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


class SRSNet_RandomSP_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSP


class SRSNet_RandomSPNoShuffle_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSPNoShuffle


class SRSNet_RandomSPRandomShuffle_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSRandomSPRandomShuffle


class SRSNet_TASP_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSTimeAware


class SRSNet_HypernetAF_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSHypernetAF


class SRSNet_PSRS_Model(_SelectivityControlsSRSNetModel):
    embedding_cls = SRSPatternSupervised


# ---------------------------------------------------------------------------
# DeepForecastingModelBase wrappers (registered in __init__.py)
# ---------------------------------------------------------------------------
class _SelectivityControlsSRSNet(SRSNet):
    """SRSNet variant that reports a distinct model_name + model class."""

    variant_name = None
    model_cls = None

    @property
    def model_name(self):
        return self.variant_name

    def _init_model(self):
        return self.model_cls(self.config)


class SRSNet_RandomSP(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSP"
    model_cls = SRSNet_RandomSP_Model


class SRSNet_RandomSPNoShuffle(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSPNoShuffle"
    model_cls = SRSNet_RandomSPNoShuffle_Model


class SRSNet_RandomSPRandomShuffle(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_RandomSPRandomShuffle"
    model_cls = SRSNet_RandomSPRandomShuffle_Model


class SRSNet_TASP(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_TASP"
    model_cls = SRSNet_TASP_Model


class SRSNet_HypernetAF(_SelectivityControlsSRSNet):
    variant_name = "SRSNet_HypernetAF"
    model_cls = SRSNet_HypernetAF_Model


class SRSNet_PSRS(_SelectivityControlsSRSNet):
    """SRSNet wrapper for PS-SRS.

    Overrides ``_process`` to expose the auxiliary descriptor-recovery
    loss as ``out_loss["additional_loss"]``.  The base
    ``DeepForecastingModelBase`` already adds that key to the training
    loss when present (see deep_forecasting_model_base.py:345-350).
    """

    variant_name = "SRSNet_PSRS"
    model_cls = SRSNet_PSRS_Model

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)
        aux = getattr(self.model.patch_embedding, "last_aux_loss", None)
        out_loss = {"output": output}
        if aux is not None and self.model.training:
            out_loss["additional_loss"] = aux
        return out_loss


# Backwards compatibility alias: the old broad-extensions code referenced
# ``srs_paper.SRSNet_RandomSRS``.  Keep that name as an alias so any stale
# tasks in older manifests still resolve, but emit no behaviour difference.
SRSNet_RandomSRS = SRSNet_RandomSP


for _cls in (
    SRSNet_RandomSP,
    SRSNet_RandomSPNoShuffle,
    SRSNet_RandomSPRandomShuffle,
    SRSNet_TASP,
    SRSNet_HypernetAF,
    SRSNet_PSRS,
):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS


__all__ = [
    # Layers -- random controls
    "SRSRandomSP",
    "SRSRandomSPNoShuffle",
    "SRSRandomSPRandomShuffle",
    # Layers -- constructive extensions
    "SRSTimeAware",
    "SRSHypernetAF",
    "SRSPatternSupervised",
    # Model wrappers
    "SRSNet_RandomSP",
    "SRSNet_RandomSPNoShuffle",
    "SRSNet_RandomSPRandomShuffle",
    "SRSNet_TASP",
    "SRSNet_HypernetAF",
    "SRSNet_PSRS",
    # Backwards-compat alias
    "SRSNet_RandomSRS",
    # Inner SRSNetModel subclasses (exposed for completeness)
    "SRSNet_RandomSP_Model",
    "SRSNet_RandomSPNoShuffle_Model",
    "SRSNet_RandomSPRandomShuffle_Model",
    "SRSNet_TASP_Model",
    "SRSNet_HypernetAF_Model",
    "SRSNet_PSRS_Model",
]
