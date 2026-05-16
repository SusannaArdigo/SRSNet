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


# Backwards compatibility alias: the old broad-extensions code referenced
# ``srs_paper.SRSNet_RandomSRS``.  Keep that name as an alias so any stale
# tasks in older manifests still resolve, but emit no behaviour difference.
SRSNet_RandomSRS = SRSNet_RandomSP


for _cls in (
    SRSNet_RandomSP,
    SRSNet_RandomSPNoShuffle,
    SRSNet_RandomSPRandomShuffle,
):
    _cls.MODEL_HYPER_PARAMS = MODEL_HYPER_PARAMS


__all__ = [
    # Layers
    "SRSRandomSP",
    "SRSRandomSPNoShuffle",
    "SRSRandomSPRandomShuffle",
    # Model wrappers
    "SRSNet_RandomSP",
    "SRSNet_RandomSPNoShuffle",
    "SRSNet_RandomSPRandomShuffle",
    # Backwards-compat alias
    "SRSNet_RandomSRS",
    # Inner SRSNetModel subclasses (exposed for completeness)
    "SRSNet_RandomSP_Model",
    "SRSNet_RandomSPNoShuffle_Model",
    "SRSNet_RandomSPRandomShuffle_Model",
]
