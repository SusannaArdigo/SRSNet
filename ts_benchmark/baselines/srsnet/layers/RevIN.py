import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., ICLR 2022).

    Normalizes each (sample, channel) along the time axis BEFORE the model,
    then puts back mean + std (and optional learned affine) AFTER the model.
    Mitigates train/test distribution shift in time series forecasting.

    -----------------------------------------------------------------------
    The idea (in two lines):
        Strip away non-stationarity BEFORE the model. Put it back AFTER.

    Step by step:
        1. For each sample in the batch and each channel, compute mean and
           stdev along the time axis.
        2. Subtract the mean and divide by the stdev -> every sample now
           has mean 0 and std 1.
        3. The model sees data on a uniform scale -> it focuses on the
           SHAPE of the signal, not on the absolute magnitude.
        4. When the model outputs a prediction (still in normalized scale),
           multiply by the SAME stdev and add the SAME mean -> the
           prediction goes back to the original scale.

    The "Reversible" in the name comes from this: the transformation is
    invertible, and we save the statistics between `norm` and `denorm`.

    -----------------------------------------------------------------------
    Numeric walkthrough (B=2, T=5, C=1, temperature in degrees C):

        Input:
            Sample 1: [22, 24, 26, 28, 30]
            Sample 2: [25, 27, 28, 29, 31]

        Step 1 (_get_statistics):
            Sample 1: mean=26.0, stdev=2.83
            Sample 2: mean=28.0, stdev=2.10

        Step 2 (_normalize):
            Sample 1: [-1.41, -0.71, 0.0, 0.71, 1.41]
            Sample 2: [-1.43, -0.48, 0.0, 0.48, 1.43]
            Both samples now look nearly identical: a centered linear ramp.

        Step 3 (the model predicts 2 future steps, still in normalized scale):
            Sample 1 prediction: [2.12, 2.83]
            Sample 2 prediction: [1.91, 2.39]

        Step 4 (_denormalize, using the SAVED stats):
            Sample 1: [2.12, 2.83] * 2.83 + 26.0 ~ [32.0, 34.0]
            Sample 2: [1.91, 2.39] * 2.10 + 28.0 ~ [32.0, 33.0]
            The output is back in degrees C, ready to be read by a human.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features:  number of channels C
        :param eps:           small constant for numerical stability
        :param affine:        if True, learn a per-channel scale + bias
        :param subtract_last: if True, center on the LAST timestep instead of the mean
                              (sometimes used on non-stationary series)
        """
        super(RevIN, self).__init__()
        self.num_features = num_features                                                            # C: number of channels
        self.eps = eps                                                                              # used in sqrt(var + eps)
        self.affine = affine                                                                        # learnable scale + bias?
        self.subtract_last = subtract_last                                                          # center on last step instead of mean?
        if self.affine:
            self._init_params()                                                                     # create the learnable params

    def forward(self, x, mode: str):
        # mode='norm'    -> compute statistics + normalize  (call BEFORE the model)
        # mode='denorm'  -> undo the normalization          (call AFTER  the model)
        if mode == 'norm':
            self._get_statistics(x)                                                                 # save mean/std on self for later
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)                                                                # reuse the saved mean/std
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # One scale + one bias per channel
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))                            # (C,)  init at 1
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))                             # (C,)  init at 0

    def _get_statistics(self, x):
        # x: [B, T, C]  ->  reduce over the TIME axes (everything between batch and channel)
        dim2reduce = tuple(range(1, x.ndim - 1))                                                    # = (1,) for 3D input
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)                                                    # [B, 1, C]: last timestep
        else:
            # mean over time, per (sample, channel). .detach(): treat as constants in the backward pass
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()                        # [B, 1, C]
        # std over time, per (sample, channel). Same .detach() reasoning.
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()  # [B, 1, C]

    def _normalize(self, x):
        # Center
        if self.subtract_last:
            x = x - self.last                                                                       # center on last value
        else:
            x = x - self.mean                                                                       # center on the mean
        # Scale to unit variance
        x = x / self.stdev
        # Optional learned per-channel affine (broadcasts along B and T) (useful when some channels are more informative then other so the model will give them more weight)
        if self.affine:
            x = x * self.affine_weight                                                              # learned per-channel scale
            x = x + self.affine_bias                                                                # learned per-channel bias
        return x

    def _denormalize(self, x):
        # Inverse of _normalize, in reverse order
        if self.affine:
            x = x - self.affine_bias                                                                # undo bias
            x = x / (self.affine_weight + self.eps * self.eps)                                      # undo scale (eps^2 guards against /0)
        x = x * self.stdev                                                                          # put back the std
        if self.subtract_last:
            x = x + self.last                                                                       # put back the last value
        else:
            x = x + self.mean                                                                       # put back the mean
        return x
