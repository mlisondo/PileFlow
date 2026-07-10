# pumml/src/models/loss.py
#
# Modified per-pixel logarithmic squared loss from PUMML paper Eq. 2.1:
#
#   L = < [ log(p_pred + p_bar) - log(p_true + p_bar) ]^2 >
#
# where p_bar = 10 GeV is a hyperparameter that controls the trade-off
# between accuracy on hard pixels (large p_bar) and soft pixels (small p_bar).
#
# The log softens the penalty for mispredictions on high-pT pixels and
# prevents the network from ignoring soft pixels entirely (which MSE would do).
#
# Paper Section 2: "After mild optimization, a value of p_bar = 10 GeV was
# chosen, though the performance is relatively robust to this choice."

import torch
import torch.nn as nn

class PUMMLLoss(nn.Module):
    """
    Modified per-pixel logarithmic squared loss.

    Parameters
    ----------
    pbar : float
        Softening parameter in GeV (paper default: 10.0).
        Controls whether the loss favours hard pixels (large pbar)
        or soft pixels (small pbar -> 0).
    """

    def __init__(self, pbar: float = 10.0):
        super().__init__()
        self.pbar = pbar

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (N, 1, 9, 9) or (N, 9, 9)  model predictions
        y_true : (N, 1, 9, 9) or (N, 9, 9)  ground truth

        Returns
        -------
        scalar loss (mean over all pixels and batch elements)
        """
        # clamp to avoid log(0); pbar already shifts away from zero
        eps = 1e-8
        pred_log = torch.log(torch.clamp(y_pred, min=eps) + self.pbar)
        true_log = torch.log(torch.clamp(y_true, min=eps) + self.pbar)
        return torch.mean((pred_log - true_log) ** 2)

    def extra_repr(self) -> str:
        return f"pbar={self.pbar}"