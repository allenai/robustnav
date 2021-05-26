"""Defining the rotation-prediction loss for actor critic type models."""

import typing
from typing import Dict, Union
from typing import Optional, Callable

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.distributions import CategoricalDistr


class RotationPred(AbstractActorCriticLoss):
    """Implementation of the rotation-prediction loss

    # Attributes

    loss_coeff : Weight of the action-prediction loss  
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        pred_rotation_logits: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        # Cross-entropy criteria for rotation prediction
        ce_loss = torch.nn.CrossEntropyLoss()

        # Obtain rotation targets and reshape
        rotation_targets = typing.cast(
            torch.LongTensor, batch["observations"]["rot_label"]
        )
        rotation_targets = rotation_targets.view(-1)

        # Obtain rotation prediction logits and reshape
        pred_rotation_logits = pred_rotation_logits.view(
            -1, pred_rotation_logits.shape[2]
        )

        # Compute rotation prediction loss
        rotation_pred_loss = ce_loss(pred_rotation_logits, rotation_targets.squeeze())

        # Compute rotation prediction accuracy
        _, rotation_preds = torch.max(pred_rotation_logits.data, 1)
        correct = (rotation_preds == rotation_targets).sum().float()
        accuracy = 100 * correct / len(rotation_targets)

        total_loss = self.loss_coeff * rotation_pred_loss

        return (
            total_loss,
            {
                "rotation_pred_loss": total_loss.item(),
                "raw_rotation_pred_loss": rotation_pred_loss.item(),
                "raw_rotation_pred_acc": accuracy.item(),
            },
        )


RotPredConfig = dict(loss_coeff=1.0)
