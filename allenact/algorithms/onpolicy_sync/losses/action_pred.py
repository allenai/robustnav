"""Defining the action-prediction loss for actor critic type models."""

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


class ActionPred(AbstractActorCriticLoss):
    """Implementation of the action prediction loss.
    Given two successive frames, predict the intermediate action
    NOTE: Excludes the 'END' action

    # Attributes

    loss_coeff : Weight of the action-prediction loss
    """

    def __init__(self, loss_coeff, *args, **kwargs):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.loss_coeff = loss_coeff

    def loss(  # type: ignore
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        pred_action_logits: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        # Cross-entropy criteria for action prediction
        ce_loss = torch.nn.CrossEntropyLoss()

        # Obtain actions and reshape
        actions = typing.cast(torch.LongTensor, batch["actions"])
        actions = actions.view(-1)

        # Obtain action-prediction logits and reshape
        pred_action_logits = pred_action_logits.view(-1, pred_action_logits.size(2))

        # Ignore instances where the intermediate action is END
        nend_bin = actions != 3
        nend_ind = nend_bin.nonzero().long()
        actions = actions[nend_ind]
        actions = torch.where(actions > 3, actions - 1, actions)
        pred_action_logits = pred_action_logits[nend_ind, :]

        actions = actions.squeeze()
        pred_action_logits = pred_action_logits.squeeze()

        if actions.nelement() == 0:
            # Hardcode action prediction loss and accuracy
            # for this edge case (occurs mostly in the beginning of training)
            # for small episodes
            action_pred_loss = torch.tensor(0).float()
            accuracy = torch.tensor(0).float()
        else:
            if len(pred_action_logits.size()) == 1:
                actions = actions.unsqueeze(0)
                pred_action_logits = pred_action_logits.unsqueeze(0)

            # Compute action prediction loss and accuracy
            action_pred_loss = ce_loss(pred_action_logits, actions)

            # Compute action prediction accuracy
            _, predicted_actions = torch.max(pred_action_logits.data, 1)
            correct = (predicted_actions == actions).sum().float()
            accuracy = 100 * correct / len(actions)

        total_loss = self.loss_coeff * action_pred_loss

        return (
            total_loss,
            {
                "action_pred_loss": total_loss.item(),
                "raw_action_pred_loss": action_pred_loss.item(),
                "raw_action_pred_acc": accuracy.item(),
            },
        )


ActPredConfig = dict(loss_coeff=1.0)
