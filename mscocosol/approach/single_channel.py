import logging

import torch
from torch import nn

from .torch_based import TorchBasedApproach
from mscocosol.utils.general import blowup_mask_torch


class SingleChannelApproach(TorchBasedApproach):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self._sets['weighted_loss']:
            weights = torch.ones(self._sets['classes_num'])
            weights[-1] = 0.2
            weights = weights.to(self._sets['device'])
            self._loss = nn.NLLLoss(weight=weights)
        else:
            self._loss = nn.NLLLoss()

    def _calc_loss(self, y_pred, y_true):
        readings = dict()

        # TODO: dynamically calculate class weights
        loss = self._loss(y_pred, y_true)

        readings['loss'] = float(loss.data.cpu().numpy())

        return loss, readings

    def predict(self, x):
        with torch.no_grad():
            inputs = x.to(self._device)

            pred = self._model(inputs)

            pred = pred.exp().detach()  # exp of the log prob = probability.
            mask = torch.argmax(pred, 1)  # index of the class with maximum probability.
            self._pred = pred.cpu().numpy()
            self._mask = mask.cpu().numpy()

            self._mask_blown = blowup_mask_torch(pred=mask, n_class=self._sets['classes_num'], mask_shape=self._sets['target_img_size'])
            self._y_pred = pred

            return self._mask_blown


def make_single_channel_approach(**kwargs):
    return SingleChannelApproach(**kwargs)
