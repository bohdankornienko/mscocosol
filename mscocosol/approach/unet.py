import torch

import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torchsummary import summary
from collections import defaultdict

from mscocosol.approach.models.torch.unet import UNet as UNetModel

import logging

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


class UNetTorch:
    def __init__(self, **kwargs):
        # TODO: set mode from arugments
        # TODO: create parent abstract class for approach

        self._sets = kwargs
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Computation device: {}'.format(self._device))

        self._model = UNetModel(self._sets['classes_num'])
        self._model = self._model.to(self._device)

        # TODO: control it with arguments, maybe ?
        # TODO: input size for summary should match with input data
        # TODO: should I leave print?
        print(summary(self._model, input_size=(3, 192, 192)))

        self._model = UNetModel(self._sets['classes_num']).to(self._device)
        self._model = self._model.to(self._device)
        logging.info('Check if model is on cuda: {}'.format(next(self._model.parameters()).is_cuda))

        # TODO: control input parameters with sets
        self._optimizer = optim.Adam(self._model.parameters(), lr=1e-4)
        self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size=25, gamma=0.1)

        self._metrics = defaultdict(float)

        self._mode = 'train'
        self._epoch_samples = 0
        self._last_evaluations = None

    def train_on_batch(self, x, y):
        inputs = x.to(self._device)
        labels = y.to(self._device)
        '''
        inputs = x.to(self._device, dtype=torch.double)
        labels = y.to(self._device, dtype=torch.double)
        '''

        # zero the parameter gradients
        self._optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(self._mode == 'train'):
            outputs = self._model(inputs)
            loss = calc_loss(outputs, labels, self._metrics)

            # backward + optimize only if in training phase
            if self._mode == 'train':
                loss.backward()
                self._optimizer.step()

        # statistics
        self._epoch_samples += inputs.size(0)

        readings = {'mode': self._mode, 'epoch_samples': self._epoch_samples}
        if self._mode == 'train':
            readings['loss'] = loss

        return readings

    def eval_on_batch(self, x, y):
        y_pred = self.predict(x)

    def predict(self, x):
        pass

    def set_mode(self, mode):
        """
        Assign the mode in which approach is being executed.
        :param mode: train for training; val, test, inference for prediction mode
        """

        self._epoch_samples = 0
        if mode == 'train':
            self._scheduler.step()
            for param_group in self._optimizer.param_groups:
                print("LR", param_group['lr'])

            self._model.train()
        else:
            self._model.eval()

    def save_checkpoint(self, epoch, iteration=None):
        pass

        # TODO: mark checkpoint as best

    def _calc_metrics(self):
        outputs = ['\n']
        for k in self._metrics.keys():
            outputs.append("{}: {:4f}".format(k, self._metrics[k] / self._epoch_samples))

        metrics = "{}: {}".format(self._mode, ", ".join(outputs))
        metrics += '\n'

        return metrics

    def evaluate(self):

        self._metrics = self._calc_metrics()

        # TODO: evaluate the model
        # TODO: save as "last evaluation results"
        self._last_evaluations = self._metrics

    def get_last_evaluation(self):
        return self._last_evaluations


def make_unet_torch(**kwargs):
    return UNetTorch(**kwargs)

