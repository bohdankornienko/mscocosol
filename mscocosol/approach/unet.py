import os
import torch
import logging

import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler
from collections import defaultdict

import numpy as np

from mscocosol.approach.models.torch.unet import make_unet
from mscocosol.utils.general import model_summary_as_string


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

    metrics['bce'] = bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] = dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] = loss.data.cpu().numpy() * target.size(0)

    return loss


class UNetTorch:
    def __init__(self, **kwargs):
        # TODO: set mode from arguments
        # TODO: create parent abstract class for approach

        self._sets = kwargs
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Computation device: {}'.format(self._device))

        # TODO: should I keep this around?
        self._model = make_unet(self._sets['model_variant'], self._sets['classes_num'])
        self._model = self._model.to(self._device)

        self._metrics_train = defaultdict(float)
        self._metrics_val = defaultdict(float)
        self._metrics = self._metrics_train

        # TODO: control it with arguments, maybe ?
        # TODO: input size for summary should match with input data
        # TODO: should I leave print?
        logging.info(model_summary_as_string(self._model))

        self._model = make_unet(self._sets['model_variant'], self._sets['classes_num']).to(self._device)
        self._model = self._model.to(self._device)
        logging.info('Check if model is on cuda: {}'.format(next(self._model.parameters()).is_cuda))

        # TODO: control input parameters with sets
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._sets['optimizer']['learning_rate'])
        self._scheduler = lr_scheduler.StepLR(self._optimizer, step_size=self._sets['optimizer']['lr_scheduler_step_size'], gamma=0.1)

        self.set_mode(kwargs['mode'])

        self._epoch_samples = 0
        self._last_evaluations = None

        if self._mode == 'train':
            self._checkpoints_dir = os.path.join(self._sets['this_exp_dir'], 'checkpoints')
            if not os.path.exists(self._checkpoints_dir):
                os.mkdir(self._checkpoints_dir)

    def train_on_batch(self, x, y):
        # TODO: do traonsfrom to torch.tenosor here insetad of datagetnerator
        inputs = x.to(self._device)
        labels = y.to(self._device)

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

        readings = {'mode': self._mode, 'epoch_samples': self._epoch_samples, 'metrics': self._metrics}

        return readings

    def eval_on_batch(self, x, y):
        pass

    def predict(self, x):
        self._y_pred = self._model(x.to(self._device))
        self._y_pred_s = F.sigmoid(self._y_pred)
        return self._y_pred_s > 0.5

    def set_mode(self, mode):
        """
        Assign the mode in which approach is being executed.
        :param mode: train for training; val, test, inference for prediction mode
        """

        if mode == 'train':
            self._mode = 'train'
            self._metrics = self._metrics_train
        else:
            self._mode = 'inference'
            self._metrics = self._metrics_val

        self._epoch_samples = 0

        if mode == 'train':
            self._scheduler.step()
            for param_group in self._optimizer.param_groups:
                print("LR", param_group['lr'])

            self._model.train()
        else:
            self._model.eval()

    def save_checkpoint(self, epoch, iteration=None):
        model_file_name = '{:0>5d}'.format(epoch)
        if iteration is not None:
            model_file_name += '_{:0>10d}'.format(iteration)
        model_file_name += '.pt'

        torch.save(self._model.state_dict(), os.path.join(self._checkpoints_dir, model_file_name))

        # TODO: mark checkpoint as best
        # TODO: saving strategies
        #  all - saves all checkpoint,
        #  better - saves checkpoint only if readings are better than previous,
        #  best - similar to better but removes all previous checkpoints

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))

    def _calc_metrics(self):
        outputs = ['\n']
        for k in self._metrics.keys():
            # outputs.append("{}: {:4f}".format(k, self._metrics[k] / self._epoch_samples))
            outputs.append("{}: {:4f}".format(k, self._metrics[k]))

        metrics = "{}: {}".format(self._mode, ", ".join(outputs))
        metrics += '\n'

        return metrics

    def evaluate(self, data_gen):
        precisions = list()
        recalls = list()

        for x, y_true in data_gen:
            y_pred = self.predict(x)

            # TODO: do computations directly on GPU
            y_pred = y_pred.detach().cpu().numpy()
            y_true = y_true.detach().cpu().numpy()

            TP = np.logical_and(y_true, y_pred)
            preds = np.logical_or(y_true, y_pred)

            R = np.sum(TP, axis=(1, 2, 3)) / np.sum(y_true, axis=(1, 2, 3))
            P = np.sum(TP, axis=(1, 2, 3)) / np.sum(preds, axis=(1, 2, 3))

            precisions.append(np.mean(P))
            recalls.append(np.mean(R))

        mAr = np.mean(recalls)
        mAp = np.mean(precisions)

        '''
        TODO: add losses and stuff
        self._metrics = self._calc_metrics()

        # TODO: evaluate the model
        # TODO: save as "last evaluation results"
        '''
        self._last_evaluations = None

        evaluations = dict()
        evaluations['map'] = float(mAp)
        evaluations['mar'] = float(mAr)

        return evaluations

    def get_last_evaluation(self):
        return self._last_evaluations


def make_unet_torch(**kwargs):
    return UNetTorch(**kwargs)
