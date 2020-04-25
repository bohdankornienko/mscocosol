import sys
import torch

from graphviz import Digraph
from io import StringIO
from pprint import pprint
from torchsummary import summary

import numpy as np


def pretty_dict_as_string(dictitionary):
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    pprint(dictitionary)

    sys.stdout = old_stdout
    result_string = result.getvalue()

    return result_string


def model_summary_as_string(model, shape=(3, 192, 192), device='cuda'):
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    summary(model, input_size=shape, device=device)

    sys.stdout = old_stdout
    result_string = result.getvalue()

    return result_string


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Origin of the function: https://discuss.pytorch.org/t/print-autograd-graph/692/16

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def squash_mask(mask):
    """
    Takes input multichannel mask and converts it into single channel mask where each pixel takes value
    corresponding to its class
    :param mask: Segmentation mask with the shape (H, W, C) where C is the number of classes.
    :return: Squashed mask.
    """
    squashed_mask = np.zeros(mask.shape[:2], dtype=np.long)

    num_classes = mask.shape[2]
    for c in reversed(range(num_classes)):
        squashed_mask[mask[:, :, c] == 1] = c

    return squashed_mask


def blowup_mask_torch(pred, n_class, mask_shape=(192, 192)):
    batch_size = pred.shape[0]
    res_batch = torch.zeros(batch_size, n_class, mask_shape[0], mask_shape[1], dtype=torch.long)
    res_batch.to(pred.device)

    for b in range(batch_size):
        res_mask = torch.zeros(n_class, mask_shape[0], mask_shape[1], dtype=torch.long)
        for i in range(n_class):
            res_mask[i][pred[b] == i] = 1
        res_batch[b] = res_mask

    return res_batch
