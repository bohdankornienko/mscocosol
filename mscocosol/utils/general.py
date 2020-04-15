import sys

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
