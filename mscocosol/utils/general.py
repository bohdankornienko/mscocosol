import sys

from io import StringIO
from pprint import pprint
from torchsummary import summary


def pretty_dict_as_string(dictitionary):
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    pprint(dictitionary)

    sys.stdout = old_stdout
    result_string = result.getvalue()

    return result_string


def model_summary_as_string(model):
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result

    summary(model, input_size=(3, 192, 192))

    sys.stdout = old_stdout
    result_string = result.getvalue()

    return result_string


