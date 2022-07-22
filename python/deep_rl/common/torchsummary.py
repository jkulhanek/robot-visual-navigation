import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def sample_space(sizes, dtype):
    if isinstance(sizes, tuple):
        if len(sizes) == 0:
            return tuple()
        elif type(sizes[0]) == int:
            return torch.rand(*sizes).type(dtype)
        else:
            return tuple(sample_space(list(sizes), dtype))
    elif isinstance(sizes, list):
        return [sample_space(x, dtype) for x in sizes]
    else:
        raise Exception('Not supported')


def sum_space(sizes):
    if isinstance(sizes, tuple):
        if len(sizes) == 0:
            return 0
        elif type(sizes[0]) == int:
            return np.prod(list(sizes))
        else:
            return sum_space(list(sizes))
    elif isinstance(sizes, list):
        return np.sum([sum_space(x) for x in sizes])
    else:
        return sizes


def shrink_shape(shape):
    res = None
    if isinstance(shape, tuple):
        res = shrink_shape(list(shape))
        if len(res) == 0 or isinstance(res[0], (tuple, list)):
            res = tuple(res)
    elif isinstance(shape, list):
        res = [shrink_shape(x) for x in shape]

    if res is not None:
        if len(res) == 1:
            shape = res[0]
        else:
            shape = res

    return shape


def get_shape(tensor, shrink=False):
    if shrink:
        return shrink_shape(get_shape(tensor))

    if isinstance(tensor, tuple):
        return tuple(get_shape(list(tensor)))
    elif isinstance(tensor, list):
        return [get_shape(x) for x in tensor]
    else:
        return list(tensor.size())


def summary(model, input_size, device="cuda"):
    # create properties
    summary = OrderedDict()
    hooks = []
    registered_modules = set()
    hook_dict = dict()

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            if module.__class__.__name__ == 'TimeDistributed':
                class_name = str(module.inner.__class__).split(
                    ".")[-1].split("'")[0]

            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())

            if isinstance(output, (tuple, list)) and isinstance(output[-1], tuple):
                summary[m_key]["output_shape"] = get_shape(output[:-1], True)
                summary[m_key]["state_shape"] = get_shape(output[-1])
            else:
                summary[m_key]["output_shape"] = get_shape(output, True)

            params = 0
            summary[m_key]["nb_params"] = sum_space(
                [x.size() for x in module.parameters()])
            summary[m_key]["nb_trainable_params"] = sum_space(
                [x.size() for x in module.parameters() if x.requires_grad])

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not module in registered_modules
            and not (module == model)
        ):
            hook_obj = module.register_forward_hook(hook)
            hooks.append(hook_obj)
            hook_dict[module] = hook_obj
            if module.__class__.__name__ == 'TimeDistributed':
                registered_modules.add(module.inner)
                if module.inner in hook_dict:
                    hook_obj = hook_dict.pop(module.inner)
                    hook_obj.remove()
                    hooks.remove(hook_obj)

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # batch_size of 2 for batchnorm
    x = sample_space(input_size, dtype)
    # print(type(x[0]))

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += sum_space(summary[layer]["output_shape"])
        trainable_params += summary[layer]["nb_trainable_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(sum_space(input_size) * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


def minimal_summary(model, input_size):
    # assume 4 bytes/number (float on cuda).
    total_params = sum_space([x.size() for x in model.parameters()])
    trainable_params = sum_space(
        [x.size() for x in model.parameters() if x.requires_grad])

    total_input_size = abs(sum_space(input_size) * 4. / (1024 ** 2.))
    total_params_size = abs(total_params * 4. / (1024 ** 2.))

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("================================================================")
