import torch
import numpy as np
from collections import namedtuple
from math import ceil

RolloutBatch = namedtuple(
    'RolloutBatch', ['observations', 'returns', 'actions', 'masks', 'states'])


class KeepTensor:
    def __init__(self, data):
        self.data = data


def to_tensor(value, device):
    if isinstance(value, RolloutBatch):
        return RolloutBatch(*to_tensor(list(value), device))

    elif isinstance(value, list):
        return [to_tensor(x, device) for x in value]

    elif isinstance(value, tuple):
        return tuple(to_tensor(list(value), device))

    elif isinstance(value, dict):
        return {key: to_tensor(val, device) for key, val in value.items()}

    elif isinstance(value, np.ndarray):
        if value.dtype == np.bool:
            value = value.astype(np.float32)

        return torch.from_numpy(value).to(device)
    elif torch.is_tensor(value):
        return value.to(device)
    else:
        raise Exception('%s Not supported' % type(value))


def to_numpy(tensor):
    if isinstance(tensor, KeepTensor):
        return tensor.data
    elif isinstance(tensor, tuple):
        return tuple((to_numpy(x) for x in tensor))
    elif isinstance(tensor, list):
        return [to_numpy(x) for x in tensor]
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, float) or isinstance(tensor, int):
        return tensor
    else:
        raise Exception('Not supported type %s' % type(tensor))


def pytorch_call(device):
    def wrap(function):
        def call(*args, **kwargs):
            results = function(*to_tensor(args, device), **
                               to_tensor(kwargs, device))
            return to_numpy(results)
        return call
    return wrap


def detach_all(data):
    if isinstance(data, list):
        return [detach_all(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(detach_all(list(data)))
    else:
        return data.detach()


def minibatch_gradient_update(inputs, compute_loss_fn, zero_grad_fn, optimize_fn, chunks=1):
    def split_inputs(inputs, chunks, axis):
        if isinstance(inputs, list):
            return list(map(list, split_inputs(tuple(inputs), chunks, axis)))
        elif isinstance(inputs, tuple):
            return list(zip(*[split_inputs(x, chunks, axis) for x in inputs]))
        else:
            return torch.chunk(inputs, chunks, axis)

    # Split inputs to chunks
    if chunks == 1:
        zero_grad_fn()
        losses = compute_loss_fn(*inputs)
        losses[0].backward()
        optimize_fn()
        return [x.item() for x in losses]

    main_inputs = split_inputs(inputs[:-1], chunks, 0)
    states_inputs = split_inputs(inputs[-1:], chunks, 1)
    if len(states_inputs) == 0:
        inputs = [x + ([],) for x in main_inputs]
    else:
        inputs = [x + y for x, y in zip(main_inputs, states_inputs)]

    # Zero gradients
    zero_grad_fn()
    total_results = None
    for minibatch in inputs:
        results = compute_loss_fn(*minibatch)
        results = list(map(lambda x: x / minibatch[1].size(0), results))
        loss = results[0]
        loss.backward()

        if total_results is None:
            total_results = results
        else:
            total_results = list(
                map(lambda x, y: x + y, total_results, results))

    # Optimize
    optimize_fn()
    return [x.item() for x in total_results]


class AutoBatchSizeOptimizer:
    def __init__(self, zero_grad_fn, compute_loss_fn, apply_gradients_fn):
        self._chunks = 1
        self.zero_grad_fn = zero_grad_fn
        self.compute_loss_fn = compute_loss_fn
        self.apply_gradients_fn = apply_gradients_fn

    def optimize(self, inputs):
        batch_size = inputs[1].size()[0]
        results = None
        while results is None:
            try:
                results = minibatch_gradient_update(
                    inputs, self.compute_loss_fn, self.zero_grad_fn, self.apply_gradients_fn, self._chunks)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and self._chunks < batch_size:
                    # We will try to recover from this error
                    torch.cuda.empty_cache()
                    print('ERROR: Training failed with mini-batch size %s' %
                          ceil(float(batch_size) / float(self._chunks)))
                    print('Trying to split the minibatch (%s -> %s)' %
                          (self._chunks, self._chunks + 1))
                    print('Resuming training')
                    self._chunks += 1
                else:
                    raise e

        return results
