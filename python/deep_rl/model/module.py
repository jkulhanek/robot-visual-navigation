import torch
from torch import nn
from ..common.pytorch import forward_masked_rnn_transposed


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class TimeDistributed(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args):
        batch_shape = args[0].size()[:2]
        args = [x.contiguous().view(-1, *x.size()[2:]) for x in args]
        results = self.inner(*args)

        def reshape_res(x):
            return x.view(*(batch_shape + x.size()[1:]))

        if isinstance(results, list):
            return [reshape_res(x) for x in results]
        elif isinstance(results, tuple):
            return tuple([reshape_res(x) for x in results])
        else:
            return reshape_res(results)


class MaskedRNN(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, inputs, masks, states):
        return forward_masked_rnn_transposed(inputs, masks, states, self.inner)
