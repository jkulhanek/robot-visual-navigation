import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class CNN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [layer]
            if activation is not None:
                output.append(getattr(nn, activation)())
            return output

        layers = []
        layers.extend(init_layer(
            nn.Conv2d(num_inputs, 32, 8, stride=4), activation='ReLU'))
        layers.extend(init_layer(
            nn.Conv2d(32, 64, 4, stride=2), activation='ReLU'))
        layers.extend(init_layer(
            nn.Conv2d(64, 32, 3, stride=1), activation='ReLU'))
        layers.append(Flatten())
        layers.extend(init_layer(
            nn.Linear(32 * 7 * 7, 512), activation='ReLU'))

        self.main = nn.Sequential(*layers)
        self.critic = init_layer(nn.Linear(512, 1))[0]
        self.policy_logits = init_layer(
            nn.Linear(512, num_outputs), gain=0.01)[0]

    def forward(self, inputs):
        main_features = self.main(inputs)
        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return policy_logits, critic

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')


def TimeDistributedCNN(num_inputs, num_outputs):
    inner = CNN(num_inputs, num_outputs)
    model = TimeDistributed(inner)
    model.output_names = property(lambda self: inner.output_names)
    _forward = model.forward
    model.forward = lambda inputs, masks, states: _forward(inputs) + (states, )
    return model


class TimeDistributedModel(nn.Module):
    def __init__(self):
        super().__init__()

    def init_layer(self, layer, activation=None, gain=None):
        if activation is not None and gain is None:
            gain = nn.init.calculate_gain(activation.lower())
        elif activation is None and gain is None:
            gain = 1.0

        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)
        output = [TimeDistributed(layer)]
        if activation is not None:
            output.append(TimeDistributed(getattr(nn, activation)()))
        return output

    def init_nondistributed_layer(self, layer, activation=None, gain=None):
        if activation is not None and gain is None:
            gain = nn.init.calculate_gain(activation.lower())
        elif activation is None and gain is None:
            gain = 1.0

        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.zeros_(layer.bias.data)
        output = [layer]
        if activation is not None:
            output.append(getattr(nn, activation)())
        return output


class TimeDistributedConv(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.main_output_size = 512

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [TimeDistributed(layer)]
            if activation is not None:
                output.append(TimeDistributed(getattr(nn, activation)()))
            return output

        layers = []
        layers.extend(init_layer(
            nn.Conv2d(num_inputs, 32, 8, stride=4), activation='ReLU'))
        layers.extend(init_layer(
            nn.Conv2d(32, 64, 4, stride=2), activation='ReLU'))
        layers.extend(init_layer(
            nn.Conv2d(64, 32, 3, stride=1), activation='ReLU'))
        layers.append(TimeDistributed(Flatten()))
        layers.extend(init_layer(
            nn.Linear(32 * 7 * 7, self.main_output_size), activation='ReLU'))
        self.main = nn.Sequential(*layers)

        self.critic = init_layer(nn.Linear(self.main_output_size, 1))[0]
        self.policy_logits = init_layer(
            nn.Linear(512, num_outputs), gain=0.01)[0]

    def forward(self, inputs, masks, states):
        main_features = self.main(inputs)
        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return policy_logits, critic, states

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')


class LSTMConv(TimeDistributedConv):
    def __init__(self, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)

        self.lstm_layers = 1
        self.lstm_hidden_size = 128
        self.rnn = nn.LSTM(self.main_output_size,
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           batch_first=True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        main_features = self.main(inputs)
        main_features, states = forward_masked_rnn_transposed(
            main_features, masks, states, self.rnn.forward)

        policy_logits = self.policy_logits(main_features)
        critic = self.critic(main_features)
        return [policy_logits, critic, states]


class TimeDistributedMultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        def init_layer(layer, activation=None, gain=None):
            if activation is not None and gain is None:
                gain = nn.init.calculate_gain(activation.lower())
            elif activation is None and gain is None:
                gain = 1.0

            nn.init.orthogonal_(layer.weight.data, gain=gain)
            nn.init.zeros_(layer.bias.data)
            output = [TimeDistributed(layer)]
            if activation is not None:
                output.append(TimeDistributed(getattr(nn, activation)()))
            return output

        hidden_size = 512
        self.actor = nn.Sequential(*
                                   init_layer(nn.Linear(input_size, hidden_size), activation='Tanh') +
                                   init_layer(nn.Linear(hidden_size, hidden_size), activation='Tanh') +
                                   init_layer(
                                       nn.Linear(hidden_size, output_size), activation=None, gain=0.01)
                                   )

        self.critic = nn.Sequential(*
                                    init_layer(nn.Linear(input_size, hidden_size), activation='Tanh') +
                                    init_layer(nn.Linear(hidden_size, hidden_size), activation='Tanh') +
                                    init_layer(nn.Linear(hidden_size, 1),
                                               activation=None, gain=1.0)
                                    )

    def forward(self, inputs, masks, states):
        x = inputs
        return self.actor(x), self.critic(x), states


class LSTMMultiLayerPerceptron(TimeDistributedMultiLayerPerceptron):
    def __init__(self, input_size, output_size):
        self.lstm_hidden_size = 64

        super().__init__(self.lstm_hidden_size, output_size)
        self.lstm = nn.LSTM(input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, 1, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features = inputs
        features, states = forward_masked_rnn_transposed(
            features, masks, states, self.lstm.forward)
        return super()(features, masks, states)
