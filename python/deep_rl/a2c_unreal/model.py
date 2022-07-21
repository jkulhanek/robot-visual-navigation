import torch
import torch.nn as nn
import math

from ..a2c.model import TimeDistributed, Flatten
from ..common.pytorch import forward_masked_rnn_transposed


class UnrealModel(nn.Module):
    def init_weights(self, module):
        if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        elif type(module) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            nn.init.zeros_(module.bias.data)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(
                module.weight.data)
            d = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(module.weight.data, -d, d)

    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.conv_base = TimeDistributed(nn.Sequential(
            nn.Conv2d(num_inputs, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
        ))

        self.conv_merge = TimeDistributed(nn.Sequential(
            Flatten(),
            nn.Linear(9 ** 2 * 32, 256),
            nn.ReLU()
        ))

        self.main_output_size = 256

        self.critic = TimeDistributed(nn.Linear(self.main_output_size, 1))
        self.policy_logits = TimeDistributed(
            nn.Linear(self.main_output_size, num_outputs))

        self.lstm_layers = 1
        self.lstm_hidden_size = 256
        self.rnn = nn.LSTM(256 + num_outputs + 1,  # Conv outputs + last action, reward
                           hidden_size=self.lstm_hidden_size,
                           num_layers=self.lstm_layers,
                           batch_first=True)

        self._create_pixel_control_network(num_outputs)
        self._create_rp_network()

        self.pc_cell_size = 4
        self.apply(self.init_weights)

    def initial_states(self, batch_size):
        return tuple([torch.zeros([batch_size, self.lstm_layers, self.lstm_hidden_size], dtype=torch.float32) for _ in range(2)])

    def forward(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        policy_logits = self.policy_logits(features)
        critic = self.critic(features)
        return [policy_logits, critic, states]

    def _forward_base(self, inputs, masks, states):
        observations, last_reward_action = inputs
        features = self.conv_base(observations)
        features = self.conv_merge(features)
        features = torch.cat((features, last_reward_action,), dim=2)
        return forward_masked_rnn_transposed(features, masks, states, self.rnn.forward)

    def _create_pixel_control_network(self, num_outputs):
        self.pc_base = TimeDistributed(nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32 * 9 * 9),
            nn.ReLU()
        ))

        self.pc_action = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
            nn.ReLU()
        ))

        self.pc_value = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, num_outputs, kernel_size=4, stride=2),
            nn.ReLU()
        ))

    def _create_rp_network(self):
        self.rp = nn.Linear(9 ** 2 * 32 * 3, 3)

    def reward_prediction(self, inputs):
        observations, _ = inputs
        features = self.conv_base(observations)
        features = features.view(features.size()[0], -1)
        features = self.rp(features)
        return features

    def pixel_control(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        features = self.pc_base(features)
        features = features.view(*(features.size()[:2] + (32, 9, 9)))
        action_features = self.pc_action(features)
        features = self.pc_value(
            features) + action_features - action_features.mean(2, keepdim=True)
        return features, states

    def value_prediction(self, inputs, masks, states):
        features, states = self._forward_base(inputs, masks, states)
        critic = self.critic(features)
        return critic, states

    @property
    def output_names(self):
        return ('policy_logits', 'value', 'states')
