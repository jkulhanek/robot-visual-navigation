import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .core import pytorch_call, AutoBatchSizeOptimizer
from .a2c import A2CTrainer


class A2CTrainerDynamicBatch(A2CTrainer):
    def _build_train(self, model, main_device):
        optimizer = optim.RMSprop(model.parameters(
        ), self.learning_rate, eps=self.rms_epsilon, alpha=self.rms_alpha)

        def compute_loss(observations, returns, actions, masks, states):
            policy_logits, value, _ = model(observations, masks, states)

            dist = torch.distributions.Categorical(logits=policy_logits)
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()

            # Compute losses
            advantages = returns - value.squeeze(-1)
            value_loss = advantages.pow(2).mean()
            action_loss = -(advantages.detach() * action_log_probs).mean()
            loss = value_loss * self.value_coefficient + \
                action_loss - \
                dist_entropy * self.entropy_coefficient

            return loss, action_loss.detach(), value_loss.detach(), dist_entropy.detach()

        def optimize():
            nn.utils.clip_grad_norm_(
                model.parameters(), self.max_gradient_norm)
            optimizer.step()

        zero_grad = optimizer.zero_grad
        batch_optimizer = AutoBatchSizeOptimizer(
            zero_grad, compute_loss, optimize)

        @pytorch_call(main_device)
        def train(*args):
            return tuple(batch_optimizer.optimize(args))

        return train
