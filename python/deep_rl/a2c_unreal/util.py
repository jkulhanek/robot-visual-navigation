import torch
import torch.nn.functional as F
import gym
import gym.spaces
import numpy as np


def autocrop_observations(observations, cell_size, output_size=None):
    shape = observations.size()[3:]
    if output_size is None:
        new_shape = tuple(map(lambda x: (x // cell_size) * cell_size, shape))
    else:
        new_shape = tuple(map(lambda x: x * cell_size, output_size))

    margin3_top = (shape[0] - new_shape[0]) // 2
    margin3_bottom = -(shape[0] - new_shape[0] - margin3_top)
    margin4_top = (shape[1] - new_shape[1]) // 2
    margin4_bottom = -(shape[1] - new_shape[1] - margin4_top)
    if margin3_bottom == 0:
        margin3_bottom = None
    if margin4_bottom == 0:
        margin4_bottom = None

    return observations[:, :, :, margin3_top:margin3_bottom, margin4_top:margin4_bottom]


def pixel_control_reward(observations, cell_size=4, output_size=None):
    '''
    Args:
    observations: A tensor of shape `[B,T+1,C,H,W]`, where
      * `T` is the sequence length, `B` is the batch size.
      * `H` is height, `W` is width.
      * `C...` is at least one channel dimension (e.g., colour, stack).
      * `T` and `B` can be statically unknown.
    cell_size: The size of each cell.
    Returns:
        shape (B, T, 1, H / cell_size, W / cell_size)
    '''
    with torch.no_grad():
        observations = autocrop_observations(
            observations, cell_size, output_size=output_size)
        abs_observation_diff = observations[:, 1:] - observations[:, :-1]
        abs_observation_diff.abs_()
        obs_shape = abs_observation_diff.size()
        abs_diff = abs_observation_diff.view(-1, *obs_shape[2:])

        avg_abs_diff = F.avg_pool2d(abs_diff, cell_size, stride=cell_size)
        avg_abs_diff = avg_abs_diff.mean(1, keepdim=True)
        return avg_abs_diff.view(*obs_shape[:2] + avg_abs_diff.size()[1:])


def pixel_control_loss(observations, actions, action_values, gamma=0.9, cell_size=4):
    action_value_shape = action_values.size()
    batch_shape = actions.size()[:2]
    with torch.no_grad():
        T = observations.size()[1] - 1
        pseudo_rewards = pixel_control_reward(
            observations, cell_size, output_size=action_values.size()[-2:])
        last_rewards = action_values[:, -1].max(1, keepdim=True)[0]
        for i in reversed(range(T)):
            previous_rewards = last_rewards if i + \
                1 == T else pseudo_rewards[:, i + 1]
            pseudo_rewards[:, i].add_(previous_rewards, alpha=gamma)

    q_actions = actions.view(*batch_shape + (1, 1, 1)).repeat(1,
                                                              1, 1, action_value_shape[3], action_value_shape[4])
    q_actions = torch.gather(action_values[:, :-1], 2, q_actions)

    loss = F.mse_loss(pseudo_rewards, q_actions)
    return loss


def reward_prediction_loss(predictions, rewards):
    with torch.no_grad():
        target = torch.zeros(predictions.size(),
                             dtype=torch.float32, device=predictions.device)
        target[:, 0] = rewards == 0
        target[:, 1] = rewards > 0
        target[:, 2] = rewards < 0

    return F.binary_cross_entropy_with_logits(predictions, target)


def discounted_commulative_reward(rewards, base_value, gamma):
    cummulative_reward = rewards.clone()
    max_t = cummulative_reward.size()[1]
    for i in reversed(range(max_t)):
        next_values = base_value if i + \
            1 == max_t else cummulative_reward[:, i + 1]
        cummulative_reward[:, i].add_(next_values, alpha=gamma)

    return cummulative_reward


def value_loss(values, rewards, gamma):
    base_value = values[:, -1]
    with torch.no_grad():
        cummulative_reward = discounted_commulative_reward(
            rewards, base_value, gamma)
    return F.mse_loss(values[:, :-1], cummulative_reward)


class UnrealEnvBaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action_reward = None
        self.observation_space = gym.spaces.Tuple((
            env.observation_space,
            gym.spaces.Box(0.0, 1.0, (env.action_space.n + 1,),
                           dtype=np.float32)
        ))

    def reset(self):
        self.last_action_reward = np.zeros(
            self.action_space.n + 1, dtype=np.float32)
        return self.observation(self.env.reset())

    def step(self, action):
        observation, reward, done, stats = self.env.step(action)
        self.last_action_reward = np.zeros(
            self.action_space.n + 1, dtype=np.float32)
        self.last_action_reward[action] = 1.0
        self.last_action_reward[-1] = np.clip(reward, -1, 1)
        return self.observation(observation), reward, done, stats

    def observation(self, observation):
        return (observation, self.last_action_reward)
