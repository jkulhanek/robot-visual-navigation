import argparse
from deep_rl import make_agent
import deep_rl
from configuration import configuration
import numpy as np
from PIL import Image


def _transform_image(img):
    std = 51.764749543249216
    mean = 172.50841217557178
    img = np.array(Image.fromarray(img).resize((84, 84)))
    return np.expand_dims(((img.astype(np.float32) - mean) / std).transpose([2, 0, 1]), 0)


def wrap_agent_observation(agent, is_end):
    old_act = agent.act
    old_reset_state = agent.reset_state
    n_actions = 5 if is_end else 4

    class FinishedAgent:
        def __init__(self):
            self._last_action = None
            self._has_finished = False

        def act(self, obs):
            if self._has_finished:
                return 4

            img, goal = obs
            obs = (_transform_image(img), _transform_image(goal))
            last_state_reward = np.zeros((1, n_actions + 1,), dtype=np.float32)
            if self._last_action:
                last_state_reward[:, self._last_action] = 1.0
            obs = (obs, last_state_reward)
            action = old_act(obs)
            self._last_action = action
            if action[0] == 4:
                self._has_finished = True
            return action[0]

        def reset_state(self):
            self._last_action = None
            self._has_finished = False
            return old_reset_state()

    return FinishedAgent()


def create_agent(name):
    deep_rl.configure(**configuration)
    import trainer

    agent = make_agent(name)
    return wrap_agent_observation(agent, True)
