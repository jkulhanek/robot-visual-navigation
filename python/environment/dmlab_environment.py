import gym
from gym import error, spaces, utils, register, make
import numpy as np

LEVELS = ['custom/house']


def _to_pascal(text):
    text = text.replace('/', '_')
    return ''.join(map(lambda x: x.capitalize(), text.split('_')))


MAP = {_to_pascal(l): l for l in LEVELS}


def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTION_LIST = [
    _action(0,   0,  0,  1, 0, 0, 0),  # forward
    _action(0,   0,  0, -1, 0, 0, 0),  # backward
    _action(-20,   0,  0,  0, 0, 0, 0),  # look_left
    _action(20,   0,  0,  0, 0, 0, 0),  # look_right
    # _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    # _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(0,   0,  0,  0, 0, 1, 0),  # collect object
]


class DeepmindLabEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, scene, screen_size=(84, 84), renderer="hardware", **kwargs):
        import deepmind_lab

        super(DeepmindLabEnv, self).__init__(**kwargs)
        height, width = screen_size

        if scene in LEVELS:
            raise Exception('Scene %s not supported' % (scene))

        self._colors = ['RGBD_INTERLEAVED', 'GOAL_RGB_INTERLEAVED', 'DISTANCE', 'SHORTEST_DISTANCE']
        self._lab = deepmind_lab.Lab(scene, self._colors,
                                     dict(fps=str(60), width=str(width), height=str(height)),
                                     renderer=renderer)

        self.action_space = gym.spaces.Discrete(len(ACTION_LIST))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
            gym.spaces.Box(0, 225, (height, width, 3), dtype=np.uint8),
            gym.spaces.Box(0, 225, (height, width, 1), dtype=np.uint8)))

        self._last_observation = None
        self._distance = None
        self._shortestDistance = None

    def step(self, action):
        reward = self._lab.step(ACTION_LIST[action], num_steps=4) / 10.0
        terminal = not self._lab.is_running()
        obs = None if terminal else self.observe(self._lab.observations())
        self._last_observation = obs if obs is not None else tuple([np.copy(x) for x in list(self._last_observation)])
        return self._last_observation, reward, terminal, dict(distance=self._distance, shortest_distance=self._shortestDistance)

    def metrics(self):
        return (self._distance, self._shortestDistance)

    def observe(self, obs):
        self._distance = obs['DISTANCE']
        self._shortestDistance = obs['SHORTEST_DISTANCE']
        return (
            obs[self._colors[0]][:, :, :3],
            obs[self._colors[1]],
            obs[self._colors[0]][:, :, 3:4]
        )

    def reset(self):
        self._distance = None
        self._shortestDistance = None
        self._lab.reset()
        self._last_observation = self.observe(self._lab.observations())
        return self._last_observation

    def seed(self, seed=None):
        self._lab.reset(seed=seed)

    def close(self):
        self._lab.close()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._lab.observations()['RGBD_INTERLEAVED'][:, :, :3]
        # elif mode is 'human':
        #   pop up a window and render
        else:
            super(DeepmindLabEnv, self).render(mode=mode)  # just raise an exception


def _action(*entries):
    return np.array(entries, dtype=np.intc)
