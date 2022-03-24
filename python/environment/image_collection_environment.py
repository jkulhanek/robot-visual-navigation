import os
import h5py
import gym
from gym.wrappers import TimeLimit
import numpy as np
import random

ACTION_LIST = []


def move_position(position, mult=1):
    rotation = position[2]
    if rotation == 0:
        return (position[0] + 1 * mult, position[1], rotation)
    elif rotation == 1:
        return (position[0], position[1] + 1 * mult, rotation)
    elif rotation == 2:
        return (position[0] - 1 * mult, position[1], rotation)
    elif rotation == 3:
        return (position[0], position[1] - 1 * mult, rotation)


def compute_complexity_distance(pointa, pointb):
    ax, ay, ar = pointa
    bx, by, br = pointb
    return abs(ax - bx) + abs(ay - by) + abs((ar - br + 1) % 4 - 1) * 2


class ImageEnvironmentWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageEnvironmentWrapper, self).__init__(env)
        self._stds = [51.764749543249216, 51.764749543249216, 1064.4242973195394, 1064.4242973195394]
        self._means = [172.50841217557178, 172.50841217557178, 980.5952, 980.5952]
        self.observation_space = gym.spaces.Tuple(
            tuple([gym.spaces.Box(-1.0, 1.0, x.shape, dtype=np.float32) for x in self.observation_space.spaces]))

    def observation(self, observation):
        return [(o.astype(np.float32) - m) / s for o, m, s in zip(list(observation), self._means, self._stds)]


class ImageEnvironment(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, screen_size=(84, 84), dataset_name='turtle_room', path=None, has_end_action=False, augment_images=True, **kwargs):
        super(ImageEnvironment, self).__init__(**kwargs)
        if path is None:
            path = os.path.expanduser(f'/mnt/ciirc/.cache/robot-visual-navigation/datasets/{dataset_name}_compiled.hdf5')
        self.path = path
        self.has_end_action = has_end_action
        self._file = None
        self.augment_images = augment_images
        self._datasetSelector = "%sx%s" % screen_size

        height, width = screen_size
        self.action_space = gym.spaces.Discrete(4 if not has_end_action else 5)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
            gym.spaces.Box(0, 225, (height, width, 3), dtype=np.uint8),
            gym.spaces.Box(0, 225, (height, width, 1), dtype=np.uint16),
            gym.spaces.Box(0, 225, (height, width, 1), dtype=np.uint16)))

        self._last_observation = None
        self._initialized = False
        self._random = random.Random()
        self._next_task = None
        self._physicalPosition = None
        self.complexity = None

    def _initialize(self):
        if self._initialized:
            return False
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        self._positions = self._file["grid"]
        self._physicalPositions = self._file["positions"]
        self._images = self._file[self._datasetSelector + ("/augmented_images" if self.augment_images else "/images")]
        self._depths = self._file[self._datasetSelector + ("/augmented_depths" if self.augment_images else "/depths")]
        wid, hei, _, nsamples = self._positions.shape
        self._allowedPoints = set()
        self._goalPoints = []
        self._nongoalPoints = []
        for x in range(wid):
            for y in range(hei):
                isany = all(any(self._positions[x, y, r, i] != -1 for i in range(nsamples)) for r in range(4))
                if isany:
                    self._allowedPoints.add((x, y))

        for (x, y) in self._allowedPoints:
            for r in range(4):
                xn, yn, _ = move_position((x, y, r))
                if (xn, yn) not in self._allowedPoints:
                    self._goalPoints.append((x, y, r))
                else:
                    self._nongoalPoints.append((x, y, r))
        self._initialized = True

        # Prepare complexity lookup
        self._complexityCache = dict()
        for g in self._goalPoints:
            cmpCache = []
            self._complexityCache[g] = cmpCache
            for i in range(wid + hei + 4 + 1):
                startPoints = []
                cmpCache.append(startPoints)
                for x in self._nongoalPoints:
                    dist = compute_complexity_distance(g, x)
                    if dist == i:
                        startPoints.append(x)

    def set_complexity(self, complexity=None):
        self.complexity = complexity

    def step(self, action):
        assert self._initialized
        self._position, collided = self._move(self._position, action)
        terminal = self.is_goal(self._position)
        reward = 1.0 if terminal else (-0.01 if collided else 0)
        if self.has_end_action:
            if action == 4:
                if terminal:
                    reward = 1.0
                    terminal = True
                else:
                    reward = 0
                    terminal = True
            else:
                reward = (-0.01 if collided else 0)
                terminal = False
        obs = None if terminal else self._observe()
        self._last_observation = obs if obs is not None else tuple([np.copy(x) for x in list(self._last_observation)])
        return self._last_observation, reward, terminal, dict()

    def _ensure_in_grid(self, position):
        x, y, _ = position
        return (x, y) in self._allowedPoints

    def _move(self, position, action):
        x, y, r = position
        if action == 0:
            # Forward
            npos = move_position(position, 1)
            if self._ensure_in_grid(npos):
                return npos, False
            else:
                return position, True
        elif action == 1:
            # Backward
            npos = move_position(position, -1)
            if self._ensure_in_grid(npos):
                return npos, False
            else:
                return position, True
        elif action == 2:
            # Left
            npos = (x, y, (r + 1) % 4)
            return npos, True
        elif action == 3:
            # Right
            npos = (x, y, (r - 1) % 4)
            return npos, True
        else:
            return position, False

    def _observe(self):
        x, y, r = self._position
        index = self._positions[x, y, r, self._random.randrange((self._positions[x, y, r] != -1).sum())]
        self._physicalPosition = (self._physicalPositions[index], self._physicalPositions[self._goalIndex])
        indexg = self._goalIndex
        if self.augment_images:
            irender = self._random.randrange(self._images.shape[1])
            return (
                self._images[index, irender, ...],
                self._images[indexg, self._goalRender, ...],
                np.expand_dims(self._depths[index, irender, ...], 2),
                np.expand_dims(self._depths[indexg, self._goalRender, ...], 2)
            )
        else:
            return (
                self._images[index, ...],
                self._images[indexg, ...],
                np.expand_dims(self._depths[index, ...], 2),
                np.expand_dims(self._depths[indexg, ...], 2)
            )

    def set_next_task(self, position, goal):
        self._next_task = (position, goal)

    @property
    def position(self):
        c, g = self._physicalPosition
        return list(c[:3]), list(g[:3])

    def reset(self):
        self._initialize()
        # Sample goal
        self._goal = self.sample_goal() if self._next_task is None else self._next_task[1]
        xg, yg, rg = self._goal
        self._goalIndex = self._positions[xg, yg, rg, self._random.randrange((self._positions[xg, yg, rg] != -1).sum())]
        if self.augment_images:
            self._goalRender = self._random.randrange(self._images.shape[1])
        self._position = self.sample_position(self._goal) if self._next_task is None else self._next_task[0]
        self._last_observation = self._observe()
        self._next_task = None
        return self._last_observation

    def sample_goal(self):
        # Sample a goal on the edge of the grid
        return self._random.choice(self._goalPoints)

    def sample_position(self, goal):
        choiceArray = None
        if self.complexity is None:
            choiceArray = self._nongoalPoints
        else:
            choiceArray = []
            for i in range(min(len(self._complexityCache[goal]), self.complexity + 1)):
                choiceArray.extend(self._complexityCache[goal][i])

        return self._random.choice(choiceArray)

    def is_goal(self, position):
        diff = abs(self._goal[0] - position[0]) + abs(self._goal[1] - position[1])
        return diff <= 1 and self._goal[2] == position[2]

    def seed(self, seed=None):
        self._random.seed(seed)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        self._initialized = False

    def browse(self):
        from .browser import GoalKeyboardAgent
        agent = GoalKeyboardAgent(self, [0, 1, 2, 3])
        agent.show()

    def render(self, mode='rgb_array', close=False):
        if mode == 'rgb_array':
            return self._last_observation[0]
        # elif mode is 'human':
        #   pop up a window and render
        else:
            super(ImageEnvironment, self).render(mode=mode)  # just raise an exception


if __name__ == "__main__":
    from PIL import Image
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    w = TimeLimit(ImageEnvironment(), max_episode_steps=300)
    i = Image.open("../assets/images/63.png")
    plt.imshow(w._image_aug(images=[np.array(i)])[0])
    plt.show()
