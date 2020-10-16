from deep_rl.core import AbstractAgent
from deep_rl import register_agent
from deep_rl import configuration
import numpy as np
import os
import random
from collections import deque

@register_agent("random", is_end = True)
class RandomAgent(AbstractAgent):
    def __init__(self, *args, is_end = True, **kwargs):
        self.is_end = is_end
        super().__init__(*args, **kwargs)

    def act(self, o):
        return [random.randrange(5 if self.is_end else 4)]

@register_agent("random-end", is_end = True)
class RandomAgent(AbstractAgent):
    def __init__(self, *args, is_end = True, **kwargs):
        self.is_end = is_end
        super().__init__(*args, **kwargs)
        self.env = None

    def set_environment(self, env):
        self.env = env.unwrapped

    def act(self, o):
        return [random.randrange(4)] if self.env._position != self.env._goal else [4]

@register_agent("turtleroom-constant-stochastic")
class StochasticAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_initialized = False

    def _initialize(self):
        if self._is_initialized:
            return

        self._is_initialized = True
        checkpoint_dir = configuration.get('models_path')

        path = os.path.join(checkpoint_dir, self.name, 'distribution.npy')
        actions = np.load(path)
        self._actions = actions

    def act(self, obs):
        return [np.random.choice(list(range(len(self._actions))), p = self._actions)]

@register_agent("shortest-path")
class ShortestPathAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = None
        self.steps = []

    def set_environment(self, env):
        self.env = env.unwrapped

    def reset_state(self):
        self.steps = []

    def _compute_optimal_steps(self):
        s = self._find_reversed_path(self.env._position, self.env._goal)
        s.reverse()
        return s

    def _find_reversed_path(self, position, goal):
        closed = set([position])
        o = deque([(position, [])])
        MAX_PATH = 38
        while len(o) != 0:
            position, s = o.popleft()
            if position == goal:
                s.append(4)
                return s

            if len(s) == MAX_PATH:
                continue

            for a in range(4):
                npos, _ = self.env._move(position, a)
                if npos in closed:
                    continue
                sn = list(s)
                sn.append(a)
                o.append((npos, sn))
                closed.add(npos)
        return None     
            
    def act(self, o):
        if self.env is None:
            raise Exception("Must call 'set_environment'")
            
        if len(self.steps) == 0:
            self.steps = self._compute_optimal_steps()

        return [self.steps.pop()]
    
        
