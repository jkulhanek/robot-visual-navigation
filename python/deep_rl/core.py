import random
import abc
from abc import abstractclassmethod
import gym
import threading
import os
from collections import defaultdict
import numpy as np

from multiprocessing import Queue, Value
from threading import Thread
from functools import partial


def _default_cum_value():
    return 0


class Schedule:
    def __init__(self):
        pass

    def __call__(self):
        return None

    def step(self, time):
        self.time = time


class MetricContext:
    def __init__(self):
        self.accumulatives = defaultdict(list)
        self.lastvalues = dict()
        self.cummulatives = defaultdict(_default_cum_value)
        self.window_size = 100

    def add_last_value_scalar(self, name, value):
        self.lastvalues[name] = value

    def add_scalar(self, name, value):
        self.accumulatives[name].append(value)

    def add_cummulative(self, name, value):
        self.cummulatives[name] += value

    def _format_number(self, number):
        if isinstance(number, int):
            return str(number)

        return '{:.3f}'.format(number)

    def summary(self, global_t):
        from .common.console_util import print_table
        values = []
        values.extend((key, value) for key, value in self.lastvalues.items())
        values.extend((key, np.mean(x[-self.window_size:]))
                      for key, x in self.accumulatives.items())
        values.extend((key, x) for key, x in self.cummulatives.items())
        values.sort(key=lambda x: x[0])
        print_table([('step', global_t)] + values)

    def flush(self, other):
        for key, val in self.lastvalues.items():
            other.lastvalues[key] = val

        for key, val in self.cummulatives.items():
            other.cummulatives[key] += val

        for key, val in self.accumulatives.items():
            other.accumulatives[key].extend(val)

    def collect(self, writer, global_t, mode='train'):
        if mode == 'train':
            metrics_row = writer.record(global_t)
        elif mode == 'validation':
            metrics_row = writer.record_validation(global_t)

        for (key, val) in self.accumulatives.items():
            metrics_row = metrics_row.scalar(
                key, np.mean(val[-self.window_size:]))

        for (key, value) in self.lastvalues.items():
            metrics_row = metrics_row.scalar(key, value)

        metrics_row = metrics_row.flush()
        self.lastvalues = dict()
        return metrics_row


class AbstractAgent:
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    @abc.abstractclassmethod
    def act(self, state):
        pass

    def wrap_env(self, env):
        return env

    def reset_state(self):
        pass


class RandomAgent(AbstractAgent):
    def __init__(self, action_space_size, seed=None):
        super().__init__('random')
        self._action_space_size = action_space_size
        self._random = random.Random(x=seed)

    def act(self, state):
        return self._random.randrange(0, self._action_space_size)


class LambdaAgent(AbstractAgent):
    def __init__(self, name, act_fn, **kwargs):
        super().__init__(name)
        self.act = lambda state: act_fn(state, **kwargs)


class AbstractTrainer:
    def __init__(self, env_kwargs, model_kwargs, *args, **kwargs):
        self.schedules = dict()
        super().__init__(*args, **kwargs)
        self.env = None
        self._env_kwargs = env_kwargs
        self.model = None
        self._model_kwargs = model_kwargs
        self.name = 'trainer'

        self.is_initialized = False

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self.schedules:
            self.schedules.pop(name)

        if isinstance(value, Schedule):
            self.schedules[name] = value

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if isinstance(value, Schedule):
            if not hasattr(self, '_global_t'):
                raise Exception(
                    'Schedules are supported only for classes with _global_t property')
            value.step(getattr(self, '_global_t'))
            return value()
        else:
            return value

    def __delattr__(self, name):
        super().__delattr__(name)
        if name in self.schedules:
            self.schedules.pop(name)

    def save(self, path):
        pass

    def create_env(self, env):
        if isinstance(env, dict):
            env = gym.make(**env)

        return env

    @abc.abstractclassmethod
    def _initialize(self, **model_kwargs):
        pass

    def _finalize(self):
        pass

    @abc.abstractclassmethod
    def process(self, **kwargs):
        pass

    def run(self, process, **kwargs):
        if process is None:
            raise Exception('Must be compiled before run')

        self.env = self.create_env(self._env_kwargs)
        self.model = self._initialize(**self._model_kwargs)
        return None

    def __repr__(self):
        return '<%sTrainer>' % self.name

    def compile(self, compiled_agent=None, **kwargs):
        if compiled_agent is None:
            compiled_agent = CompiledTrainer(self)

        return compiled_agent


class AbstractTrainerWrapper(AbstractTrainer):
    def __init__(self, trainer, *args, **kwargs):
        super().__init__(None, None)
        self.trainer = trainer
        self.unwrapped = trainer.unwrapped if hasattr(
            trainer, 'unwrapped') else trainer
        self.summary_writer = trainer.summary_writer if hasattr(
            trainer, 'summary_writer') else None

    def process(self, **kwargs):
        return self.trainer.process(**kwargs)

    def stop(self, **kwargs):
        self.trainer.stop(**kwargs)

    def run(self, process, **kwargs):
        return self.trainer.run(process, **kwargs)

    def save(self, path):
        self.trainer.save(path)


class CompiledTrainer(AbstractTrainerWrapper):
    def __init__(self, target, *args, **kwargs):
        super().__init__(target, *args, **kwargs)
        self.process = target.process

    def run(self, **kwargs):
        return self.trainer.run(self.process)

    def test(self, *args, **kwargs):
        from .common.tester import test_trainer
        test_trainer(self, *args, **kwargs)

    def __repr__(self):
        return '<Compiled %s>' % self.trainer.__repr__()


class SingleTrainer(AbstractTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_t = None
        pass

    def run(self, process, **kwargs):
        self._global_t = 0
        self._is_stopped = False

        super().run(process, **kwargs)

        while not self._is_stopped:
            tdiff, _, _ = process(mode='train', context=dict())
            self._global_t += tdiff

        self._finalize()
        return None

    def stop(self):
        self._is_stopped = True


class ThreadServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs=env_kwargs, model_kwargs=model_kwargs, **kwargs)
        self.name = name
        self._report_queue = Queue(maxsize=16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)
        self._num_workers = 16

    @property
    def _global_t(self):
        return self._shared_global_t.value

    def _child_run(self, id):
        worker = self.create_worker(id)

        def _process(process, *args, **kwargs):
            result = process(*args, **kwargs)
            self._report_queue.put(result)
            worker._shared_global_t = self._shared_global_t.value
            worker._shared_is_stopped = self._shared_is_stopped.value
            return result

        worker.run(process=partial(_process, process=worker.process))

    def _initialize(self, **model_kwargs):
        self.workers = [Thread(target=self._child_run, args=(i,))
                        for i in range(self._num_workers)]

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.workers:
            t.join()

    @abstractclassmethod
    def create_worker(self, id):
        pass

    def process(self, mode='train', **kwargs):
        assert mode == 'train'
        delta_t, epend, stats = self._report_queue.get()
        return delta_t, epend, stats

    def run(self, process, **kwargs):
        # Initialize
        self._sub_create_env = self.create_env
        self.create_env = lambda **env_kwargs: None
        super().run(process, **kwargs)
        self.create_env = self._sub_create_env
        del self._sub_create_env

        # Initialize globals
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False

        # Start created threads
        for t in self.workers:
            t.start()

        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode='train', context=dict())
            self._shared_global_t.value += tdiff

        self._finalize()
        return None
