from torch.multiprocessing import Queue, Value, Process
import torch.multiprocessing as mp
from threading import Thread
from functools import partial
from abc import abstractclassmethod
from .util import serialize_function
from time import sleep
from ..core import AbstractTrainer


def _run_process(children, report_queue, global_t, is_stopped, is_paused, env_fn):
    try:
        if not hasattr(children, 'create_env') or children.create_env is None:
            children.create_env = env_fn(children)

        if hasattr(children, 'initialize'):
            children.initialize()

        def _process(process, *args, **kwargs):
            result = process(*args, **kwargs)
            report_queue.put(result)
            return result

        def _run(process, **kwargs):
            children._global_t = 0
            children._is_stopped = False
            while not children._is_stopped:
                children._global_t = global_t.value
                tdiff, _, _ = process(mode='train', context=dict())
                children._global_t += tdiff

                # TODO: better pausing
                if is_paused.value:
                    report_queue.put('paused')

                while is_paused.value and not is_stopped.value:
                    sleep(0.005)

                children._global_t = global_t.value
                children._is_stopped = is_stopped.value

            return None

        return _run(process=partial(_process, process=children.process))
    except Exception as e:
        report_queue.put('error')
        raise e


class PausedToken:
    def __init__(self, is_paused, queue, num_processes):
        self.is_paused = is_paused
        self.queue = queue
        self.num_processes = num_processes

    def __enter__(self):
        self.is_paused.value = True
        paused_count = 0
        while paused_count != self.num_processes:
            r = self.queue.get()
            if r == 'paused':
                paused_count += 1

        return self

    def __exit__(self, *args, **kwargs):
        self.is_paused.value = False
        return None


class ProcessServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs=env_kwargs, model_kwargs=model_kwargs)
        self.name = name
        self.num_processes = 16

        self._report_queue = Queue(maxsize=16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)
        self._shared_is_paused = Value('i', False)
        self._validation_token = None

    @property
    def _global_t(self):
        return self._shared_global_t.value

    def _initialize(self, **model_kwargs):
        self.workers = [self.create_worker(i)
                        for i in range(self.num_processes)]
        env_fn = serialize_function(self._sub_create_env, self._env_kwargs)

        # Create processes
        self.processes = [mp.Process(target=_run_process, args=(
            worker,
            self._report_queue,
            self._shared_global_t,
            self._shared_is_stopped,
            self._shared_is_paused,
            env_fn)) for worker in self.workers]

        # self.validation_trainer = self.create_worker(-1)
        # self.validation_trainer.env = self.
        return None

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.processes:
            t.join()

    @abstractclassmethod
    def create_worker(self, id):
        pass

    def paused(self):
        return PausedToken(self._shared_is_paused, self._report_queue, self.num_processes)

    def process(self, mode='train', **kwargs):
        if mode == 'train':
            if self._validation_token is not None:
                self._validation_token.__exit__()
                self._validation_token = None

            value = self._report_queue.get()
            if value == 'error':
                raise Exception('Error in trainer')

            delta_t, epend, stats = value
        elif mode == 'validation':
            if self._validation_token is None:
                self._validation_token = self.paused().__enter__()

            return self.process_validation(**kwargs)

        return delta_t, epend, stats

    def process_validation(self, **kwargs):
        return self.validation_trainer.process(mode='validation', **kwargs)

    def run(self, process, **kwargs):
        # Initialize
        self._sub_create_env = self.create_env
        self.create_env = lambda *args, **env_kwargs: None
        super().run(process, **kwargs)
        self.create_env = self._sub_create_env
        del self._sub_create_env

        # Initialize globals
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False

        # Start created threads
        for t in self.processes:
            t.start()

        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode='train', context=dict())
            self._shared_global_t.value += tdiff

        for t in self.processes:
            t.join()

        self._finalize()
        return None


class ThreadServerTrainer(AbstractTrainer):
    def __init__(self, name, env_kwargs, model_kwargs, **kwargs):
        super().__init__(env_kwargs=env_kwargs, model_kwargs=model_kwargs)
        self.name = name
        self.num_processes = 16

        self._report_queue = Queue(maxsize=16)
        self._shared_global_t = Value('i', 0)
        self._shared_is_stopped = Value('i', False)

    @property
    def _global_t(self):
        return self._shared_global_t.value

    def _initialize(self, **model_kwargs):
        self.workers = [self.create_worker(i)
                        for i in range(self.num_processes)]
        env_fn = serialize_function(self._sub_create_env, self._env_kwargs)

        # Create processes
        self.processes = [Thread(target=_run_process, args=(
            worker,
            self._report_queue,
            self._shared_global_t,
            self._shared_is_stopped,
            self._shared_is_paused,
            env_fn)) for worker in self.workers]

        return None

    def stop(self):
        self._shared_is_stopped.value = True
        for t in self.processes:
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
        self.create_env = lambda *args, **env_kwargs: None
        super().run(process, **kwargs)
        self.create_env = self._sub_create_env
        del self._sub_create_env

        # Initialize globals
        self._shared_global_t.value = 0
        self._shared_is_stopped.value = False

        # Start created threads
        for t in self.processes:
            t.start()

        while not self._shared_is_stopped.value:
            tdiff, _, _ = process(mode='train', context=dict())
            self._shared_global_t.value += tdiff

        for t in self.processes:
            t.join()

        self._finalize()
        return None
