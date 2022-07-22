import numpy as np
import os
import math
from collections import defaultdict, OrderedDict

from ..core import AbstractTrainerWrapper, MetricContext
from .metrics import MetricWriter
from .util import DefaultOrderedDict
from .console_util import print_table


class SaveWrapper(AbstractTrainerWrapper):
    def __init__(self, *args, model_root_directory=None, saving_period=10000, **kwargs):
        super().__init__(*args, **kwargs)

        if model_root_directory is None:
            from ..configuration import configuration
            model_root_directory = configuration.get('models_path')

        self.model_root_directory = model_root_directory
        self._last_save = 0
        self.saving_period = saving_period
        self._enabled = True

    def process(self, **kwargs):
        res = self.trainer.process(**kwargs)
        (tdiff, _, _) = res
        self._last_save += tdiff

        if self._last_save >= self.saving_period:
            self._save()
            self._last_save = 0

        return res

    def _save(self):
        if not self._enabled:
            return

        print('Saving')
        path = os.path.join(self.model_root_directory, self.unwrapped.name)
        if not os.path.exists(path):
            os.makedirs(path)

        self.save(path)

    def run(self, *args, **kwargs):
        try:
            ret = super().run(*args, **kwargs)
        except KeyboardInterrupt:
            self._save()
            raise

        self._save()
        return ret

    def __repr__(self):
        return '<Save %s>' % repr(self.trainer)


class TimeLimitWrapper(AbstractTrainerWrapper):
    def __init__(self, max_time_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_t = 0
        self.max_time_steps = max_time_steps

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        self._global_t += tdiff
        if self._global_t >= self.max_time_steps:
            self.trainer.stop()
        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<TimeLimit(%s) %s>' % (self.max_time_steps, repr(self.trainer))


class EpisodeNumberLimitWrapper(AbstractTrainerWrapper):
    def __init__(self, max_number_of_episodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._number_of_episodes = 0
        self.max_number_of_episodes = max_number_of_episodes

    def process(self, **kwargs):
        tdiff, episode_end, stats = self.trainer.process(**kwargs)
        if episode_end is not None:
            self._number_of_episodes += 1
            if self._number_of_episodes >= self.max_number_of_episodes:
                self.trainer.stop()
        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<EpisodeNumberLimit(%s) %s>' % (self.max_number_of_episodes, repr(self.trainer))


class EpisodeLoggerWrapper(AbstractTrainerWrapper):
    def __init__(self, logging_period=10, validation_episodes=100, validation_period=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_t = 0
        self.logging_period = logging_period
        self.validation_period = validation_period
        self.validation_episodes = validation_episodes
        self.metric_collector = None
        self.validation_metric_context = None
        self.metric_writer = None

        self._global_t = 0
        self._episodes = 0
        self._data = []
        self._losses = []
        self._last_validation = 0
        self._extended_context = dict()

    def run(self, process, **kwargs):
        if self.metric_writer is None:
            self.metric_collector = MetricContext()
            self.validation_metric_context = MetricContext()
            self.metric_writer = MetricWriter(session_name=self.unwrapped.name)
            if hasattr(self.metric_writer, 'visdom'):
                self._extended_context['visdom'] = self.metric_writer.visdom

        def _late_process(*args, context=None, **kwargs):
            context.update(self._extended_context)
            data = process(*args, context=context, **kwargs)
            if self._log_t >= self.logging_period:
                self.metric_collector.summary(self._global_t)
                self.metric_collector.collect(
                    self.metric_writer, self._global_t, 'train')
                self._log_t = 0
            return data
        return super().run(_late_process, **kwargs)

    def _process_episode_end(self, episode_end, mode):
        if episode_end is None:
            return 0

        eps = 0
        collector = self.metric_collector if mode == 'train' else self.validation_metric_context
        if len(episode_end) == 3:
            eps, episode_lengths, rewards = episode_end
            for l, rw in zip(episode_lengths, rewards):
                collector.add_scalar('episode_length', l)
                collector.add_scalar('reward', rw)
        else:
            eps = 1
            episode_length, reward = episode_end
            collector.add_scalar('episode_length', episode_length)
            collector.add_scalar('reward', reward)
        if mode == 'train':
            self._episodes += eps
            self._log_t += eps
            self.metric_collector.add_last_value_scalar(
                'episodes', self._episodes)
        return eps

    def _process_stats(self, stats, mode):
        if stats is None:
            return

        collector = self.metric_collector if mode == 'train' else self.validation_metric_context
        if isinstance(stats, dict):
            if 'loss' in stats:
                collector.add_scalar('loss', stats.get('loss'))

            if 'win' in stats:
                collector.add_scalar('win_rate', float(stats.get('win')))
                collector.add_cummulative('win_count', int(stats.get('win')))

        else:
            stats.flush(collector)

    def run_validation(self, **kwargs):
        tval = 0
        while tval < self.validation_episodes:
            _, epend, stats = self.trainer.process(mode='validation', **kwargs)
            tval += self._process_episode_end(epend, 'validation')
            self._process_stats(stats, 'validation')

        print('Validation finished')
        self.validation_metric_context.summary(self._global_t)
        self.validation_metric_context.collect(
            self.metric_writer, self._global_t, 'validation')
        self._last_validation = self._episodes

    def process(self, mode='train', **kwargs):
        tdiff, episode_end, stats = self.trainer.process(mode=mode, **kwargs)
        self._global_t += tdiff if mode == 'train' else None
        epend = self._process_episode_end(episode_end, mode)
        self._process_stats(stats, mode)

        # Run validation step if time is right
        if self.validation_period is not None and \
            (self._episodes - self._last_validation) > self.validation_period and \
                epend > 0:

            # Run validation
            self.run_validation(**kwargs)

        return (tdiff, episode_end, stats)

    def __repr__(self):
        return '<EpisodeLogger %s>' % repr(self.trainer)

    def save(self, path):
        super().save(path)
        self.metric_writer.save(path)

    def log(self, stats):
        episode_length, reward = tuple(map(lambda *x: np.mean(x), *self._data))
        loss = np.mean(self._losses) if len(self._losses) > 0 else float('nan')
        self._data = []
        self._losses = []
        report = 'steps: {}, episodes: {}, reward: {:0.5f}, episode length: {}, loss: {:0.5f}'.format(
            self._global_t, self._episodes, reward, episode_length, loss)

        metrics_row = self.metric_writer \
            .record(self._global_t) \
            .scalar('reward', reward) \
            .scalar('episode_length', episode_length)

        if stats is not None and 'epsilon' in stats:
            report += ', epsilon:{:0.3f}'.format(stats.get('epsilon'))
            metrics_row = metrics_row.scalar('epsilon', stats.get('epsilon'))

        metrics_row.flush()

        if not math.isnan(loss):
            metrics_row = metrics_row.scalar('loss', loss)

        print(report)


def wrap(trainer, max_number_of_episodes=None, validation_period=None, validation_episodes=100, max_time_steps=None, episode_log_interval=None, save=True, saving_period=10000, **kwargs):
    if episode_log_interval is not None:
        trainer = EpisodeLoggerWrapper(episode_log_interval, trainer=trainer,
                                       validation_period=validation_period, validation_episodes=validation_episodes)

    if max_time_steps is not None:
        trainer = TimeLimitWrapper(max_time_steps, trainer=trainer)

    if max_number_of_episodes is not None:
        trainer = EpisodeNumberLimitWrapper(
            max_number_of_episodes, trainer=trainer)

    if save:
        trainer = SaveWrapper(trainer, saving_period=saving_period)

    return trainer
