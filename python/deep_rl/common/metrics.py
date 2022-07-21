import os
from abc import abstractclassmethod
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from ..configuration import configuration


class MetricHandlerBase:
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @abstractclassmethod
    def collect(self, collection, time, mode='train'):
        pass


PLOT_METRICS = ['reward', 'episode_length']


class MatplotlibHandler(MetricHandlerBase):
    def __init__(self, interactive=False, *args, **kwargs):
        super().__init__('matplotlib', *args, **kwargs)
        self._metrics = defaultdict(lambda: ([], []))
        self._validation_metrics = defaultdict(lambda: ([], []))
        self._figures = dict()
        self.interactive = interactive

    def collect(self, collection, time, mode='train'):
        store = self._metrics if mode == 'train' else self._validation_metrics
        for (tag, val) in collection:
            t, v = store[tag]
            t.append(time)
            v.append(val)
        self.plot()

    def _get_figure(self, name):
        if name in self._figures:
            fig = self._figures[name]
            plt.figure(num=fig.number, clear=True)
        else:
            fig = plt.figure()
            fig.canvas.set_window_title(name)
            self._figures[name] = fig
        return fig

    def _plot(self):
        for name, metric in self._metrics.items():
            if name in PLOT_METRICS:
                fig = self._get_figure(name)
                plt.plot(metric[0], metric[1], 'b')

                if name in self._validation_metrics:
                    metric = self._validation_metrics[name]
                    plt.plot(metric[0], metric[1], 'r')

                fig.canvas.flush_events()

    def plot(self):
        if self.interactive:
            self._plot()

    def save(self, path):
        self._plot()
        for name, fig in self._figures.items():
            plt.figure(num=fig.number, clear=False)
            plt.savefig(os.path.join(path, '%s.pdf' % name), format='pdf')
            plt.savefig(os.path.join(path, '%s.eps' % name), format='eps')


class VisdomHandler(MetricHandlerBase):
    def __init__(self, visdom, *args, **kwargs):
        super().__init__('visdom', *args, **kwargs)
        self.visdom = visdom
        self._metrics = defaultdict(lambda: ([], []))
        self._validation_metrics = defaultdict(lambda: ([], []))
        self._figures = dict()

    def collect(self, collection, time, mode='train'):
        store = self._metrics if mode == 'train' else self._validation_metrics
        for (tag, val) in collection:
            t, v = store[tag]
            t.append(time)
            v.append(val)
        self.plot()

    def _update_figure(self, name, update):
        if name in self._figures:
            update.update(dict(
                win=self._figures[name],
                update='append',
                opts=dict(
                    title=name,
                    **update.get('opts', dict())
                )
            ))

        self._figures[name] = self.visdom.line(**update)

    def plot(self):
        for name, metric in self._metrics.items():
            if name in PLOT_METRICS:
                if len(metric[0]) > 1:
                    self._update_figure(name, dict(
                        name='train',
                        X=np.array(metric[0]),
                        Y=np.array(metric[1])
                    ))

                    metric[0].clear()
                    metric[1].clear()

                if name in self._validation_metrics:
                    metric = self._validation_metrics[name]
                    if len(metric[0]) > 1:
                        self._update_figure(name, dict(
                            name='validation',
                            X=np.array(metric[0]),
                            Y=np.array(metric[1])
                        ))

                        metric[0].clear()
                        metric[1].clear()


class DataHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        super().__init__('data', *args, **kwargs)
        self._metrics = defaultdict(lambda: ([], []))
        self._was_initialized = False

    def collect(self, collection, time, mode='train'):
        for (tag, val) in collection:
            if mode != 'train':
                tag = mode + '_' + tag

            t, v = self._metrics[tag]
            t.append(time)
            v.append(val)

    def save(self, path):
        import csv
        with open(os.path.join(path, 'metrics.txt'), 'a' if self._was_initialized else 'w+') as f:
            writer = csv.writer(f)
            for key, vals in self._metrics.items():
                writer.writerow([key, str(len(vals[0]))] + vals[0] + vals[1])

            f.flush()
        self._was_initialized = True
        self._metrics = defaultdict(lambda: ([], []))


class ConsoleHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        super().__init__('data', *args, **kwargs)
        self._was_initialized = False

    def collect(self, collection, time, mode='train'):
        for (tag, val) in collection:
            if mode != 'train':
                tag = mode + '_' + tag

            print("=============================================")
            print("time: %s" % time)
            print("\n".join(["%s: %s" % (tag, val)
                  for (tag, val) in collection]))
            print("=============================================")


def create_visdom(session_name, configuration):
    if configuration is None or configuration.server is None:
        return None

    from visdom import Visdom
    return Visdom(env=session_name, **configuration.as_dict())


def load_metrics(file):
    import csv

    metrics = defaultdict(lambda: ([], []))
    for line in csv.reader(file):
        name = line[0]
        count = int(line[1])
        times = list(map(int, line[2:(2 + count)]))
        values = list(map(float, line[(2 + count):]))
        metrics[name][0].extend(times)
        metrics[name][1].extend(values)
    return metrics


ALIASES = {
    "matplotlib": "deep_rl.common.metrics;MatplotlibHandler",
    "file": "deep_rl.common.metrics;DataHandler",
}


class MetricWriter:
    class _MetricRecordFactory:
        def __init__(self, time, flush, mode):
            self._flush = flush
            self.collection = []
            self._time = time
            self.mode = mode

        def scalar(self, name, value):
            self.collection.append((name, value))
            return self

        def flush(self):
            self._flush(self.collection, self._time, self.mode)
            self.collection.clear()
            return self

    def __init__(self, use_tensorboard=False, session_name='main', logdir='./logs'):
        self._use_tensorboard = use_tensorboard
        self._logdir = logdir

        handlers = configuration.get("logging.handlers")
        if handlers is None:
            self.visdom = create_visdom(session_name, configuration.visdom)
            if self.visdom is not None:
                self.handlers = [VisdomHandler(
                    self.visdom), MatplotlibHandler(interactive=False)]
            else:
                self.handlers = [MatplotlibHandler()]
            self.handlers.append(DataHandler())
        else:
            self.handlers = []
            import importlib
            for handler in handlers:
                if isinstance(handler, str):
                    if handler in ALIASES:
                        handler = ALIASES[handler]
                    handler = handler.split(";")
                    pkg = importlib.import_module(handler[0])
                    handler = getattr(pkg, handler[1])()
                elif callable(handler):
                    handler = handler()

                self.handlers.append(handler)

        if logdir is not None and len(logdir) > 0:
            if not os.path.exists(logdir):
                os.mkdir(logdir)

        if use_tensorboard:
            import tensorflow as tf
            assert logdir is not None and len(logdir) > 0
            self._tensorboard_writer = tf.summary.FileWriter(logdir)

    def record(self, time):
        return MetricWriter._MetricRecordFactory(time, self._flush, mode='train')

    def record_validation(self, time):
        return MetricWriter._MetricRecordFactory(time, self._flush, mode='validation')

    def _flush(self, collection, time, mode):
        if self._use_tensorboard and mode == 'train':
            import tensorflow as tf
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                         simple_value=value) for (tag, value) in collection])
            self._tensorboard_writer.add_summary(summary, time)
            self._tensorboard_writer.flush()

        for handler in self.handlers:
            handler.collect(collection, time, mode=mode)

    def save(self, path):
        for handler in self.handlers:
            if hasattr(handler, 'save'):
                handler.save(path)
