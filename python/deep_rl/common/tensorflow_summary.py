from collections import namedtuple
import tensorflow as tf
import os


def create_row(time, data, metricname):
    return namedtuple('DataRow', ('monitor'))(namedtuple('DataRowMonitor',  ('l', 'r'))(time, data[metricname]))


def extract_tensorflow_summary(path, metricname='score'):
    SUMMARY_NAMES = [metricname]
    time_steps = []
    metrics = {x: [] for x in SUMMARY_NAMES}

    log_files = (os.path.join(path, x) for x in os.listdir(path))
    for filename in log_files:
        for e in tf.train.summary_iterator(filename):
            if set(SUMMARY_NAMES).intersection((v.tag for v in e.summary.value)):
                time_steps.append(e.step)
                for v in e.summary.value:
                    if v.tag in SUMMARY_NAMES:
                        metrics[v.tag].append(v.simple_value)

    return create_row(time_steps, metrics, metricname)
