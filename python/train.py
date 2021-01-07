#!/bin/env python
# import stacktracer
# stacktracer.trace_start("/home/kulhajon/repos/robot-visual-navigation/trace.html")
# print("starting trace")

import argparse
import deep_rl
from deep_rl import make_trainer
from deep_rl.common.metrics import MetricHandlerBase
import deep_rl.configuration as config
from configuration import configuration
from utils import bind_arguments, add_arguments
import wandb


class WandbMetricHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        self.run = None
        super(WandbMetricHandler, self).__init__("wandb", *args, **kwargs)

    def collect(self, collection, time, mode='train'):
        if self.run is None:
            self.run = wandb.init(project='robot-visual-navigation')
        self.run.log(dict(collection), step=time)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train", add_help=False)
    parser.add_argument("trainer")
    args, _ = parser.parse_known_args()

    deep_rl.configure(**configuration)
    config.logging = {}
    config.logging.handlers = [
        "file",
        "matplotlib",
        WandbMetricHandler
    ]

    import trainer as _  # noqa:F401
    trainer = make_trainer(args.trainer)
    parser = add_arguments(argparse.ArgumentParser(parents=[parser]), trainer.unwrapped.__class__)
    args = parser.parse_args()
    trainer_kwargs = dict(args.__dict__)
    del trainer_kwargs['trainer']
    trainer = make_trainer(args.trainer, **trainer_kwargs)
    print("starting %s" % args.trainer)
    trainer.run()
