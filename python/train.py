import argparse
from deep_rl import make_trainer, register_trainer
import deep_rl
from deep_rl.core import AbstractTrainerWrapper
from deep_rl.common.metrics import MetricHandlerBase
import deep_rl.configuration as config
from configuration import configuration


class WandbMetricHandler(MetricHandlerBase):
    def __init__(self, *args, **kwargs):
        super(WandbMetricHandler, self).__init__("wandb", *args, **kwargs)

        import wandb
        self._wandb = wandb
        wandb.init()

    def collect(self, collection, time, mode='train'):
        self._wandb.log(dict(collection), step=time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train")
    parser.add_argument("trainer")
    parser.add_argument("--no-wandb", dest="wandb", action="store_false", help="Use wandb for logging.")
    args = parser.parse_args()

    deep_rl.configure(**configuration)
    config.logging = {}
    config.logging.handlers = [
        "file",
        "matplotlib",
    ]
    if args.wandb:
        config.logging.handlers.append(WandbMetricHandler)

    import trainer as package
    trainer = make_trainer(args.trainer)
    print("starting %s" % args.trainer)
    trainer.run()
