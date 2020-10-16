#!/bin/env python
import argparse
from deep_rl import make_trainer
import deep_rl
import torch.multiprocessing as mp
from configuration import configuration

if __name__ == '__main__':
    # Set mp method to spawn
    # Fork does not play well with pytorch
    parser = argparse.ArgumentParser("test")
    parser.add_argument("trainer")
    args = parser.parse_args()

    mp.set_start_method('spawn')

    deep_rl.configure(**configuration)

    import trainer as package
    trainer = make_trainer(args.trainer)
    trainer.test(iterations=1000)
    print("Trainer ok")
