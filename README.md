# Visual Navigation in Real-World Indoor Environments Using End-to-End Deep Reinforcement Learning [Official]
This repository contains the official implementation of paper Visual Navigation in Real-World Indoor Environments Using End-to-End Deep Reinforcement Learning.

[Paper](https://arxiv.org/pdf/2010.10903.pdf)&nbsp;&nbsp;&nbsp;
[Web](https://jkulhanek.github.io/robot-visual-navigation/)

<br>

## Getting started
Before getting started, ensure, that you have Python 3.6+ ready. Start by cloning this repository with all submodules.
```bash
$ git clone --recurse-submodules https://github.com/jkulhanek/robot-visual-navigation.git
```

For training in the simulator environment, you have to install our fork of DeepMind Lab. Please follow the instructions in `./dmlab-vn/python/pip_package/README.md`.

Install the deeprl package.

```bash
pip install -e deep-rl-pytorch
```

Training scripts are in the `python` directory. To test if your environment is correctly configured, run `./test-train.py` and `./test-env.py`.

Start the training by running `./train.py <trainer>`, where `trainer` is the experiment you want to run. Available experiments are in the `trainer.py` file.
