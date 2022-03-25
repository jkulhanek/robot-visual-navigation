# Visual Navigation in Real-World Indoor Environments Using End-to-End Deep Reinforcement Learning
This repository contains the official implementation of paper Visual Navigation in Real-World Indoor Environments Using End-to-End Deep Reinforcement Learning.

[Paper](https://arxiv.org/pdf/2010.10903.pdf)&nbsp;&nbsp;&nbsp;
[Web](https://jkulhanek.github.io/robot-visual-navigation/)&nbsp;&nbsp;&nbsp;
[Demo](https://colab.research.google.com/github/jkulhanek/robot-visual-navigation/blob/master/notebooks/robot-visual-navigation-playground.ipynb)
 
<br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg?style=for-the-badge)](https://colab.research.google.com/github/jkulhanek/robot-visual-navigation/blob/master/notebooks/robot-visual-navigation-playground.ipynb)
![Python Versions](https://img.shields.io/badge/Python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)

<br>

## Getting started
Before getting started, ensure, that you have Python 3.6+ ready.
We recommend activating a new virtual environment for the repository:
```bash
python -m venv robot-visual-navigation-env
source robot-visual-navigation-env/bin/activate
```

Start by cloning this repository and installing the dependencies:
```bash
git clone https://github.com/jkulhanek/robot-visual-navigation.git
cd robot-visual-navigation
pip install -r requirements.txt
cd python
```

For DMHouse package, we recommend starting with Ubuntu 18+ and installing dependencies as follows:
```bash
apt-get install libsdl2-dev libosmesa6-dev gettext g++ unzip zip curl gnupg libstdc++6
```


## Downloading the trained models and datasets
You can download the pre-trained models from:
[https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/dmhouse-models.tar.gz](https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/dmhouse-models.tar.gz)
[https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/turtlebot-models.tar.gz](https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/turtlebot-models.tar.gz)


Download the pre-trained models using the following commands:
```bash
mkdir -p ~/.cache/robot-visual-navigation/models

# Download DMHouse models
curl -L https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/dmhouse-models.tar.gz | tar -xz -C ~/.cache/robot-visual-navigation/models

# Download real-world dataset models
curl -L https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/checkpoints/turtlebot-models.tar.gz | tar -xz -C ~/.cache/robot-visual-navigation/models

# Download real-world dataset
mkdir -p ~/.cache/robot-visual-navigation/datasets
curl -L -o ~/.cache/robot-visual-navigation/datasets/turtle_room_compiled.hdf5 https://data.ciirc.cvut.cz/public/projects/2021RealWorldNavigation/datasets/turtle_room_compiled.hdf5
```

## Evaluation
Run the evaluation on the DMHouse simulator to verify that everything is working correctly:
```bash
python evaluate_dmhouse.py dmhouse --num-episodes 100
```

Similarly for the real-world dataset:
```bash
python evaluate_turtlebot.py turtlebot --num-episodes 100
```

## Training
Start the training by running `./train.py <trainer>`, where `trainer` is the experiment you want to run. Available experiments are in the `trainer.py` file.
