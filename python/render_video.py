import gym
from argparse import ArgumentParser
from configuration import configuration
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import os
from deep_rl import make_agent
import deep_rl
import torch.multiprocessing as mp
import environment  # noqa: F401
from trainer import create_wrapped_environment


AGENT_ENV_MAP = {
    'dmhouse': dict(id="DMHouseCustom-v1"),
    'dmhouse-unreal': dict(id="DMHouseCustom-v1"),
    'dmhouse-a2c': dict(id="DMHouseCustom-v1"),
    'dmhouse-dqn': dict(id="DMHouseCustom-v1"),
    'dmhouse-ppo': dict(id="DMHouseCustom-v1"),
    'turtlebot': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
    'turtlebot-noprior': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
    'turtlebot-a2c': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
    'turtlebot-a2c-noprior': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
    'turtlebot-unreal': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
    'turtlebot-unreal-noprior': dict(id="TurtleLab-v0", has_end_action=True, augment_images=False),
}


class ExpandDimWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.unwrapped = agent.unwrapped if hasattr(agent, 'unwrapped') else agent

    def _transform_observation(self, observation):
        if isinstance(observation, (tuple)):
            return tuple((self._transform_observation(o) for o in observation))
        elif isinstance(observation, list):
            return list((self._transform_observation(o) for o in observation))
        elif isinstance(observation, dict):
            return {key: self._transform_observation(val) for key, val in observation.items()}
        else:
            return np.expand_dims(observation, 0)

    def act(self, observation):
        observation = self._transform_observation(observation)
        action = self.agent.act(observation)
        if not isinstance(action, (int, float)):
            action = action[0]
        return action

    def reset_state(self):
        self.agent.reset_state()


def wrap_last_env(env, last_wrapper):
    if hasattr(env, 'env') and env.env is not None:
        setattr(env, 'env', wrap_last_env(env.env, last_wrapper))
        return env
    else:
        return last_wrapper(env)


if __name__ == '__main__':
    deep_rl.configure(**configuration)
    videos_path = os.path.join(configuration.get('videos_path'), '{experiment}')

    # Set mp method to spawn
    # Fork does not play well with pytorch
    mp.set_start_method('spawn')
    parser = ArgumentParser()
    parser.add_argument('experiment')
    parser.add_argument('--path', default=videos_path)

    args = parser.parse_args()
    NAME = args.experiment
    PATH = args.path.format(experiment=NAME)

    deep_rl.configure(**configuration)
    os.makedirs(PATH, exist_ok=True)
    epcount = 100
    epmaxlen = 500
    frame_repeat = 5

    import trainer  # noqa: F401

    env = create_wrapped_environment(screen_size=(84, 84), **AGENT_ENV_MAP[NAME])
    agent = make_agent(NAME)
    observations = []

    class GrabImageWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
                gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)))

        @staticmethod
        def _resize_obs(obs):
            return (
                cv2.resize(obs[0], dsize=(84, 84), interpolation=cv2.INTER_NEAREST),
                cv2.resize(obs[1], dsize=(84, 84), interpolation=cv2.INTER_NEAREST),
            )

        def reset(self, *args, **kwargs):
            out = self.env.reset(*args, **kwargs)
            observations.append(out[:2])
            # return self._resize_obs(out)
            return out

        def step(self, *args, **kwargs):
            out = self.env.step(*args, **kwargs)
            observations.append(out[0][:2])
            # out = (self._resize_obs(out[0]),) + out[1:]
            return out

    env = wrap_last_env(env, GrabImageWrapper)
    env = agent.wrap_env(env)
    agent = ExpandDimWrapper(agent)
    if hasattr(agent.unwrapped, "set_environment"):
        agent.unwrapped.set_environment(env)

    # Run the episode
    o = env.reset()
    agent.reset_state()
    actions = []
    terminated = False
    max_env_steps = 1000
    reward = 0
    while not terminated and max_env_steps > 0:
        action = agent.act(o)
        actions.append(action)
        o, reward, terminated, _ = env.step(action)
        max_env_steps -= 1

    # Save video
    video_id = len(os.listdir(PATH)) + 1
    output_filename = "vid-%s.avi" % video_id
    height, width = (84, 84)
    writer = VideoWriter(os.path.join(PATH, output_filename),
                         VideoWriter_fourcc(*"XVID"), 30.0, (2 * width, height))
    for o1, o2 in observations:
        frame = np.concatenate([o1, o2], axis=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for _ in range(frame_repeat):
            writer.write(frame)
    writer.release()
