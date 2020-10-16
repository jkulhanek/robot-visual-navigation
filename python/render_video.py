import gym
from configuration import configuration
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import cv2
import os
from deep_rl import make_agent
import deep_rl
import torch.multiprocessing as mp
import environment

NAME = "dmhouse"


class RenderVideoWrapper(gym.Wrapper):
    def __init__(self, env, path, width=500, height=500, frame_repeat=1):
        super().__init__(env)
        self.path = path
        self.size = (height, width)
        self.frame_repeat = frame_repeat
        self.ep_states = []
        pass

    def reset(self):
        observation = self.env.reset()
        self.ep_states = [observation]
        return self._public_render(observation)

    def _public_render(self, obs):
        def transform_obs(x):
            x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_CUBIC)
            if len(x.shape) == 2:
                x = np.expand_dims(x, 2)
            return x
        return tuple((transform_obs(x) for x in obs))

    def step(self, action):
        obs, reward, done, stats = self.env.step(action)
        self.ep_states.append(obs)
        if done:
            self.render_video(self.ep_states)
            self.ep_states = []

        return self._public_render(obs), reward, done, stats

    def render_video(self, states):
        video_id = len(os.listdir(self.path)) + 1
        output_filename = "vid-%s.avi" % video_id
        height, width = self.size
        writer = VideoWriter(os.path.join(self.path, output_filename),
                             VideoWriter_fourcc(*"XVID"), 30.0, (2 * width, height))

        for state in self.ep_states:
            frame = np.concatenate([state[0], state[1]], axis=1)
            for _ in range(self.frame_repeat):
                writer.write(frame)
        writer.release()


if __name__ == '__main__':
    # Set mp method to spawn
    # Fork does not play well with pytorch
    mp.set_start_method('spawn')

    deep_rl.configure(**configuration)
    videos_path = os.path.join(configuration.get('videos_path'), NAME)
    os.makedirs(videos_path, exist_ok=True)
    epcount = 100
    epmaxlen = 500
    frame_repeat = 5

    import trainer
    default_args = trainer.default_args
    agent = make_agent(NAME)
    agent.model = agent.model.cpu()
    env = trainer.create_envs(1, dict(renderer="hardware", screen_size=(500, 500), **default_args()['env_kwargs']),
                              wrap=lambda env: RenderVideoWrapper(env, videos_path, frame_repeat=frame_repeat))[0]

    for i in range(epcount):
        print("Rendering episode %s" % (i + 1))
        obs = env.reset()
        agent.reset_state()
        done = False
        for _ in range(epmaxlen):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if reward != 0.0:
                print("collected reward: %s" % reward)
            if done:
                break

