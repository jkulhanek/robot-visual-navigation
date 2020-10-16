from deep_rl import make_agent
import deep_rl
import torch.multiprocessing as mp
from configuration import configuration
import gym
import environment

if __name__ == '__main__':
    # Set mp method to spawn
    # Fork does not play well with pytorch
    mp.set_start_method('spawn')

    deep_rl.configure(**configuration)

    import trainer
    default_args = trainer.default_args
    agent = make_agent("dmhouse")
    env = trainer.create_envs(1, dict(renderer="software", **default_args()['env_kwargs']))[0]

    while True:
        obs = env.reset()
        agent.reset_state()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if reward != 0.0:
                print("collected reward: %s" % reward)

