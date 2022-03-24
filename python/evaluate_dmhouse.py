from trainer import create_wrapped_environment
from deep_rl import make_agent
import argparse
import csv
import deep_rl
from collections import OrderedDict
import numpy as np
from configuration import configuration
from deep_rl.common.console_util import print_progress

AGENT_ENV_MAP = {
    'dmhouse': dict(id = "DMHouseCustom-v1"),
    'dmhouse-unreal': dict(id = "DMHouseCustom-v1"),
    'dmhouse-a2c': dict(id = "DMHouseCustom-v1"),
    'dmhouse-dqn': dict(id = "DMHouseCustom-v1"),
    'dmhouse-ppo': dict(id = "DMHouseCustom-v1"),
    'random': dict(id = "DMHouseCustom-v1"),
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

class Evaluator:
    def __init__(self, agent, env, num_episodes=1000):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes

    def _run_episode(self):
        o = self.env.reset()
        self.agent.reset_state()
        actions = []
        terminated = False
        max_env_steps = 1000
        reward = 0
        while not terminated and max_env_steps > 0:
            action = self.agent.act(o)
            actions.append(action)
            o, reward, terminated, _ = self.env.step(action)
            max_env_steps -= 1
            

        goal_reached = reward > 0
        return (len(actions), goal_reached, actions,) + self.env.unwrapped.metrics()

    @staticmethod
    def write_turtleimage_statistics(filename, results):
        with open(filename, "w+") as f:
            keys = ["goal_reached_mean", "goal_reached_std", "steps_mean", "steps_std", "distance_mean", "distance_std", "shortest_distance_mean", "shortest_distance_std", "actions"]
            writer = csv.DictWriter(f, keys)
            writer.writerow({ k: k for k in keys})
            writer.writerow(results)

    def evaluate_agent_statistics(self, output_filename = None, agent_name = None):
        goal_reached = []
        actions = []
        steps = []
        shortest_distances = []
        distances = []
        for i in range(self.num_episodes):
            print_progress(i, self.num_episodes)
            s, g, action_arr, gd, sd = self._run_episode()
            if g:
                steps.append(s)
            goal_reached.append(g)
            distances.append(gd)
            shortest_distances.append(sd)
            if len(action_arr) > 1:
                actions.append([len([x for x in action_arr if x == i]) / (len(action_arr) - 1.0) for i in range(4)])

        print_progress(self.num_episodes, self.num_episodes)

        result = OrderedDict(
            goal_reached_mean = np.mean(goal_reached),
            goal_reached_std = np.std(goal_reached),
            steps_mean = np.mean(steps),
            steps_std = np.std(steps),
            distance_mean = np.mean(distances),
            distance_std = np.std(distances),
            shortest_distance_mean = np.mean(shortest_distances),
            shortest_distance_std = np.std(shortest_distances),
            actions = tuple(np.array(actions).mean(0))
        )

        print('success rate: {:.4f}%'.format(result["goal_reached_mean"] * 100))
        print('avg. episode steps: {:.4f}'.format(result["steps_mean"]))
        print('avg. distance travelled: {:.4f}'.format(result["distance_mean"]))

        if output_filename is not None:
            Evaluator.write_turtleimage_statistics(output_filename, result)

if __name__ == "__main__":
    deep_rl.configure(**configuration)
    import trainer
    import testing_agents
    _ = trainer
    _ = testing_agents

    parser = argparse.ArgumentParser("evaluate")
    parser.add_argument("model", type=str)
    parser.add_argument("--output", type = str, default=None)
    parser.add_argument('--num-episodes', type=int, default=1000)
    args = parser.parse_args()

    env = create_wrapped_environment(**AGENT_ENV_MAP[args.model])
    agent = make_agent(args.model)
    env = agent.wrap_env(env)
    agent = ExpandDimWrapper(agent)
    if hasattr(agent.unwrapped, "set_environment"):
        agent.unwrapped.set_environment(env)

    evaluator = Evaluator(agent, env, num_episodes=args.num_episodes)
    evaluator.evaluate_agent_statistics(args.output, agent_name = args.model)
    
