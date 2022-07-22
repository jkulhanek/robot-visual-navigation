from trainer import create_wrapped_environment
from deep_rl import make_agent
import argparse
import csv
import math
import deep_rl
import gym
import numpy as np
from configuration import configuration
from deep_rl.common.console_util import print_progress

AGENT_ENV_MAP = {
    'turtlebot': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'turtlebot-noprior': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'turtlebot-a2c': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'turtlebot-a2c-noprior': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'turtlebot-unreal': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'turtlebot-unreal-noprior': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'shortest-path': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'random-end': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
    'random': dict(id = "TurtleLab-v0", has_end_action = True, augment_images = False),
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
        action = self.agent.act(observation)[0]
        return action

    def reset_state(self):
        self.agent.reset_state()


def dist(position1, position2):
    return math.sqrt((position1[0] - position2[0]) ** 2 + (position1[1] - position2[1]) ** 2)


class Evaluator:
    def __init__(self, agent, env, num_episodes):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes

    @staticmethod
    def load_tasks(filename):
        tasks = []
        with open(filename, "r") as f:
            for line in f:
                if len(line) > 0:
                    split = line.rstrip().split(" ")
                    x = round(float(split[0]) / 0.2 + 2)
                    y = round(float(split[1]) / 0.2 + 3)
                    r = round(float(split[2]) / 1.57 % 4)
                    tasks.append(((x, y, r), tuple(map(int, split[3:]))))
        return tasks

    def _run_episode(self, task=None):
        if task is None:
            o = self.env.reset()
        else:
            self.env.unwrapped.set_next_task(*task)
            o = self.env.reset()

        self.agent.reset_state()
        positions = [self.env.position[0]]
        actions = []
        terminated = False
        max_env_steps = 1000
        reward = 0
        while not terminated and max_env_steps > 0:
            action = self.agent.act(o)
            actions.append(action)
            positions.append(self.env.position[0])
            o, reward, terminated, _ = self.env.step(action)
            max_env_steps -= 1
            

        goal_reached = reward > 0
        goal_distance = dist(positions[-1], self.env.position[1])
        return (len(actions), goal_reached, goal_distance, actions, positions)

    @staticmethod
    def write_turtleimage_results(tasks, filename, results):
        with open(filename, "w+") as f:
            writer = csv.writer(f)            
            writer.writerow(["id", "start_x", "start_y", "start_r", "goal_x", "goal_y", "goal_r", "steps", "goal_reached", "goal_distance", "actions", "positions"])
            for i, ((start, end), r) in enumerate(zip(tasks,results)):
                writer.writerow((i,) + tuple(start) + tuple(end) + tuple(map(str, r)))

    @staticmethod
    def write_turtleimage_statistics(filename, results):
        with open(filename, "w+") as f:
            keys = ["goal_distance_mean", "goal_distance_std", "goal_reached_mean", "goal_reached_std", "steps_mean", "steps_std", "actions"]
            writer = csv.DictWriter(f, keys)
            writer.writerow({ k: k for k in keys})
            writer.writerow(results)


    def evaluate_agent(self, tasks, output_filename = None, agent_name = None):
        if output_filename is None:
            output_filename = "results-%s.csv" % agent_name
        
        results = []
        for i,task in enumerate(tasks):
            print_progress(i, len(tasks))
            results.append(self._run_episode(task))
            
        print_progress(len(tasks), len(tasks))
        Evaluator.write_turtleimage_results(tasks, output_filename, results)

    def evaluate_agent_statistics(self, output_filename = None, agent_name = None):
        goal_distances = []
        goal_reached = []
        actions = []
        steps = []
        for i in range(self.num_episodes):
            print_progress(i, self.num_episodes)
            s, g, d, action_arr, _ = self._run_episode()
            if g:
                steps.append(s)
            goal_reached.append(g)
            goal_distances.append(d)
            if len(action_arr) > 1:
                actions.append([len([x for x in action_arr if x == i]) / (len(action_arr) - 1.0) for i in range(4)])
        print_progress(self.num_episodes, self.num_episodes)

        result = dict(
            goal_distance_mean = np.mean(goal_distances),
            goal_distance_std = np.std(goal_distances),
            goal_reached_mean = np.mean(goal_reached),
            goal_reached_std = np.std(goal_reached),
            steps_mean = np.mean(steps),
            steps_std = np.std(steps),
            actions = tuple(np.array(actions).mean(0))
        )

        print('success rate: {:.4f}%'.format(result["goal_reached_mean"] * 100))
        print('avg. episode steps: {:.4f}'.format(result["steps_mean"]))
        print('avg. goal distance: {:.4f}'.format(result["goal_distance_mean"]))

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
    parser.add_argument("--locations", type=str, default=None)
    parser.add_argument("--output", type = str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    args = parser.parse_args()

    env = create_wrapped_environment(**AGENT_ENV_MAP[args.model])
    agent = ExpandDimWrapper(make_agent(args.model))
    if hasattr(agent.unwrapped, "set_environment"):
        agent.unwrapped.set_environment(env)

    evaluator = Evaluator(agent, env, args.num_episodes)
    if args.locations is not None:
        tasks = Evaluator.load_tasks(args.locations)
        evaluator.evaluate_agent(tasks, args.output, agent_name = args.model)
    else:
        evaluator.evaluate_agent_statistics(args.output, agent_name=args.model)
    
