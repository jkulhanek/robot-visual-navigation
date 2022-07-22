#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import environment
import gym

import deep_rl
from configuration import configuration
deep_rl.configure(**configuration)


class GoalKeyboardAgent:
    def __init__(self, env, actions=[0, 1, 2, 3]):
        self.env = env
        self.actions = actions

    def show(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.o = self.env.reset()

        def redraw():
            a = self.o[0][:, :, :3]
            b = self.o[1]

            ax1.imshow(a)
            ax2.imshow(b)
            fig.canvas.draw()

        def press(event):
            done = False
            reward = 0.0
            if event.key == 's':
                mpimg.imsave("output.png", self.env.render(mode='rgbarray'))
            elif event.key == 'up':
                self.o, reward, done, _ = self.env.step(self.actions[0])
                redraw()
            elif event.key == 'right':
                self.o, reward, done, _ = self.env.step(self.actions[1])
                redraw()
            elif event.key == 'down':
                self.o, reward, done, _ = self.env.step(self.actions[2])
                redraw()
            elif event.key == 'left':
                self.o, reward, done, _ = self.env.step(self.actions[3])
                redraw()
            elif event.key == ' ':
                print('Collecting reward')
                self.o, reward, done, _ = self.env.step(self.actions[4])
                redraw()

            elif event.key == 'r':
                self.o = self.env.reset()
                redraw()

            if reward != 0.0:
                print('Reward collected: {reward}'.format(reward=reward))

            if hasattr(self.env.unwrapped, 'state'):
                print(self.env.unwrapped.state)

            if done:
                print('Goal reached')
                self.o = self.env.reset()
                redraw()

        plt.rcParams['keymap.save'] = ''
        fig.canvas.mpl_connect('key_press_event', press)
        plt.axis('off')
        plt.show()

        redraw()


if __name__ == '__main__':
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(
        description='Deep reactive agent scene explorer.')
    args = vars(parser.parse_args())

    env = gym.make('DeepmindLabCustomHouse-v0', screen_size=(500, 500))
    GoalKeyboardAgent(env, actions=[0, 3, 1, 2, 4]).show()
