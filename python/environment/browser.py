import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class GoalKeyboardAgent:
    def __init__(self, env, actions=[0, 2, 3]):
        self.env = env
        self.actions = actions

    def show(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.o = self.env.reset()

        def redraw():
            a = self.o[0]
            b = self.o[1]
            ax1.imshow(a)
            ax2.imshow(b)
            fig.canvas.draw()

        def press(event):
            done = False
            if event.key == 's':
                mpimg.imsave("output.png", self.env.render(mode='rgbarray'))
            elif event.key == 'up':
                self.o, _, done, _ = self.env.step(self.actions[0])
                redraw()
            elif event.key == 'down':
                self.o, _, done, _ = self.env.step(self.actions[1])
                redraw()
            elif event.key == 'left':
                self.o, _, done, _ = self.env.step(self.actions[2])
                redraw()
            elif event.key == 'right':
                self.o, _, done, _ = self.env.step(self.actions[3])
                redraw()
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
