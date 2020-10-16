from navigator import Navigator, GRID_SIZE, move_position
from wrappers import visualize
from math import pi
import unittest

class ProxyController:
    def __init__(self, test):
        maze = []
        with open("assets/maze.txt", 'r') as file:
            for line in file:
                line = line.rstrip()
                row = []
                for c in line:
                    if c == ' ':
                        row.append(1) # spaces are 1s
                    else:
                        row.append(0) # walls are 0s
                maze.append(row)
        self.maze = maze
        self.startPosition = (1,1,)
        self.position = (1,1)
        self.rotation = 0
        self.test = test

    def move_to(self, pos):
        (x,y) = pos
        x = int(x // GRID_SIZE) + self.startPosition[0]
        y = int(y // GRID_SIZE) + self.startPosition[1]
        pos = (x, y)
        self.test.assertEqual(self.maze[y][x], 1)
        self.position = pos

    def rotate_to(self, orientation):
        orientation = int(orientation * 4 / 2 / pi) 
        self.rotation = orientation

    def is_occupied(self):
        (x, y) = move_position(self.position, self.rotation)
        if x < 0 or y < 0 or y >= len(self.maze) or x >= len(self.maze[y]):
            return True
        return self.maze[y][x] == 0

class NavigatorTest(unittest.TestCase):
    def test(self):
        proxy = ProxyController(self)
        navigator = Navigator(proxy)
        navigator = visualize(navigator)
        navigator.explore()
        pass

if __name__ == "__main__":
    unittest.main()
