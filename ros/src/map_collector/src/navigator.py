import numpy as np
from controller import Controller
from wrappers import visualize
from collections import defaultdict
from math import pi

GRID_SIZE = 0.5
ROTATION_STEPS = 4

def move_position(position, rotation):
    if rotation == 0:
        return (position[0] + 1, position[1])
    elif rotation == 1:
        return (position[0], position[1] + 1)
    elif rotation == 2:
        return (position[0] - 1, position[1])
    elif rotation == 3:
        return (position[0], position[1] - 1)

class Navigator:
    def __init__(self, controller):
        self.controller = controller
    
    def _move(self, position):
        return self.controller.move_to((position[0] * GRID_SIZE, position[1] * GRID_SIZE))

    def _rotate(self, rotation):
        return self.controller.rotate_to(rotation * 2 * pi / 4)
    
    def _can_move(self):
        return not self.controller.is_occupied()

    def collect(self, observation, position, rotation):
        print("Collecting %s-%s" % (position, rotation))

    def explore(self):
        self.maze = defaultdict(lambda: 0)
        self.maze[(0,0)] = 1
        position = (0,0)
        rotation = 0
        self._explore(position, rotation)

    def _explore(self, position, rotation):
        self.maze[position] = 2
        collect_spots = []
        for i in range(4):
            if self.maze[move_position(position, rotation)] == 0:
                canMove = self._can_move()
                state = 1 if canMove else 3
                self.maze[move_position(position, rotation)] = state
                if canMove:
                    collect_spots.append((move_position(position, rotation), rotation))

            for r in range(3):                
                self.collect(self.controller.observe(), position, rotation + (float(r) / 3))
                self._rotate(rotation + (float(r + 1) / 3))

            if i != 3:
                rotation = (rotation + 1) % 4
            else:
                self._rotate(rotation)
            
        for i in range(4):
            if len(collect_spots) > 0:
                pos, rot = collect_spots.pop()
                if rot == rotation:
                    self._move(pos)
                    self._explore(pos, rot)
                    self._move(position)
                else:
                    collect_spots.append((pos, rot))

            if i != 3:
                rotation = (rotation - 1) % 4
                self._rotate(rotation)


