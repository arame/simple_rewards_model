import numpy as np
import math
import random


class Dungeon:
    
    def __init__(self, N):
        self.N = N
        self.number_of_items = math.floor(self.N/2)
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.LAVA = 2
        self.EXIT = 3
        self.AGENT = 99
        self.rewards = [-1, -5, -20, N**2]

        # Numpy array that holds the information about the environment
        self._initialise_environment()
        
        # position of the agent and exit will be decided by resetting the environment.
        self.position_agent = None
        self.position_exit = None
        
        # run time
        self.time_elapsed = 0
        self.time_limit = N**2
        
    def step(self, action):
        
        # action is 'up', 'down', 'left', or 'right'
        # modify the position of the agent
        # Store the current position of the agent in self.new_position_agent
        # If the agent cannot move because of an obstacle, then it does not move.
        # Otherwise the agent is moved to the new position
        self.new_position_agent = self.position_agent
        curr_x, curr_y = self.position_agent
        if action == 'up':
            self.new_position_agent = (curr_x - 1, curr_y)
        if action == 'down':
            self.new_position_agent = (curr_x + 1, curr_y)
        if action == 'left':
            self.new_position_agent = (curr_x, curr_y - 1)
        if action == 'right':
            self.new_position_agent = (curr_x, curr_y + 1)
        
        new_x, new_y = self.new_position_agent
        new_state = int(self.dungeon[new_x][new_y])
        print("agent = ", self.position_agent)
        if new_state != self.OBSTACLE:
            # Return cell to previous value as the agent moves
            self.dungeon[curr_x][curr_y] = self.prev_agent_content
            self.prev_agent_content = self.dungeon[new_x][new_y]
            # Put agent on the new cell
            self.dungeon[new_x][new_y] = self.AGENT
            self.position_agent = self.new_position_agent
        print("new agent = ", self.position_agent)
        print("action = ", action)
        print("exit = ", self.position_exit)
        self.display()
        # calculate reward
        reward = self.rewards[new_state]

        # calculate observations
        observations = self._get_observtions()
        # update time
        self.time_elapsed += 1
        # verify termination condition
        done = False
        if new_state == self.EXIT or self.time_elapsed > self.time_limit:
            if self.time_elapsed >= self.time_limit:
                print("Run out of timesteps")
            else:
                print("Number of timesteps to completion = ", self.time_elapsed)
            done = True
        return observations, reward, done
    
    def display(self):
        
        # prints the environment
        print(self.dungeon)
        
    def reset(self):
        """
        This function resets the environment to its original state (time = 0).
        Then it places the agent and exit at new random locations.
        
        It is common practice to return the observations, 
        so that the agent can decide on the first action right after the resetting of the environment.
        
        """
        self.time_elapsed = 0
        # position of the agent is a numpy array
        self.position_agent = self._place_single_object(self.AGENT)
        self.prev_agent_content = self.EMPTY
        
        # position of the exit is a numpy array
        self.position_exit = self._place_single_object(self.EXIT)
        # Calculate observations
        observations = self._get_observtions()
        return observations

    def _initialise_environment(self):
        self.total_reward = 0
        self.dungeon = np.zeros((self.N, self.N))
        # Put obstacles on the grid borders
        self.dungeon[:1] = self.dungeon[0:, 0] = self.dungeon[0:,-1] = self.dungeon[-1:] = self.OBSTACLE
        # Put obstacles and lava randomly in the remaining dungeon grid
        self._place_objects(self.OBSTACLE)
        self._place_objects(self.LAVA)

    def _place_objects(self, instance_value):
        # place more than 1 of the same instance in the grid
        for _ in range(self.number_of_items):
            _, _ = self._place_object(instance_value)

    def _place_single_object(self, instance_value):
        # replace previous instance with empty cell. That way there is only one such object
        self.dungeon[self.dungeon == instance_value] = self.EMPTY
        return self._place_object(instance_value)

    def _place_object(self, instance_value):
        lower = 1
        upper = self.N - 2
        # Ensure instance is placed only in an empty cell.
        while True:
            x, y = self._get_random_coords(lower, upper)
            if self.dungeon[x][y] == self.EMPTY:
                self.dungeon[x][y] = instance_value
                return x, y

    def _get_random_coords(self, lower, upper):
        x = random.randint(lower, upper)
        y = random.randint(lower, upper)
        return x, y

    def _get_observtions(self):
        # Get a 3x3 subgrid from the dungeon of the cells surrounding the agent
        x, y = self.position_agent
        corner_x = x - 1
        corner_y = y - 1
        observations = self.dungeon[corner_x: corner_x + 3, corner_y: corner_y + 3]
        return observations

