# # Lab 01 - Exit the dungeon!
# 
# In this first lab, we will create by hand our first Reinforcement Learning environment.
# A lot of agents will be harmed in the process of solving the lab.
# 
# ## The environment
# 
# The environment is a NxN array of integers. 
# Each cell of this environment can have the following values:
# - 0 : empty cell
# - 1 : obstacle, non-traversable
# - 2 : lava
# - 3 : exit
# 
# All border cells are obstacles.
# Upon initialization, the environment has:
# - N/2 obstacles placed randomly in the maze.
# - N/2 lava cells placed randomly in the cell.
# 
# ## The game
# 
# The agent starts in a random empty cell, and has to reach the exit.
# The exit is randomly positioned in an other empty cell.
# 
# At each timestep:
# - the agent decides on an action (move up, left, right or down)
# - the action is sent to the environment
# - the environment sends back observations, rewards and a boolean that indicates whether the environment terminated.
# 
# The environment terminates if the agent reaches the exit, or if the environement reaches a time limit of N^2 timesteps.
# 
# ## Observations
# 
# The agent receives a dictionary of observations:
# - target: relative coordinates of the exit 
# - proximity: a 3x3 array that encodes for the value of the cells around the agent.
# 
# ## Rewards
# 
# When acting, an agent receives a reward depending on the cell it ends up on:
# - if the agent moves towards an obstacle, it gets a reward of -5 and stays at its original position
# - if the agent is on a lava cell after its action, it receives a reward of -20
# - at each timestep, the agent receives an additional reward of -1
# - when the agent reaches the goal, it receives a reward of N**2
# 

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from textwrap import wrap
from dungeon import Dungeon

# # Part 1 - Defining the environment.
# 
# We will define the environment as a class.
# We are providing pseudo code which is incomplete and probably not completely error-free.
# 
# You have to fill the blanks.
# We advise you to look at the pseudo-code for Part 2 and 3 to have an idea of how things work together.
# 
# In order to make sure that your environment runs as intended, you will create a display function.

def main():
    dungeon = Dungeon(10)
    number_exp = 10
    r_max_reward, r_mean_reward, r_var_reward, r_all_rewards, r_all_timesteps = run_experiments(dungeon, random_policy, number_exp)
    max_reward, mean_reward, var_reward, all_rewards, all_timesteps = run_experiments(dungeon, intelligent_policy, number_exp)
    plot_rewards(r_all_rewards, all_rewards)
    print("max reward = ", max_reward)
    print("mean reward = ", mean_reward)
    print("var reward = ", var_reward)
    print("all rewards = ", all_rewards)
    print("number of timesteps = ", all_timesteps)
    print("\n***************")
    print("*** The END ***")
    print("***************")

# # Part 2 - Defining a policy
# 
# A policy tells the agent how to act depending on its current observation and internal beliefs.
# 
# As a first simple case, we will define policy as a function that maps observations to actions.
# 
# As your agent is stupid and doesn't have any way of learning what to do, in this first lab we will write by hand the policy.
# Try to come up with a strategy to terminate the game with the maximum reward.
# 
# We advise you to start with a very simple policy, then maybe try a random policy, and finally an 'intelligent' policy.
# 

def basic_policy(observation):
    action = "up"
    return action

def random_policy(observation):
    actions = ["up", "down", "left", "right"]
    upper = len(actions) - 1
    index = random.randint(0, upper)
    action = actions[index]
    return action

def _check_obs_up(type, obs_up):
    if obs_up == type:
        return "up"
    return ""

def _check_obs_down(type, obs_down):
    if obs_down == type:
        return "down"
    return ""

def _check_obs_left(type, obs_left):
    if obs_left == type:
        return "left"
    return ""

def _check_obs_right(type, obs_right):
    if obs_right == type:
        return "right"
    return ""
    
def intelligent_policy(observation):
    EMPTY = 0
    OBSTACLE = 1
    LAVA = 2
    EXIT = 3
    obs_up = observation[0][1]
    obs_down = observation[2][1]
    obs_left = observation[1][0]
    obs_right = observation[1][2]

    # Check observations starting from the highest rewards
    # Move to the exit. 
    # If exit not found, move to an empty cell
    # If empty cell not found, move to one with lava. Highly unlikely to happen
    # If surrounded by obstacles in all directions, choose a random policy. Also highly unlikely to happen
    if obs_up == EXIT:
        return "up"
    if obs_down == EXIT:
        return "down"
    if obs_left == EXIT:
        return "left"
    if obs_right == EXIT:
        return "right"

    # If the checks are always made in the same sequence, this put a bias in the model
    # The bias might cause the agent to be trapped in the corner of the grid
    # So the sequence of checks are randomised to prevent this bias.
    select_check = [_check_obs_up, _check_obs_down, _check_obs_left, _check_obs_right]
    select_param = [obs_up, obs_down, obs_left, obs_right]
    _idx = list(range(0, 4))
    random.shuffle(_idx)
    action = ""
    for idx in _idx:
        _function = select_check[idx]
        _observation = select_param[idx]
        action = _function(EMPTY, _observation)
        if len(action) > 0:
            return action

    # unlikely this code gets executed. Once an agent has moved, it does so from an empty cell.
    # and the agent can move back to that cell. 
    # Only if the agent has not moved yet will this code be reached
    if obs_up == LAVA:
        return "up"
    if obs_down == LAVA:
        return "down"
    if obs_left == LAVA:
        return "left"
    if obs_right == LAVA:
        return "right"
    return random_policy(observation)   # Agent is surrounded by obstacles and cannot move! 

# # Part 3a - Evaluating your policy
# 
# Now that you have the environment and policies, you can simulate runs of your games under different policies and evaluate the reward that particular policies will get upon termination of the environment. 
# 
# To that effect, we will create a function run_single_experiment, which will have as input:
# - an instance of an environment
# - a policy
# 
# And it will return the reward obtained once the environment terminates.
# 

def run_single_exp(envir, policy):
    
    obs = envir.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(obs)
        print("action = ", action)
        obs, reward, done = envir.step(action)
        total_reward += reward
        envir.display()
    
    return total_reward

# # Part 3b - Evaluating your policy (multiple runs)
# 
# Now that you can evaluate how a policy performs on a particular environment, consider the following.
# Because of stochasticity of initial agent position and exit position, different runs will lead to different total rewards.
# 
# To properly evaluate our policies, we must calculate the statistics over multiple runs.
# 
# To that effect, we will create a function run_experiments, which will have as input:
# - an instance of an environment
# - a policy
# - a number of times that the experiment will be run
# 
# It will return the maximum reward obtained over all the runs, the average and variance over the rewards.
# 

def run_experiments(envir, policy, number_exp):
    
    all_rewards = []
    all_timesteps = []
    for n in range(number_exp):
        
        final_reward = run_single_exp(envir, policy)
        all_rewards.append(final_reward)
        all_timesteps.append(envir.time_elapsed)
    
    max_reward = max(all_rewards)
    mean_reward = np.mean(all_rewards)
    var_reward = np.var(all_rewards)
    
    return max_reward, mean_reward, var_reward, all_rewards, all_timesteps
    

# # Part 4
# 
# Draw some plots to compare how your different policies perform depending on the environment size.
# 
# As the environment generation is also stochastic (random obstacles and lava), you might need to compute additional statistics.
# 
def plot_rewards(r_all_rewards, all_rewards):
    fig=plt.figure()
    plt.plot(r_all_rewards,'*r', all_rewards, 'Dg')
    title = "Comparing random policies (red stars) with intelligent policies (green diamonds)"
    plt.title("\n".join(wrap(title, 60)))
    plt.ylabel('rewards')
    plt.xlabel('episode')
    filename = "rewards.png"
    plt.savefig(filename)

if __name__ == "__main__":
    main()

