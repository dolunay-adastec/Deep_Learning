# Importing the necessary libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from collections import deque
from IPython.display import clear_output
import function_library as fl
import Custom_Environment
# Import the Agent
from DQN_Agent import DQN_Agent
# Instantiating the agent
agent = DQN_Agent(state_size=14, action_size=21, seed=0)
# Creating a folder to store the results
folder_path = fl.create_folder()
# Training the agent
def dqn(n_episodes=1000, max_t=1001, eps_start=1.0, eps_end=0.01, eps_decay=0.995,folder_path=folder_path):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): Maximum number of training episodes. In this case, it is 2000.
        max_t (int): Maximum number of timesteps per episode. In this case, it is 1000.
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection. In this case, it is 1.
        eps_end (float): Minimum value of epsilon. In this case, it is 0.01.
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon. In this case, it is 0.995.
    """
    
    # Creating a database to store the results
    columnm = ['Pos','act','x','y','v','a','Duration','Gradient','SOC','Power','Energy','F_load','F_traction','Wind','dt','Brake','Throttle','Power','Torque','dt_Energy_wh','Total_Reward','dt_Time_reward','dt_Energy_reward','dt_Position_reward']
    database = pd.DataFrame(columns=columnm)
    # Creating a list to store the scores
    scores = []
    # Creating a list to store the scores
    scores_window = deque(maxlen=100)
    # Defining the epsilon value
    eps = eps_start

    # Instantiating the Environment
    env = Custom_Environment.CustomEnv()
    env.reset()

    # Iterating over the number of episodes
    #for i_episode in range(1, n_episodes+1):
    for i_episode in range(1, 50+1):
        # Resetting the environment
        state = env.reset()
        # Resetting the score
        score = 0
        # Iterating over the number of timesteps
        for t in range(max_t+1):
            # Selecting an action
            action = agent.act(state, eps)
            # Taking a step
            next_state, reward, done, _ = env.step(action)
            # Storing the experience
            agent.step(state, action, reward, next_state, done)
            # Updating the state
            state = next_state
            # Updating the score
            score += reward
            # Storing the results
            database.loc[t] = np.concatenate([env.current_state, env.debug_values])
            # Checking if the episode is done
            if done:
                break
        # Appending the score to the scores list
        scores_window.append(score)
        # Appending the score to the scores list
        scores.append(score)
        # Decreasing the epsilon value
        eps = max(eps_end, eps_decay*eps)
        # Printing the results
        print('\rEpisode {}\tAverage Score: {:.2f}\t'.format(i_episode, np.mean(scores_window)), end="")
        # Printing the results
        Score, plot = fl.grapher(env, database,show_fig=True)
        # Saving the results
        fl.save_outputs(folder_path, database, i_episode, Score)
        # Clearing the output
        clear_output(wait=True)
    return scores

# main function
if __name__ == '__main__':
    # Instantiating the agent
    agent = DQN_Agent(state_size=14, action_size=21, seed=0)
    # Creating a folder to store the results
    folder_path = fl.create_folder()
    # Training the agent
    scores = dqn()
    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Results of DQN Agent')
    plt.savefig(folder_path + '/Score.png')
    plt.show()
