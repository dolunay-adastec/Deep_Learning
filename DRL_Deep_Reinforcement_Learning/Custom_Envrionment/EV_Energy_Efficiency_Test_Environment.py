# Environment for Testing an Control Method Energy Efficiency Performance for EV
# Author: Doğacan Dolunay Acar
# Date: 10.08.2021
# Version: 1.0
# Status: Development
# Last Modified Date: 10.08.2021
# References: 
# References: 
# _________________________________________________________________
# Environment Specifications:
# 1 - Observation Space: x, y, v, a, t, gradient, SOC
# 2 - Action Space:      -1, -0.9, -0.8, ..., 0.8, 0.9, 1. [-1, 0) = decelerate, (0, 1] = accelerate, 0 = no action
# 3 - Reward Function:   r = - energy_consumption - time_penalty + (10000 if x = 10000 else 0) + (1 if x_current > x_prev else 0) 
# 4 - Termination Condition:    x = 10000 or t = 3600
# 5 - Initial State:     x = 0, y = 0, v = 0, a = 0, t = 0, gradient = 0
# 6 - Episode Length:    10000
# 7 - Number of Episodes: 1000
# 8 - Number of Agents:  1
# 9 - Number of Actions: 21
# 10 - Number of Observations: 7
# 11 - Road: 10000 m. 0-2000 m gradient=0.0, 2000-4000 m gradient=0.04, 4000-6000 m gradient=0.0, 6000-8000 m gradient=-0.04, 8000-10000 m gradient=0
# 12 - Vehicle: 8000 kg mass, 5 m^2 frontal area, 0.3 drag coefficient, 0.023 rolling resistance coefficient, 0.9 drivetrain efficiency, 0.9 battery efficiency, 0.9 motor efficiency, 0.9 regenerative braking efficiency
# 13 - Battery: 300 kWh capacity, 300 kW power, 0.9 efficiency
# 14 - Motor: 150 kW power, 0.9 efficiency
# 15 - Regenerative Braking: 150 kW power, 0.9 efficiency
# 16 - Max Velocity: 70 km/h
# 17 - Max Acceleration: 3.0 m/s^2
# 18 - Max Deceleration: -3.0 m/s^2
# _________________________________________________________________



# 1 - Importing Libraries
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import emobpy
import math
import random
import time


# 2 - Defining Environment Class
class EV_Energy_Efficiency_Test_Environment(gym.Env):
    # 2.0 - Defining Constructor
    def __init__(self, episode_length, number_of_episodes, max_velocity, max_acceleration, max_deceleration ):
        """
        Constructor for EV Energy Efficiency Test Environment
        This environment is used to test the energy efficiency performance of a control method for EV
        Parameters:
            episode_length: Length of an episode, i.e. number of steps in an episode.
            number_of_episodes: Number of episodes, i.e. number of times the agent will be trained.
            max_velocity: Maximum velocity of the vehicle. Positive value, km/h.
            max_acceleration: Maximum acceleration of the vehicle. Positive value, m/s^2.
            max_deceleration: Maximum deceleration of the vehicle. Positive value, m/s^2.
        Returns:
            None
        """
        # 2.1 - Defining Environment Variables
        self.action_space = spaces.Discrete(21) # 21 actions: -1, -0.9, -0.8, ..., 0.8, 0.9, 1
        # 2.2 - Defining Observation Space. Observation Space: x, y, v, a, t, gradient, SOC, Power, Energy.
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -3.0, 0, -0.04, 0, -float('inf'), -float('inf')]), high=np.array([10000, 10000, max_velocity, 3.0, 3600, 0.04, 1, float('inf'), float('inf')]), dtype=np.float32) # x, y, v, a, t, gradient, SOC
        # 2.3 - Defining Reward Function
        self.reward_function = lambda energy_consumption, time_penalty, x_current, x_prev: - energy_consumption - time_penalty + (10000 if x_current > x_prev else 0)
        # 2.4 - Defining Termination Condition
        self.termination_condition = lambda x, t: x >= 9999 or t >= 3600
        # 2.5 - Defining Initial State
        self.initial_state = np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0]) # x, y, v, a, t, gradient, SOC, Power, Energy        
        # 2.6 - Defining Episode Length
        self.episode_length = episode_length
        # 2.7 - Defining Number of Episodes
        self.number_of_episodes = number_of_episodes
        # 2.8 - Defining Number of Agents
        self.number_of_agents = 1
        # 2.9 - Defining Number of Actions
        self.number_of_actions = 21
        # 2.10 - Defining Number of Observations
        self.number_of_observations = 9
        # 2.11 - Defining Road
        D1 = Road_Profiler(0, episode_length/5)  # 100 m. 0-100 m gradient=0
        D2 = Road_Profiler(0.04, episode_length/5)
        D3 = Road_Profiler(0, episode_length/5) 
        D4 = Road_Profiler(-0.04, episode_length/5)
        D5 = Road_Profiler(0, episode_length/5)
        R = [D1, D2, D3, D4, D5]
        self.road = Road_Merger(R)
        # 2.12 - Defining Vehicle
        self.vehicle = "EV"
        self.vehicle.mass = 8000
        self.vehicle.frontal_area = 5
        self.vehicle.drag_coefficient = 0.3
        self.vehicle.rolling_resistance_coefficient = 0.023
        self.vehicle.battery_capacity = 300
        self.vehicle.max_motor_torque = 1140
        self.vehicle.wheel_radius = 0.3
        self.vehicle.gear_ratio = 10


        
        = emobpy.Vehicle(mass=8000, frontal_area=5, drag_coefficient=0.3, rolling_resistance_coefficient=0.023, drivetrain_efficiency=0.9, battery_efficiency=0.9, motor_efficiency=0.9, regenerative_braking_efficiency=0.9) # 8000 kg mass, 5 m^2 frontal area, 0.3 drag coefficient, 0.023 rolling resistance coefficient, 0.9 drivetrain efficiency, 0.9 battery efficiency, 0.9 motor efficiency, 0.9 regenerative braking efficiency
        # 2.13 - Defining Battery
        self.battery = emobpy.Battery(capacity=300, power=300, efficiency=0.9) # 300 kWh capacity, 300 kW power, 0.9 efficiency
        # 2.14 - Defining Motor
        self.motor = emobpy.Motor(power=150, efficiency=0.9) # 150 kW power, 0.9 efficiency
        # 2.15 - Defining Regenerative Braking
        self.regenerative_braking = emobpy.RegenerativeBraking(power=150, efficiency=0.9) # 150 kW power, 0.9 efficiency
        # 2.16 - Defining Max Velocity
        self.max_velocity = max_velocity # 70 km/h
        # 2.17 - Defining Max Acceleration
        self.max_acceleration = max_acceleration # 3.0 m/s^2
        # 2.18 - Defining Max Deceleration
        self.max_deceleration = -max_deceleration # -3.0 m/s^2
        # 2.19 - Defining Current State
        self.current_state = self.initial_state
        # 2.20 - Defining Current Action
        self.current_action = 0
        # 2.21 - Defining Current Reward
        self.current_reward = 0
        # 2.22 - Defining Current Done
        self.current_done = False
        # 2.23 - Defining Current Info
        self.current_info = {}
        # 2.24 - Defining Current Episode
        self.current_episode = 0
        # 2.25 - Defining Current Step
        self.current_step = 0
        # 2.26 - Defining Current Total Reward
        self.current_total_reward = 0
        # 2.27 - Defining Current Total Energy Consumption
        self.current_total_energy_consumption = 0
        # 2.28 - Defining Current Total Time Penalty
        self.current_total_time_penalty = 0
        # 2.29 - Defining Current Total Distance
        self.current_total_distance = 0
        # 2.30 - Defining Current Total Time
        self.current_total_time = 0
        # 2.31 - Defining Current Total Energy Consumption List
        self.current_total_energy_consumption_list = []
        # 2.32 - Defining Current Total Time Penalty List
        self.current_total_time_penalty_list = []
        # 2.33 - Defining Current Total Distance List
        self.current_total_distance_list = []
        # 2.34 - Defining Current Total Time List
        self.current_total_time_list = []
        # 2.35 - Defining Current Total Reward List
        self.current_total_reward_list = []
        # 2.36 - 

        pass

    # 3 - Defining Reset Method
    def reset(self):
        """
        Reset method.
        Resetting the environment.
        This method is called when the environment is initialized and when an episode is finished.
        Before the environment is initialized, this method must be called.
        Parameters
                None
        Returns
                None
        """
        # 3.1 - Resetting Current State
        self.current_state = self.initial_state
        # 3.2 - Resetting Current Action
        self.current_action = 0
        # 3.3 - Resetting Current Reward
        self.current_reward = 0
        # 3.4 - Resetting Current Done
        self.current_done = False
        # 3.5 - Resetting Current Info
        self.current_info = {}
        # 3.6 - Resetting Current Step
        self.current_step = 0
        # 3.7 - Resetting Current Total Reward
        self.current_total_reward = 0
        # 3.8 - Resetting Current Total Energy Consumption
        self.current_total_energy_consumption = 0
        # 3.9 - Resetting Current Total Time Penalty
        self.current_total_time_penalty = 0
        # 3.10 - Resetting Current Total Distance
        self.current_total_distance = 0
        # 3.11 - Resetting Current Total Time
        self.current_total_time = 0
        # 3.12 - Resetting Current Total Energy Consumption List
        self.current_total_energy_consumption_list = []
        # 3.13 - Resetting Current Total Time Penalty List
        self.current_total_time_penalty_list = []
        # 3.14 - Resetting Current Total Distance List
        self.current_total_distance_list = []
        # 3.15 - Resetting Current Total Time List
        self.current_total_time_list = []
        # 3.16 - Resetting Current Total Reward List
        self.current_total_reward_list = []
        # 3.17 - Returning Current State
        return self.current_state
    
    # 4 - Defining Step Method
    def step(self, action):
        """
        Step method.
        Step method is used to update the environment state.
        This method is called when an action is taken.
        Before the action is taken, the current state is saved.
        """
        # 4.0 - Defining Current State
        self.previous_state = self.current_state # Saving Current State
        # 4.1 - Defining Current Action
        self.current_action = action # Saving Current Action
        # 4.2 - Defining Current State
        self.current_state = self.next_state(self.current_state, self.current_action) # Updating Current State
        # 4.3 - Defining Current Reward
        self.current_reward = self.reward(self.current_state, self.current_action) # Updating Current Reward
        # 4.4 - Defining Current Done
        self.current_done = self.done(self.current_state, self.current_action) # Updating Current Done
        # 4.5 - Defining Current Info
        self.current_info = self.info(self.current_state, self.current_action) # Updating Current Info
        # 4.6 - Defining Current Step
        self.current_step += 1 # Updating Current Step
        # 4.7 - Defining Current Total Reward
        self.current_total_reward += self.current_reward # Updating Current Total Reward
        # 4.8 - Defining Current Total Energy Consumption
        self.current_total_energy_consumption += self.current_info["energy_consumption"] # Updating Current Total Energy Consumption
        # 4.9 - Defining Current Total Time Penalty
        self.current_total_time_penalty += self.current_info["time_penalty"] # Updating Current Total Time Penalty
        # 4.10 - Defining Current Total Distance
        self.current_total_distance += self.current_info["distance"] # Updating Current Total Distance
        # 4.11 - Defining Current Total Time
        self.current_total_time += self.current_info["time"] # Updating Current Total Time
        # 4.12 - Appending Current Total Energy Consumption to Current Total Energy Consumption List
        self.current_total_energy_consumption_list.append(self.current_total_energy_consumption) # Updating Current Total Energy Consumption List
        # 4.13 - Appending Current Total Time Penalty to Current Total Time Penalty List
        self.current_total_time_penalty_list.append(self.current_total_time_penalty) # Updating Current Total Time Penalty List
        # 4.14 - Appending Current Total Distance to Current Total Distance List
        self.current_total_distance_list.append(self.current_total_distance) # Updating Current Total Distance List
        # 4.15 - Appending Current Total Time to Current Total Time List
        self.current_total_time_list.append(self.current_total_time) # Updating Current Total Time List
        # 4.16 - Appending Current Total Reward to Current Total Reward List
        self.current_total_reward_list.append(self.current_total_reward) # Updating Current Total Reward List
        # 4.17 - Returning Current State, Current Reward, Current Done, Current Info
        return self.current_state, self.current_reward, self.current_done, self.current_info # Returning Current State, Current Reward, Current Done, Current Info
    



    # 5 - Defining Next State Method
    def next_state(self, state, action):
        """
        Next State method.
        This method is used to update the environment state.
        This method is called when an action is taken.
        Parameters
                state: Current state.
                action: Current action.
        Returns
                next_state: Next state.
        """
        # Compute the next state based on the current state and action
        # Return the next state
        # Variables Parsing
        # 1 - Observation Space: np.array([0, 0, 0, 0, 0, 0, 0.5, 0, 0]) # x, y, v, a, t, gradient, SOC, Power, Energy 
        x = state[0] # Current X Position
        y = state[1] # Current Y Position
        v = state[2] # Current Velocity
        a = state[3] # Current Acceleration
        t = state[4] # Current Time
        gradient = state[5] # Current Gradient
        SOC = state[6] # Current SOC
        Power = state[7] # Current Power
        Energy = state[8] # Current Energy

        # ____________________________________________
        x += 1 # Updating X Position (m) every step = 1m


        



    # Electric Motor Power, Torque, and Speed Calculation
    # Reference: https://www.electrical4u.com/electric-motor-power-speed-torque-equations/
    # Reference: https://www.electrical4u.com/electric-motor-power-speed-torque-equations-and-torque-curve/
    # Reference: https://github.com/rmrubin/pymotor/tree/master/pymotor
    
    def motor_torque_model(N,pedal_input): # N is the speed in RPM, max speed is 3000 RPM
    
        T = 0
        P = 0

        if N > 1000:
                T = 1140 - 0.387 * (N-1000)
        else:
                T = 1140

        # Calculate power
        # P = 2 * np.pi * N * T / 60
    
        return T * pedal_input
    
    # Electric Vehicle Drive Train Model
        # Reference: https://www.researchgate.net/publication/318672012_Electric_vehicle_drive_train_modeling_and_simulation
        # Reference: https://www.researchgate.net/publication/318672012_Electric_vehicle_drive_train_modeling_and_simulation

        #def drive_train_model(T, N, v, a, t, gradient): # T is the torque in Nm, N is the speed in RPM, v is the velocity in m/s, a is the acceleration in m/s^2, t is the time in s, gradient is the road gradient in %






def Road_Profiler(gradient, distance):
        D = {}
        D['gradient'] = gradient
        D['degree'] = math.atan(D['gradient']) * 180 / math.pi
        D['radian'] = math.atan(D['gradient'])
        D['x'] = [i * math.cos(D['radian']) for i in range(distance+1)]
        if D['radian'] == 0:
            D['y'] = [0] * (distance+1)
        else:
            D['y'] = [i * math.sin(D['radian']) for i in range(distance+1)]
        D['Distance'] = distance
    
        return D


def Road_Merger(R):
        Road = {'x': [], 'y': [], 'gradients': [], 'Distances': [], 'Total_Distance': 0}

        road_count = len(R)
    
        for i in range(road_count):
            if i == 0:
                Road['x'].extend(R[i]['x'])
                Road['y'].extend(R[i]['y'])
            else:
                Road['x'].extend([Road['x'][-1] + x for x in R[i]['x'][1:]])
                Road['y'].extend([Road['y'][-1] + y for y in R[i]['y'][1:]])
        
            Road['Distances'].append(R[i]['Distance'])
            Road['gradients'].append(R[i]['gradient'])
            Road['Total_Distance'] += R[i]['Distance']
    
        return Road

