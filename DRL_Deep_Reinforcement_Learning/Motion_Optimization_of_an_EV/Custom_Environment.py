# Custom Gym Environment for the Motion Optimization of an Electric Vehicle for Range Extension
# Author: Doğacan Dolunay Acar
# Date: 2023-06-10
# Version: 1.4.0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Action Space      : [-1,0) Brake, 0 None, (0,1] Accelerate, Discrimination step is 0.1.
# Observation Space : pos, act, x, y, v, a, t, gradient, SOC, Power, Energy, F_load, F_traction. act is the action taken in the previous time step. pos is the position of the vehicle in the road profile.
# Space limits: act [-1,1], x [0,1000] m, y [0,8] m, v [0,20] mps, a [-3,3], t [0,1000], gradient [-0.04,0.04], SOC [0,1], Power [-inf,inf], Energy [-inf,inf].
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing Libraries
import gym          # Gym Library
from gym import spaces # Gym Environment
import math         # Mathematical Functions
import numpy as np  # Numerical Python
import pandas as pd # Data Analysis
import sympy as sp  # Symbolic Python
from function_library import Road_Profiler, Road_Merger # my own function library
import random       # Random Number Generator
import copy         # Copying Objects
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Environment Class
class CustomEnv(gym.Env):
    
    def __init__(self,wind_disturbance_enabled=True):
        self.action_space = spaces.Discrete(21) # 21 actions: -1, -0.9, ..., -0.1, 0, 0.1, ..., 0.9, 1
        # pos [0-1001] road index, act[-1 1], x [0-1000] m, y [0-8] m, v [0-20] mps, a [-5 5] ms^-2, t [0-1000] s, gradient [-0.04 0.04] %, SOC [0 1], Power [-inf inf] kW, Energy [-inf inf] kWh, Wind Speed [-10 10] mps
        self.observation_space = spaces.Box(low=np.array([0, -1, 0, 0, 0, -5, 0, -0.04, 0, -float('inf'), -float('inf'), -10]), high=np.array([1001, 1, 1000, 8, 20, 5, 1000, 0.04, 1, float('inf'), float('inf'), 10]), dtype=np.float32)
        # Defining Road Profile
        D1 = Road_Profiler(0,200)
        D2 = Road_Profiler(0.04,200)
        D3 = Road_Profiler(0,200)
        D4 = Road_Profiler(-0.04,200)
        D5 = Road_Profiler(0,200)
        self.road = Road_Merger([D1,D2,D3,D4,D5]) # 1001 points, 1000 m long road, 4% uphill and downhill
        # Hyperparameters
        self.wind_disturbance_enabled = wind_disturbance_enabled
        # Vehicle Parameters
        self.gear_ratio = 10            # 10:1
        self.wheel_radius = 0.3         # m
        self.mass = 2000                # kg
        self.Cd = 0.3                   # Drag Coefficient
        self.A = 2.5                    # Frontal Area
        self.rho = 1.225                # Air Density
        self.rolling_resistance = 0.008 # Rolling Resistance Coefficient

        # Motor paramters
        self.max_torque = 180           # Nm
        self.max_rpm = 6366             # rpm, The maxiumum speed of the electric vehicle is 72 km/h, 20 m/s
        self.max_speed = 20             # m/s

        # Storing Variables for Code Optimization (SVCO)
        self.SVCO_rolling_resistance_force  = self.rolling_resistance * self.mass * 9.81    # N
        self.SVCO_aero_drag_force           = 0.5 * self.Cd * self.A * self.rho             # N
        self.SVCO_gradient_force            = self.mass * 9.81                              # N
        self.SVCO_gear_wheel_radius         = self.gear_ratio / self.wheel_radius           # m
        # Electrical Parameters
        self.battery_capacity = 10     # kWh
        self.regenerative_efficiency = 0.8 # 80% of the energy is recovered during regenerative braking and stored in the battery pack. Because of the losses in the battery pack and the motor, the efficiency is 80%.
        # Wind Disturbance
        self.noise = OUNoise(1, 8, 0, 0.05, 5)	# 1, 0, 0, 0.05, 4

    def reset(self):
        self.pos = 0
        self.act = 0
        self.x = 0
        self.y = 0
        self.v = 0
        self.a = 0
        self.t = 0
        self.gradient = 0
        self.SOC = 0.5
        self.Power = 0
        self.Energy = 0
        self.F_load = 0
        self.F_traction = 0
        self.w = 0
        self.noise.reset()
        self.done = False
        self.info = {}
        return np.array([self.pos, self.act, self.x, self.y, self.v, self.a, self.t, self.gradient, self.SOC, self.Power, self.Energy, self.F_load, self.F_traction, self.w])
    
    # Step Function of the Environment (Action to Reward)
    def step(self, action):

        # State Update
        self.act = action
        self.current_state = np.array([self.pos, self.act, self.x, self.y, self.v, self.a, self.t, self.gradient, self.SOC, self.Power, self.Energy, self.F_load, self.F_traction, self.w])

        # Update Environment Variables
        self.gradient = self.road['Gradient'][self.pos]
        self.x = self.road['x'][self.pos]
        self.y = self.road['y'][self.pos]


        # Action to Brake and Throttle
        brake, throttle = self.action_to_brake_and_throttle(action)
        # Brake and Throttle to Power and Torque
        P, T = self.vehicle_motor_model(self.v, brake, throttle)    # P [kW], T [Nm]

        # Wind Disturbance ___ Max Wind Speed = 10 m/s, Min Wind Speed = -10 m/s, Std = 4.68, Mean = 1
        if self.wind_disturbance_enabled:
            self.w = 50 - self.noise.sample() # Wind Speed [m/s]
            self.w = np.clip(self.w, -4, 4)
        else:
            self.w = 0
        # ____________________________________________________________________________

        # Vehicle Dynamic Model
        self.v, self.a, self.F_load, self.F_traction, self.d_t = self.vehicle_dynamic_model(self.v, T, self.w)

        # Update Time
        self.t += self.d_t # Time [s]

        if self.v > 0:   self.pos += 1
        elif self.v < 0: self.pos -= 1
        else: self.pos = self.pos

        # TODO: Update the SOC, Power, Energy
        # Energy, SOC and Power Calculation
        self.Power = P                              # Power [kW]
        if self.Power < 0: self.Power = self.Power * self.regenerative_efficiency # Power [kW], Regenerative Efficiency [0-1]
        self.dt_Energy = self.Power * self.d_t / 3.6 # Energy [Wh], Power [kW], d_t [s]
        self.Energy += self.dt_Energy/1000           # Energy [kWh]
        self.SOC = self.SOC - (self.dt_Energy / 1000) / self.battery_capacity # SOC [0-1], Energy [kWh], Battery Capacity [kWh]

        # Update the state
        self.next_state = np.array([self.pos, self.act, self.x, self.y, self.v, self.a, self.t, self.gradient, self.SOC, self.Power, self.Energy, self.F_load, self.F_traction, self.w])
        
        # TODO: CREATE a REWARD FUNCTION
        # Reward Calculation # if SOC decrease -> reward decrease, if pos increase -> reward increase, if d_t decrease -> reward increase #### REWARD FUNCTION ####
        # self.reward = lambda x: +1 if
        dt_Total_Reward, dt_Time_point, dt_Energy_point, dt_Position_point =  self.reward_function()
        self.reward = dt_Total_Reward

        # TODO: Store all data in a dataframe to visualize the results. In Main Loop ________________________________________________ TODO
        # Create a np array to store other variables for debugging
        self.debug_values = np.array([self.d_t,brake, throttle, P, T, self.dt_Energy, self.reward, dt_Time_point, dt_Energy_point, dt_Position_point])
        
        # Return the next state, reward, done and info
        return np.array(self.next_state), self.reward, self.done, self.info
    
    # Reward Function of the Environment
    # self.dt max = 3.2884 sec, min = 0.0526 sec. Mean = 1.6705 , Deviation = 3.2358
    # self.dt_Energy max = 1.654 Wh, min = -1.654 Wh. Mean = 0.0, Deviation = 3.308
    def reward_function(self):
        # Time Reward
        dt_Time_point   = - self.d_t        # Encourage to drive faster
        if self.t > 200:
            self.done = True                    # If the time is greater than 200 sec, the episode is done
            dt_Time_point = -1000               # Penalize the agent for taking too long
        # Energy Reward
        dt_Energy_point = - self.dt_Energy/2      # Encourage to save energy
        # Position Reward
        dt_Position_point = 0.2 if (float(self.next_state[0]) - float(self.current_state[0])>0) else -2 # pos_next - pos_current = 1 else -1. Encourage to drive forward
        # Goal Reward
        if self.pos == self.road['Total_Distance']:
            dt_Position_point = 100             # Reward the agent for reaching the goal
            self.done = True                    # If the goal is reached, the episode is done
        # Total Reward
        dt_Total_Reward = dt_Time_point + dt_Energy_point + dt_Position_point
        # Return the values
        return dt_Total_Reward, dt_Time_point, dt_Energy_point, dt_Position_point

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -------------------------- A D D I T I O N A L ---------------------------------- F U N C T I O N S ------------------------------------ S E C T I O N ---------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Action to Brake and Throttle
    def action_to_brake_and_throttle(self, action):
        brake = 0
        throttle = 0
        action = float(action)
        if action > 1: action = 1
        if action < -1: action = -1
        if action == 0:
            brake = 0
            throttle = 0
        elif 0 < action <= 1:
            brake = 0
            throttle = action
        else:
            brake = action
            throttle = 0
        return brake, throttle
    
    # Motor Model
    # TODO: Validate and Verify the Motor Model: DONE
    def vehicle_motor_model(self, v, brake, throttle):
        # Motor Parameters
        # Max Torque = 180 Nm
        # Max RPM = 6876 -> Vehicle max speed = 72 km/h
        T = 0
        P = 0
        # Vehicle speed to motor RPM, rpm means rotation per minute
        N = v * self.gear_ratio * 30 / ( np.pi * self.wheel_radius ) # v [mps], N [rpm]
        # Limit the RPM
        N = min(N, self.max_rpm) # N [rpm], 6876 rpm = 72 km/h
        # Convert RPM to rad/s
        N_rads = N * 2 * np.pi / 60 # N [rpm], N_rads [rad/s]

        # Calculate torque
        # motor model: Motor supply its max torque from 0 rpm to 4000 rpm, after 4000 rpm torque decreases linearly to 20 at 6366 rpm
        T = self.max_torque if N <= 4000 else self.max_torque - 0.06339 * (N - 4000)    # T_min = 30 Nm at 6876 rpm
        
        # Throttle, Brake, or No Torque Calculation. When throttle is 1, brake is 0. When brake is 1, throttle is 0.
        if self.v >= 0:
            if throttle > 0:
                T = T * throttle # T [Nm]
            elif brake < 0:
                T = T * brake   # T [Nm]
            else:
                T = 0           # T [Nm]
        else:
            v = 0               # v [km/h], v cannot be negative

        # Calculate Power
        P = 0.001 * N_rads * T # P [kW]
        return P, T
    
    # Vehicle Dynamic Model
    # TODO: Validate and Verify the Vehicle Dynamic Model: DONE
    def vehicle_dynamic_model(self,v,T,w):
        # Load Forces
        # Rolling Resistance
        if v > 0:
            F_rolling = self.SVCO_rolling_resistance_force  # F_rolling [N] , precalcutated
        else:
            F_rolling = 0
        # Aerodynamic Drag
        # F_aero = 0.5 * self.rho * self.Cd * self.A * (v-w)**2   # F_aero [N], v [km/h], wind [km/h]
        F_aero = self.SVCO_aero_drag_force * (v-w)**2   # F_aero [N], v [km/h], wind [km/h]
        # Gradient Resistance (Hill)
        F_gradient = self.SVCO_gradient_force * np.sin(np.arctan(self.gradient))   # F_gradient [N], slope to radian
        # Total Load Force
        F_load = F_rolling + F_aero + F_gradient                # F_load [N]
        # Traction Force
        F_traction = T * self.SVCO_gear_wheel_radius            # F_traction [N]
        # Acceleration
        a = (F_traction - F_load) / self.mass                   # a [m/s^2]
        # Time Calculation
        self.d_t = self.Time_Calculation(v, a)                  # d_t [s] | There is not any solution for negative speeds
        # Velocity
        v = v + a * self.d_t                                    # v [m/s]
        if v != 0 and self.d_t == 0: v = 0                        # If the vehicle speed goes to the under the zero, the velocity is 0
        if v > self.max_speed: v = self.max_speed               # v [m/s] # Limit the speed
        # output
        return v, a, F_load, F_traction, self.d_t
        

    # Time Calculation for 1 meter distance
    def Time_Calculation(self, V_initial, a):
        # solve the quadratic equation 0.5*a*t^2 + V_initial*t - 1 = 0
        # equation form is at^2 + bt + c = 0
        # solution is t = [-b ± sqrt(b^2 - 4ac)] / (2a)
        a *= 0.5
        b = V_initial
        c = -1

        # calculate discriminant
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            # no real solutions
            return 0
        epsilon = 1e-8
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a + epsilon)
        t2 = (-b + sqrt_discriminant) / (2*a + epsilon)

        # return the smallest positive solution in range (0, 10)
        if 0 < t1 < 10 and 0 < t2 < 10:
            return min(t1, t2)
        elif 0 < t1 < 10:
            return t1
        elif 0 < t2 < 10:
            return t2
        else:
            return 0



# Ornstein-Uhlenbeck process
class OUNoise:
    """Ornstein-Uhlenbeck process.
    Source: 1. https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
            2. https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, size, seed, mu=0., theta=0.1, sigma=10/3): # sigma=0.3
        """Initialize parameters and noise process."""
        # Initialize the parameters
        self.mu = mu * np.ones(size)    # mu is the mean

        self.theta = theta              # theta is the rate of mean reversion. It is the rate at which the process pulls towards the mean

        # For the maximum and minimum wind speeds to be 10 and -10 respectively: The OU process fluctuates around a normal distribution,
        # the standard deviation of which is determined by the sigma parameter. In a standard normal distribution,
        # 99.73% of the values fall within +/- 3 standard deviations from the mean. Therefore, if we want 99.73% of the wind speeds to fall between -10 and 10,
        # we can choose sigma = 10 / 3.
        self.sigma = sigma              # sigma is the volatility parameter. It is the standard deviation of the process
        self.seed = random.seed(seed)   # seed is the random seed
        # Initialize the noise process
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        # Reset the internal state
        self.OU_state = copy.copy(self.mu) # state is the internal state of the process

    def sample(self):
        """Update internal state and return it as a noise sample."""
        # Update the internal state
        x = self.OU_state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        # dx: the change in the noise variable at each time step
        # self.theta:  is the parameter that determines how fast the noise variable reverts to the mean (self.mu).
        # (self.mu - x): represents the difference between the mean and the current state of the noise variable.
        # self.sigma: is the standard deviation of the process. It determines the amount of variation in the noise variable.
        # np.array([random.random() for i in range(len(x))]): is the random component of the process. It is a random variable that is drawn from a normal distribution with mean 0 and standard deviation 1.
        # Generates an array of random numbers drawn from a uniform distribution between 0 and 1. The length of this array matches the length of the input x, which is typically the number of dimensions in the action space.
        # Return the noise sample
        # self.sigma * np.array([random.random() for i in range(len(x))]) scales the random array by the standard deviation (self.sigma).
        # 1 - The difference between the mean (self.mu) and the current state (x), scaled by the parameter self.theta. This component represents the tendency of the noise variable to revert towards the mean.
        # 2 - A random array scaled by the standard deviation (self.sigma). This component introduces random fluctuations to the noise variable.
        # By adjusting the values of self.theta and self.sigma, you can control the properties of the generated noise, such as its speed of reversion to the mean and its overall magnitude.
        #self.OU_state = 50 - x + dx
        #self.OU_state = np.clip(self.OU_state, -10, 10) # clip the wind speed between -10 and 10
        self.OU_state = x + dx
        return self.OU_state.item() # return the noise sample as a float