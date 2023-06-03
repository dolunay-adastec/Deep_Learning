# Modeling a simple electric motor's power-speed-torque characteristic in Python

# 1 - Importing libraries
import numpy as np

# 2 - Define the motor's power-speed-torque relationship. You can choose a mathematical equation or use empirical data to model this relationship. For example, let's assume a linear relationship between power (P), speed (N), and torque (T):
# Motor Parameters:
# TM4 SUMA MD MV2500-6P
# Peak Power: 230 kW
# Peak Torque: 2500 Nm
# Max Speed: 3000 rpm
# Nominal Power: 115 kW
# Nominal Torque: 1140 Nm
# Nominal Speed: 1000 rpm
# @3000 rpm, T = 366 Nm
def motor_torque_model(N,pedal_input): # N is the speed in RPM, max speed is 3000 RPM
    
    T = np.zeros(3000) # T is the torque in Nm
    P = np.zeros(3000) # P is the power in W

    if N > 1000:
        T = 1140 - 0.387 * (N-1000)
    else:
        T = 1140

    # Calculate power
    # P = 2 * np.pi * N * T / 60
    
    return T*pedal_input

# 3 - Generate speed values for the desired range
# speed_range = np.linspace(0, 3000, 100)  # Speed range in RPM

# 4 - Calculate power and torque for each speed value using the motor_characteristic function:
# power_values, torque_values = motor_characteristic(speed_range)

# 5 - Plot the power-speed and torque-speed characteristics:
# plt.figure(figsize=(10, 6))
# plt.plot(speed_range, power_values/100, label='Power')
# plt.plot(speed_range, torque_values, label='Torque')
# plt.xlabel('Speed (RPM)')
# plt.ylabel('Power (100W) / Torque (Nm)')
# plt.legend()
# plt.grid(True)
# plt.title('Motor Power-Speed-Torque Characteristic')
# plt.show()