# Function Library for Motion Optimization of an Electric Vehicle for Range Extension
# Author: Doğacan Dolunay Acar
# Date: 2023-06-10
# Version: 1.0.0

# Importing Libraries
import math
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Optimized Road Profiler Function: This function creates a road profile with given gradient and distance.
def Road_Profiler(gradient, distance):
    """This function creates a road profile with given gradient and distance.
        Input: gradient [%], distance [m].
        Output: A Road D; D.gradient, D.degree, D.radian, D.x, D.y, D.Distance.
        ----------
        How To Use
        D1 = Road_Profiler(0, 200).
        D2 = Road_Profiler(0.04, 200).
        D3 = Road_Profiler(0, 200).
        D4 = Road_Profiler(-0.04, 200).
        D5 = Road_Profiler(0, 200).
        """
    D = {}
    D['Gradient'] = np.full(distance+1, gradient)
    D['degree'] = math.atan(D['Gradient'][1]) * 180 / math.pi
    D['radian'] = math.atan(D['Gradient'][1])
    D['x'] = np.arange(distance+1) * math.cos(D['radian'])
    D['y'] = np.where(D['radian'] == 0, 0, np.arange(distance+1) * math.sin(D['radian']))
    D['Distance'] = distance

    return D
# Optimized Road Merger Function: This function merges the road profiles.
def Road_Merger(R):
    """This function merges the road profiles.
        Input: R = [D1, D2, D3, D4, D5, ...]
        Output: Merged roads; Road.x, Road.y, Road.gradients, Road.Distances, Road.Total_Distance. 
        ----------
        """
    Road = {'x': [], 'y': [], 'Gradient': [], 'Distances': [], 'Total_Distance': 0}

    road_count = len(R)
    
    for i in range(road_count):
        if i == 0:
            Road['x'].extend(R[i]['x'])
            Road['y'].extend(R[i]['y'])
            Road['Gradient'].extend(R[i]['Gradient'])
        else:
            Road['x'].extend(Road['x'][-1] + R[i]['x'][1:])
            Road['y'].extend(Road['y'][-1] + R[i]['y'][1:])
            Road['Gradient'].extend(R[i]['Gradient'][1:])

        Road['Distances'].append(R[i]['Distance'])
        Road['Total_Distance'] += R[i]['Distance']

    # Convert lists to NumPy arrays
    for key in ['x', 'y', 'Gradient', 'Distances']:
        Road[key] = np.array(Road[key])

    return Road

def grapher(env, database, database_2, show_fig):
    # Reward distribution calculation
    database = database.astype(float)
    total_time_reward = database['dt_Time_reward'].sum()
    total_energy_reward = database['dt_Energy_reward'].sum()
    total_position_reward = database['dt_Position_reward'].sum()
    total_reward = database['Total_Reward'].sum()
    time_reward_percentage = float(total_time_reward) / float(total_reward) *100          # % of the total reward
    energy_reward_percentage = float(total_energy_reward) / float(total_reward) *100      # % of the total reward
    position_reward_percentage = float(total_position_reward) / float(total_reward) *100  # % of the total reward
    # For Dummy
    database_2 = database_2.astype(float)
    total_time_reward_2 = database_2['dt_Time_reward'].sum()
    total_energy_reward_2 = database_2['dt_Energy_reward'].sum()
    total_position_reward_2 = database_2['dt_Position_reward'].sum()
    total_reward_2 = database_2['Total_Reward'].sum()
    time_reward_percentage_2 = float(total_time_reward_2) / float(total_reward_2) *100          # % of the total reward
    energy_reward_percentage_2 = float(total_energy_reward_2) / float(total_reward_2) *100      # % of the total reward
    position_reward_percentage_2 = float(total_position_reward_2) / float(total_reward_2) *100  # % of the total reward
    print("Total Reward:           Agent: ", total_reward, "Dummy: ", total_reward_2)
    print("Total Time Reward:      Agent: ", total_time_reward, "Dummy: ", total_time_reward_2)
    print("Total Energy Reward:    Agent: ", total_energy_reward, "Dummy: ", total_energy_reward_2)
    print("Total Position Reward:  Agent: ", total_position_reward, "Dummy: ", total_position_reward_2)


    fig, axs = plt.subplots(5,3,figsize=(20,20))
    fig.suptitle('Environment and Agent Performance Graph', fontsize=16, fontweight='bold')
    
    # Adding the Performance Indicator Squares
    # Subplot 1
    # Total Reward
    axs[0,0].title.set_text('Agent Controlled Vehicle')
    axs[0,0].title.set_fontsize(12)
    axs[0,0].title.set_fontweight('bold')
    axs[0,0].axis('off')
    # Total Energy
    axs[0,0].add_patch(patches.Rectangle((0.0, 0.8), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Energy'].iloc[-1]*1000,".2f")
    axs[0,0].text(0.5, 0.9, 'Net Energy [Wh] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Consumted Energy
    axs[0,0].add_patch(patches.Rectangle((0.0, 0.6), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['dt_Energy_wh'][database['dt_Energy_wh'] > 0].sum(),".2f")
    axs[0,0].text(0.5, 0.7, 'Consumed Energy [Wh] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Regenerated Energy
    axs[0,0].add_patch(patches.Rectangle((0.0, 0.0), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['dt_Energy_wh'][database['dt_Energy_wh'] < 0].sum(),".2f")
    axs[0,0].text(0.5, 0.1, 'Regenerated Energy [Wh] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Average Energy Consumption, Wh/km
    axs[0,0].add_patch(patches.Rectangle((0.0, 0.2), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Energy'].iloc[-1]/(database['Pos'].iloc[-1]/1000),".2f")
    axs[0,0].text(0.5, 0.3, 'Avrg. Enrgy Cons. [kWh/km] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Distace Traveled
    axs[0,0].add_patch(patches.Rectangle((0.0, 0.4), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Pos'].iloc[-1],".2f")
    axs[0,0].text(0.5, 0.5, 'Travelled Distance [m] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')

    
    
    # Subplot 2
    axs[0,1].title.set_text('Agent Controlled Vehicle')
    axs[0,1].title.set_fontsize(12)
    axs[0,1].title.set_fontweight('bold')
    axs[0,1].axis('off')
    
    # Regen Ratio
    axs[0,1].add_patch(patches.Rectangle((0.0, 0.4), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(abs((database['dt_Energy_wh'][database['dt_Energy_wh'] < 0].sum()/database['dt_Energy_wh'][database['dt_Energy_wh'] > 0].sum())*100),".2f")
    axs[0,1].text(0.5, 0.5, 'Regen Ratio = {} %'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Total Reward
    axs[0,1].add_patch(patches.Rectangle((0.0, 0.8), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Total_Reward'].sum(),".2f")
    axs[0,1].text(0.5, 0.9, 'Total Reward = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Total Time
    axs[0,1].add_patch(patches.Rectangle((0.0, 0.6), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Duration'].iloc[-1],".2f")
    axs[0,1].text(0.5, 0.7, 'Duration [sec] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    
    # Average Speed, km/h
    axs[0,1].add_patch(patches.Rectangle((0.0, 0.2), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database['Pos'].iloc[-1]/database['Duration'].iloc[-1]*3.6,".2f")
    axs[0,1].text(0.5, 0.3, 'Average Speed [km/h] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Reward distribution over time, energy and position
    axs[0,1].add_patch(patches.Rectangle((0.0, 0.0), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text_1 = format(time_reward_percentage,".1f")
    formatted_text_2 = format(energy_reward_percentage,".1f")
    formatted_text_3 = format(position_reward_percentage,".1f")
    axs[0,1].text(0.5, 0.11, 'Reward Distr| Time, Energy, Position', ha='center', va='bottom', color='white', fontsize=12,fontweight='bold')
    axs[0,1].text(0.65, 0.09, '{}%, {}%, {}%'.format(formatted_text_1,formatted_text_2,formatted_text_3), ha='center', va='top', color='white', fontsize=12,fontweight='bold')
    del formatted_text, formatted_text_1, formatted_text_2, formatted_text_3
    
    
    
    # Subplot 3
    # Plot the road
    axs[1,0].plot(env.road['x'], env.road['y'])
    axs[1,0].set_xlabel('x [m]')
    axs[1,0].set_ylabel('y [m]')
    axs[1,0].set_title('Road Profile (Environment)', fontsize=11, fontweight='bold')
    axs[1,0].grid()
    
    # Subplot 4
    # Plot the actions
    axs[1,1].plot(database['Pos'], database['act'])
    axs[1,1].set_xlabel('Position [m]')
    axs[1,1].set_ylabel('Action [-]')
    axs[1,1].set_title('Action Profile (Agent)', fontsize=11, fontweight='bold')
    axs[1,1].grid()
    
    # Subplot 5
    # Plot the velocity
    axs[3,0].plot(database['Pos'], database['v']*3.6)
    axs[3,0].plot(database_2['Pos'], database_2['v']*3.6, '--')
    axs[3,0].set_xlabel('Position [m]')
    axs[3,0].set_ylabel('Velocity [km/h]')
    axs[3,0].set_title('Velocity Profile (Agent & Dummy)', fontsize=11, fontweight='bold')
    axs[3,0].legend(['Agent','Dummy'], loc='upper left', fontsize=10)
    axs[3,0].grid()
    
    # Subplot 6
    # Plot the Rewards
    axs[2,1].plot(database['Pos'], database['Total_Reward'], color='blue')
    axs[2,1].plot(database['Pos'], database['dt_Time_reward'], '--', color='red')
    axs[2,1].plot(database['Pos'], database['dt_Energy_reward'], '--', color='green')
    axs[2,1].plot(database['Pos'], database['dt_Position_reward'], '--', color='orange')
    axs[2,1].set_xlabel('Position [m]')
    axs[2,1].set_ylabel('Reward [-]')
    axs[2,1].set_title('Reward Profile (Agent)', fontsize=11, fontweight='bold')
    axs[2,1].legend(['Total Reward','Time Reward','Energy Reward','Position Reward'], loc='upper left', fontsize=10)
    axs[2,1].grid()
    
    # Subplot 7
    # Plot the Power and Energy
    axs[2,0].plot(database['Pos'], database['Wind'])
    axs[2,0].set_xlabel('Position [m]')
    axs[2,0].set_ylabel('Wind Speed [m/s]')
    axs[2,0].set_title('Wind Disturbance Profile (Environment)', fontsize=11, fontweight='bold')
    axs[2,0].grid()
    
    # Subplot 8
    # Plot the SOC
    axs[4,0].plot(database['Pos'], database['SOC'])
    axs[4,0].plot(database_2['Pos'], database_2['SOC'])
    axs[4,0].set_xlabel('Position [m]')
    axs[4,0].set_ylabel('SOC [%]')
    axs[4,0].set_title('SOC Profile (Agent & Dummy)', fontsize=11, fontweight='bold')
    axs[4,0].legend(['Agent','Dummy'], loc='lower left', fontsize=10)
    axs[4,0].grid()
    
    # Subplot 9
    # Plot the Torque and Power
    axs[4,1].plot(database['Pos'], database['Torque'], color='blue')
    axs[4,1].plot(database['Pos'], database['Power'], '--', color='red')
    axs[4,1].set_xlabel('Position [m]')
    axs[4,1].set_ylabel('Torque [Nm] / Power [W]')
    axs[4,1].set_title('Torque and Power Profile (Agent)', fontsize=11, fontweight='bold')
    axs[4,1].legend(['Torque','Power'], loc='lower left', fontsize=10)
    axs[4,1].grid()
    
    
    
    # Plot the Total Reward
    axs[3,1].plot(database['Pos'], database['Total_Reward'].cumsum())
    axs[3,1].set_xlabel('Position [m]')
    axs[3,1].set_ylabel('Total Reward [-]')
    axs[3,1].set_title('Total Reward (Agent)', fontsize=11, fontweight='bold')
    axs[3,1].grid()
    
    # 3rd Column
    # Subplot 1
    # Total Reward
    axs[0,2].title.set_text('Dummy Vehicle @ Max Speed')
    axs[0,2].title.set_fontsize(12)
    axs[0,2].title.set_fontweight('bold')
    axs[0,2].axis('off')
    axs[0,2].add_patch(patches.Rectangle((0.0, 0.8), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database_2['Total_Reward'].sum(),".2f")
    axs[0,2].text(0.5, 0.9, 'Total Reward = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Total Time
    axs[0,2].add_patch(patches.Rectangle((0.0, 0.6), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database_2['Duration'].iloc[-1],".2f")
    axs[0,2].text(0.5, 0.7, 'Duration [sec] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Distace Traveled
    # Regen Ratio
    axs[0,2].add_patch(patches.Rectangle((0.0, 0.4), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(abs((database_2['dt_Energy_wh'][database_2['dt_Energy_wh'] < 0].sum()/database_2['dt_Energy_wh'][database_2['dt_Energy_wh'] > 0].sum())*100),".2f")
    axs[0,2].text(0.5, 0.5, 'Regen Ratio = {} %'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Average Speed, km/h
    axs[0,2].add_patch(patches.Rectangle((0.0, 0.2), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text = format(database_2['Pos'].iloc[-1]/database_2['Duration'].iloc[-1]*3.6,".2f")
    axs[0,2].text(0.5, 0.3, 'Average Speed [km/h] = {}'.format(formatted_text), ha='center', va='center', color='white', fontsize=12,fontweight='bold')
    # Reward distribution over time, energy and position
    axs[0,2].add_patch(patches.Rectangle((0.0, 0.0), 1.0, 0.2, facecolor='red', edgecolor='black'))
    formatted_text_1 = format(time_reward_percentage_2,".1f")
    formatted_text_2 = format(energy_reward_percentage_2,".1f")
    formatted_text_3 = format(position_reward_percentage_2,".1f")
    axs[0,2].text(0.5, 0.11, 'Reward Distr| Time, Energy, Position', ha='center', va='bottom', color='white', fontsize=12,fontweight='bold')
    axs[0,2].text(0.65, 0.09, '{}%, {}%, {}%'.format(formatted_text_1,formatted_text_2,formatted_text_3), ha='center', va='top', color='white', fontsize=12,fontweight='bold')
    del formatted_text, formatted_text_1, formatted_text_2, formatted_text_3

    # Plot the actions
    axs[1,2].plot(database_2['Pos'], database_2['act'])
    axs[1,2].set_xlabel('Position [m]')
    axs[1,2].set_ylabel('Action [-]')
    axs[1,2].set_title('Action Profile (Dummy)', fontsize=11, fontweight='bold')
    axs[1,2].grid()

    # Plot the Rewards
    axs[2,2].plot(database_2['Pos'], database_2['Total_Reward'], color='blue')
    axs[2,2].plot(database_2['Pos'], database_2['dt_Time_reward'], '--', color='red')
    axs[2,2].plot(database_2['Pos'], database_2['dt_Energy_reward'], '--', color='green')
    axs[2,2].plot(database_2['Pos'], database_2['dt_Position_reward'], '--', color='orange')
    axs[2,2].set_xlabel('Position [m]')
    axs[2,2].set_ylabel('Reward [-]')
    axs[2,2].set_title('Reward Profile (Dummy)', fontsize=11, fontweight='bold')
    axs[2,2].legend(['Total Reward','Time Reward','Energy Reward','Position Reward'], loc='upper left', fontsize=10)
    axs[2,2].grid()

    # Plot the Total Reward
    axs[3,2].plot(database_2['Pos'], database_2['Total_Reward'].cumsum())
    axs[3,2].set_xlabel('Position [m]')
    axs[3,2].set_ylabel('Total Reward [-]')
    axs[3,2].set_title('Total Reward (Dummy)', fontsize=11, fontweight='bold')
    axs[3,2].grid()

    # Plot the Torque and Power
    axs[4,2].plot(database_2['Pos'], database_2['Torque'], color='blue')
    axs[4,2].plot(database_2['Pos'], database_2['Power'], '--', color='red')
    axs[4,2].set_xlabel('Position [m]')
    axs[4,2].set_ylabel('Torque [Nm] / Power [W]')
    axs[4,2].set_title('Torque and Power Profile (Dummy)', fontsize=11, fontweight='bold')
    axs[4,2].legend(['Torque','Power'], loc='lower left', fontsize=10)
    axs[4,2].grid()

    # Plot Adjusting
    plot = plt.subplots_adjust(hspace=0.3, top=0.95)
    if show_fig == True: plt.show()
    formatted_text = format(total_reward,".2f")
    return total_reward, plot, fig

# Simulations Outputs Folder Creation
def create_folder():
    import datetime
    import os
    # Get the current time
    now = datetime.datetime.now()
    
    # Define the folder name
    folder_name = now.strftime("Simulation_Outputs__%Y-%m-%d_%H-%M-%S")
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Create the folder path
    folder_path = os.path.join(current_dir, "Simulation_Outputs", folder_name)
    # Create the folder
    os.makedirs(folder_path)

    return folder_path

#  Outputs Saving
def save_outputs(folder_path, database, episode_number, Score):
    
    # Get the current time
    now = datetime.datetime.now()
    Score = format(Score,".2f")
    # Define the figure name
    simulation_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    fig_name = f"Episode_{episode_number}__Score_{Score}__Date_{simulation_time}.png"
    database_name = f"Episode_{episode_number}_Database_.csv"
    database_name_2 = f"Episode_{episode_number}_Database_.df"
    # Save the figure
    full_path = os.path.join(folder_path, fig_name)
    print(full_path)
    plt.savefig(full_path)
    # Save the database
    full_path = os.path.join(folder_path, database_name)
    database.to_csv(full_path, index=False)
    full_path = os.path.join(folder_path, database_name_2)
    database.to_pickle(full_path)
    # Close the figure
    plt.close()


# Road Profiler Function: This function creates a road profile with given gradient and distance.
#def Road_Profiler(gradient, distance):
#        """This function creates a road profile with given gradient and distance.
#        Input: gradient [%], distance [m].
#        Output: A Road D; D.gradient, D.degree, D.radian, D.x, D.y, D.Distance.
#        ----------
#        How To Use
#        D1 = Road_Profiler(0, 200).
#        D2 = Road_Profiler(0.04, 200).
#        D3 = Road_Profiler(0, 200).
#        D4 = Road_Profiler(-0.04, 200).
#        D5 = Road_Profiler(0, 200).
#        """
#        D = {}
#        D['Gradient'] = [gradient] * (distance+1)
#        D['degree'] = math.atan(D['Gradient'][1]) * 180 / math.pi
#        D['radian'] = math.atan(D['Gradient'][1])
#        D['x']      = [i * math.cos(D['radian']) for i in range(distance+1)]
#        if D['radian'] == 0:
#            D['y'] = [0] * (distance+1)
#        else:
#            D['y'] = [i * math.sin(D['radian']) for i in range(distance+1)]
#        D['Distance'] = distance
#    
#        return D
#
## Road Merger Function: This function merges the road profiles.
#def Road_Merger(R):
#        """This function merges the road profiles.
#        Input: R = [D1, D2, D3, D4, D5, ...]
#        Output: Merged roads; Road.x, Road.y, Road.gradients, Road.Distances, Road.Total_Distance. 
#        ----------
#        """
#        Road = {'x': [], 'y': [], 'Gradient': [], 'Distances': [], 'Total_Distance': 0}
#
#        road_count = len(R)
#    
#        for i in range(road_count):
#            if i == 0:
#                Road['x'].extend(R[i]['x'])
#                Road['y'].extend(R[i]['y'])
#                Road['Gradient'].extend(R[i]['Gradient'])
#            else:
#                Road['x'].extend([Road['x'][-1] + x for x in R[i]['x'][1:]])
#                Road['y'].extend([Road['y'][-1] + y for y in R[i]['y'][1:]])
#                Road['Gradient'].extend(R[i]['Gradient'][1:])
#
#            Road['Distances'].append(R[i]['Distance'])
#            Road['Total_Distance'] += R[i]['Distance']
#    
#        return Road