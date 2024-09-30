from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.vcd import parse_vcd_to_df

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

df = parse_vcd_to_df('sim3.vcd')
#df.to_csv('sim3.csv')
#df = pd.read_csv('sim3.csv')

valid = lambda port: f'io_data_tcdm_req_{port}_valid'
ready = lambda port: f'io_data_tcdm_req_{port}_ready'
data  = lambda port: f'io_data_tcdm_req_{port}_bits_data'
addr  = lambda port: f'io_data_tcdm_req_{port}_bits_addr'
bank  = lambda port: f'io_data_tcdm_req_{port}_bank'
fire  = lambda port: f'io_data_tcdm_req_{port}_fire'
stall = lambda port: f'io_data_tcdm_req_{port}_stall'

new_columns = [(fire(port), stall(port)) for port in range(56)]
new_columns = [a for tup in new_columns for a in tup]

# add new columns to df
df[new_columns] = False


for port in range(56):

    # annotate df with fire, valid, and bank nb

    df[fire(port)] = (df[valid(port)] == 1) & (df[ready(port)] == 1)
    df[stall(port)] = (df[valid(port)] == 1) & (df[ready(port)] == 0)
    df[bank(port)] = df[addr(port)] // 8 % 32


# create first graph: bar plot of nb of fires/stalls per tcdm req port

# Sample data for demonstration
# Replace these with your actual data
ports = np.arange(0, 56)  # Port numbers from 1 to 56

# Generate random sample data
number_of_stalls = np.zeros(shape=(56,))
number_of_fires = np.zeros(shape=(56,))

for port in range(56):
    if True in df[stall(port)].value_counts():
        nb_stalls = df[stall(port)].value_counts()[True]
    else:
        nb_stalls = 0
    if True in df[fire(port)].value_counts():
        nb_fires = df[fire(port)].value_counts()[True]
    else:
        nb_fires = 0
    number_of_stalls[port] = nb_stalls
    number_of_fires[port] = nb_fires

# Plotting
fig, ax = plt.subplots(figsize=(18, 8))

bar_width = 0.35
indices = np.arange(len(ports))
# Create bars for stalls and fires
bars_stalls = ax.bar(indices, number_of_stalls, bar_width, label='Number of Stalls')
bars_fires = ax.bar(indices + bar_width, number_of_fires, bar_width, label='Number of Fires')

# Labeling and aesthetics
ax.set_xlabel('Port Number', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Number of Stalls and Fires per Port of GeMMX', fontsize=16)
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels(ports, rotation=90)
ax.legend()

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Annotate port groups
# Define the positions and labels for the groups
group_positions = [(0, 7, 'A'),    # Ports 1-8
                   (8, 15, 'B'),   # Ports 9-16
                   (16, 23, 'D8'), # Ports 17-24
                   (24, 55, 'C/D32')]  # Ports 25-56

# Add annotations
for start, end, label in group_positions:
    # Calculate the center position for the label
    center = (indices[start] + indices[end] + bar_width) / 2
    # Add a horizontal line to demarcate the group (optional)
    ax.hlines(-8, indices[start], indices[end] + bar_width, color='black', linewidth=2)
    # Add the group label below the x-axis labels
    ax.text(center, -11, label, ha='center', va='top', fontsize=12)

# Adjust the plot limits to make space for annotations
ax.set_ylim(bottom=-20)

# Hide the part of the plot below y=0 (if desired)
ax.spines['bottom'].set_position(('data', 0))

# Display the plot
plt.savefig('nb_of_stalls_per_port.png')


# create second graph: bar plot of nb of fires/stalls per tcdm bank

# Sample data for demonstration
# Replace these with your actual data
ports = np.arange(0, 32)  # Bank numbers from 1 to 32

# Generate random sample data
number_of_stalls = np.zeros(shape=(32,))
number_of_fires = np.zeros(shape=(32,))

for bank_nb in range(32):
    nb_stalls = 0
    nb_fires = 0
    for port in range(56):
        if True in df[stall(port)][df[bank(port)] == bank_nb].value_counts():
            nb_stalls += df[stall(port)][df[bank(port)] == bank_nb].value_counts()[True]
        if True in df[fire(port)][df[bank(port)] == bank_nb].value_counts():
            nb_fires += df[fire(port)][df[bank(port)] == bank_nb].value_counts()[True]
    number_of_stalls[bank_nb] = nb_stalls
    number_of_fires[bank_nb] = nb_fires

# Plotting
fig, ax = plt.subplots(figsize=(18, 8))

bar_width = 0.35
indices = np.arange(len(ports))
# Create bars for stalls and fires
bars_stalls = ax.bar(indices, number_of_stalls, bar_width, label='Number of Stalls')
bars_fires = ax.bar(indices + bar_width, number_of_fires, bar_width, label='Number of Fires')

# Labeling and aesthetics
ax.set_xlabel('Bank Number', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Number of Stalls and Fires per bank of tcdm', fontsize=16)
ax.set_xticks(indices + bar_width / 2)
ax.set_xticklabels(ports, rotation=90)
ax.legend()

# Display the plot
plt.savefig('nb_of_stalls_per_bank.png')


# third plot
#
# determine start cycle and end cycle



#
num_cycles = 140
start_cycle = 3650
num_banks = 32
operands = ['A', 'B', 'F', 'C']
event_types = ['fire', 'stall']

events = []

# Fill up the events
for port in range(56):
    if port < 8:
        operand = 'A'
    elif port < 16:
        operand = 'B'
    elif port < 24:
        operand = 'F'
    else:
        operand = 'O'
    stalls = df[start_cycle : start_cycle + num_cycles][df[stall(port)] == 1][bank(port)]
    fires = df[start_cycle : start_cycle + num_cycles][df[fire(port)] == 1][bank(port)]
    for cc, b in stalls.items():
        event = {
            'Time': cc - start_cycle,
            'Operand': operand,
            'Bank': int(b) + 1,  # Banks numbered from 1 to num_banks
            'EventType': 'stall'
        }
        events.append(event)
    for cc, b in fires.items():
        event = {
            'Time': cc - start_cycle,
            'Operand': operand,
            'Bank': int(b) + 1,  # Banks numbered from 1 to num_banks
            'EventType': 'fire'
        }
        events.append(event)


df_events = pd.DataFrame(events)

fig, ax = plt.subplots(figsize=(num_banks/2, num_cycles/4))

# Plot settings
ax.set_xlabel('Bank Number', fontsize=12)
ax.set_ylabel('Clock Cycle', fontsize=12)
ax.set_title('Bank Fires/Stalls Over Time', fontsize=14)
ax.set_xlim(0.5, num_banks + 0.5)
ax.set_ylim(0.5, num_cycles + 0.5)
ax.set_xticks(range(1, num_banks + 1))
ax.set_yticks(range(1, num_cycles + 1))
ax.invert_yaxis()  # Invert y-axis so that time increases downward

# Set minor ticks at positions between cycles
minor_ticks_y = np.arange(1, num_cycles) + 0.5
ax.set_yticks(minor_ticks_y, minor=True)


# Set minor ticks at positions between cycles
minor_ticks_x = np.arange(1, num_banks) + 0.5
ax.set_xticks(minor_ticks_x, minor=True)

# Enable gridlines at minor ticks on y-axis (between cycles)
ax.grid(which='minor', axis='y', linestyle='-', linewidth=1)
# Disable gridlines at major ticks on y-axis
ax.grid(which='major', axis='y', linestyle='')

# Enable gridlines at major ticks on x-axis (banks)
ax.grid(which='minor', axis='x', linestyle='-', linewidth=1)
ax.grid(which='major', axis='x', linestyle='')

# Add grid lines
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


operand_offsets = {
    'A': (-0.15, -0.2),
    'B': (0.15, -0.2),
    'F': (-0.15, 0.25),
    'C': (0.15, 0.25)
}


operand_markers = {'A': 'o', 'B': 's', 'D8': '^', 'D32': 'D'}

# Plot events
for idx, event in df_events.iterrows():
    x = event['Bank']
    y = event['Time']
    operand = event['Operand']
    event_type = event['EventType']
    # Choose color based on event type
    color = 'green' if event_type == 'fire' else 'red'
    # Plot the operand letter at the (x, y) position
    assert isinstance(operand, str)
    dx, dy = operand_offsets.get(operand, (0, 0))
    ax.text(x + dx, y + dy, operand, color=color, fontsize=10, ha='center', va='center')
    #marker = operand_markers.get(operand, 'o')
    #ax.plot(x + dx, y + dy, marker=marker, color=color, markersize=8, linestyle='None')

# Adjust plot limits and aspect
ax.set_aspect('auto')
plt.tight_layout()

# Display the plot
plt.savefig('banking_conflicts.pdf')
