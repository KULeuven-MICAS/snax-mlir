import argparse
import os
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_and_prepare_data(input_file):
    """
    Parses the VCD file and prepares the DataFrame with necessary columns.
    """
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    df = pd.read_csv(input_file)

    # Initialize new columns for 'fire', 'stall', and 'bank' for each port
    for port in range(56):
        valid_col = f'io_data_tcdm_req_{port}_valid'
        ready_col = f'io_data_tcdm_req_{port}_ready'
        addr_col = f'io_data_tcdm_req_{port}_bits_addr'
        fire_col = f'io_data_tcdm_req_{port}_fire'
        stall_col = f'io_data_tcdm_req_{port}_stall'
        bank_col = f'io_data_tcdm_req_{port}_bank'

        df[fire_col] = (df[valid_col] == 1) & (df[ready_col] == 1)
        df[stall_col] = (df[valid_col] == 1) & (df[ready_col] == 0)
        df[bank_col] = df[addr_col] // 8 % 32

    return df


def plot_stalls_and_fires_per_port(df, output_path):
    """
    Creates a bar plot of the number of stalls and fires per port.
    """
    ports = np.arange(56)
    number_of_stalls = np.zeros(56)
    number_of_fires = np.zeros(56)

    for port in range(56):
        stall_col = f'io_data_tcdm_req_{port}_stall'
        fire_col = f'io_data_tcdm_req_{port}_fire'

        number_of_stalls[port] = df[stall_col].sum()
        number_of_fires[port] = df[fire_col].sum()

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    bar_width = 0.35
    indices = np.arange(len(ports))

    # Create bars for stalls and fires
    ax.bar(indices, number_of_stalls, bar_width, label='Number of Stalls')
    ax.bar(indices + bar_width, number_of_fires, bar_width, label='Number of Fires')

    # Labeling and aesthetics
    ax.set_xlabel('Port Number', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Number of Stalls and Fires per Port of GeMMX', fontsize=16)
    ax.set_xticks(indices + bar_width / 2)
    ax.set_xticklabels(ports, rotation=90)
    ax.legend()

    # Annotate port groups
    group_positions = [
        (0, 7, 'A'),      # Ports 0-7
        (8, 15, 'B'),     # Ports 8-15
        (16, 23, 'D8'),   # Ports 16-23
        (24, 55, 'C/D32') # Ports 24-55
    ]

    # Add annotations
    for start, end, label in group_positions:
        center = (indices[start] + indices[end] + bar_width) / 2
        ax.hlines(-8, indices[start], indices[end] + bar_width, color='black', linewidth=2)
        ax.text(center, -11, label, ha='center', va='top', fontsize=12)

    # Adjust plot limits
    ax.set_ylim(bottom=-20)
    ax.spines['bottom'].set_position(('data', 0))
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_path, 'nb_of_stalls_per_port.png')
    plt.savefig(output_file)
    plt.close()


def plot_stalls_and_fires_per_bank(df, output_path):
    """
    Creates a bar plot of the number of stalls and fires per bank.
    """
    banks = np.arange(32)
    number_of_stalls = np.zeros(32)
    number_of_fires = np.zeros(32)

    for bank_nb in range(32):
        nb_stalls = 0
        nb_fires = 0
        for port in range(56):
            stall_col = f'io_data_tcdm_req_{port}_stall'
            fire_col = f'io_data_tcdm_req_{port}_fire'
            bank_col = f'io_data_tcdm_req_{port}_bank'

            stalls = df[(df[stall_col]) & (df[bank_col] == bank_nb)]
            fires = df[(df[fire_col]) & (df[bank_col] == bank_nb)]

            nb_stalls += stalls.shape[0]
            nb_fires += fires.shape[0]

        number_of_stalls[bank_nb] = nb_stalls
        number_of_fires[bank_nb] = nb_fires

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 8))
    bar_width = 0.35
    indices = np.arange(len(banks))

    # Create bars for stalls and fires
    ax.bar(indices, number_of_stalls, bar_width, label='Number of Stalls')
    ax.bar(indices + bar_width, number_of_fires, bar_width, label='Number of Fires')

    # Labeling and aesthetics
    ax.set_xlabel('Bank Number', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Number of Stalls and Fires per Bank of TCDM', fontsize=16)
    ax.set_xticks(indices + bar_width / 2)
    ax.set_xticklabels(banks, rotation=90)
    ax.legend()

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_path, 'nb_of_stalls_per_bank.png')
    plt.savefig(output_file)
    plt.close()


def plot_banking_conflicts(df, output_path):
    """
    Creates a plot showing bank fires and stalls over time.
    """
    # num_cycles = 140
    # start_cycle = 3650
    num_banks = 32
    events = []

    # Map ports to operands
    for port in range(56):
        if port < 8:
            operand = 'A'
        elif port < 16:
            operand = 'B'
        elif port < 24:
            operand = 'F'
        else:
            operand = 'O'

        stall_col = f'io_data_tcdm_req_{port}_stall'
        fire_col = f'io_data_tcdm_req_{port}_fire'
        bank_col = f'io_data_tcdm_req_{port}_bank'

        stalls = df[df[stall_col]]
        fires = df[df[fire_col]]

        for cc, b in stalls[bank_col].items():
            event = {
                'Time': cc,
                'Operand': operand,
                'Bank': int(b) + 1,  # Banks numbered from 1
                'EventType': 'stall'
            }
            events.append(event)

        for cc, b in fires[bank_col].items():
            event = {
                'Time': cc,
                'Operand': operand,
                'Bank': int(b) + 1,
                'EventType': 'fire'
            }
            events.append(event)

    if not events:
        print("No events found for plotting banking conflicts.")
        return

    df_events = pd.DataFrame(events)

    # Determine start and end cycles
    min_cycle = df_events['Time'].min()
    max_cycle = df_events['Time'].max()
    start_cycle = max(0, min_cycle - 10)
    end_cycle = max_cycle + 10
    num_cycles = end_cycle - start_cycle + 1  # +1 to include end_cycle

    # Plotting
    fig, ax = plt.subplots(figsize=(num_banks / 2, num_cycles / 4))

    # Plot settings
    ax.set_xlabel('Bank Number', fontsize=12)
    ax.set_ylabel('Clock Cycle', fontsize=12)
    ax.set_title('Bank Fires/Stalls Over Time', fontsize=14)
    ax.set_xlim(0.5, num_banks + 0.5)
    ax.set_ylim(0.5, num_cycles + 0.5)
    ax.set_xticks(range(1, num_banks + 1))
    ax.set_yticks(range(1, num_cycles + 1))
    ax.invert_yaxis()

    # Set minor ticks
    minor_ticks_y = np.arange(1, num_cycles) + 0.5
    minor_ticks_x = np.arange(1, num_banks) + 0.5
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.set_xticks(minor_ticks_x, minor=True)

    # Enable gridlines
    ax.grid(which='minor', axis='y', linestyle='-', linewidth=1)
    ax.grid(which='minor', axis='x', linestyle='-', linewidth=1)

    # Operand offsets for plotting
    operand_offsets = {
        'A': (-0.15, -0.2),
        'B': (0.15, -0.2),
        'F': (-0.15, 0.25),
        'O': (0.15, 0.25)
    }

    # Plot events
    for _, event in df_events.iterrows():
        x = event['Bank']
        y = event['Time']
        operand = event['Operand']
        event_type = event['EventType']

        color = 'green' if event_type == 'fire' else 'red'
        assert isinstance(operand, str)
        dx, dy = operand_offsets.get(operand, (0, 0))

        ax.text(
            x + dx, y + dy, operand, color=color,
            fontsize=10, ha='center', va='center'
        )

    ax.set_aspect('auto')
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_path, 'banking_conflicts.pdf')
    plt.savefig(output_file)
    plt.close()


def main():
    """
    Main function to parse arguments and execute the plotting functions.
    """
    parser = argparse.ArgumentParser(description='Process CSV file and generate plots.')
    parser.add_argument('--input_file', type=str, required=True, help='Input .csv file')
    parser.add_argument('--output_path', type=str, default='.', help='Output directory for plots')
    args = parser.parse_args()

    df = parse_and_prepare_data(args.input_file)

    plot_stalls_and_fires_per_port(df, args.output_path)
    plot_stalls_and_fires_per_bank(df, args.output_path)
    plot_banking_conflicts(df, args.output_path)


if __name__ == '__main__':
    main()
