import re
import matplotlib.pyplot as plt
import numpy as np

# Path to the DSE results file
dse_file_path = 'results.txt'

# Clock speed (940)
clock = 940

# Regular expression pattern to extract relevant data
pattern = re.compile(r"(\w+)\s\(([^)]+)\)\s->\sCycles\s(\d+)")

# This function parses the DSE file and calculates latencies
def parse_and_get_cocos_data(file_path):
    cocos_data = []

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                op = match.group(1)
                dims_str = match.group(2)
                cycles = int(match.group(3))

                # Calculate the latency in µs
                latency = cycles / clock

                # Append the cycle value to cocos_data (this could be latency or cycles depending on use case)
                cocos_data.append(latency)  # Here we are appending cycles; modify if latencies are required

    return cocos_data

# This function is the original COCOSSim data extraction function
def extract_cocos_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Assuming the cycle data is in a specific format
    cocos_data = []
    for line in lines:
        if line.startswith('COCOSSim'):
            parts = line.strip().split()
            cocos_data.append(float(parts[1]))  # Assuming the cycle data is the second part of the line
    return cocos_data

# This function writes the updated data back to the DSE file
def write_updated_data_to_file(file_path, updated_data):
    with open(file_path, 'w') as file:
        file.writelines(updated_data)

# Main execution
cocos_data = parse_and_get_cocos_data(dse_file_path)


# Data from your DSE results and the TPU performance
tpu_data = [92.8, 74, 86, 41.6, 944, 55, 42.4, 2.2, 238, 142, 168, 89, 665, 43, 4.022, 1.08, 515.2, 324.4, 327.2, 203.6, 2536, 255, 161, 82]

# Operations for labeling the x-axis
categories = [
    "Matmul (4096, 320, 320)", "Matmul (1024, 640, 640)", "Matmul (256, 1280, 1280)", "Matmul (64, 1280, 1280)",
    "DotProduct (8, 4096, 40, 4096)", "DotProduct (8, 1024, 80, 1024)", "DotProduct (8, 256, 160, 256)", "DotProduct (8, 64, 160, 64)",
    "Conv (4096, 2880, 320)", "Conv (1024, 5760, 640)", "Conv (256, 11520, 1280)", "Conv (64, 11520, 1280)",
    "Softmax (8, 4096)", "Softmax (8, 1024)", "Softmax (8, 256)", "Softmax (8, 64)",
    "ResNet (4096, 2880, 320)", "ResNet (1024, 5760, 640)", "ResNet (256, 11520, 1280)", "ResNet (64, 11520, 1280)",
    "MultiHeadSelfAttention (4096, 320, 40)", "MultiHeadSelfAttention (1024, 640, 80)", "MultiHeadSelfAttention (256, 1280, 160)", "MultiHeadSelfAttention (64, 1280, 160)"
]

# Create the figure and axis for the bar plot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

ax = axes[0]
ax_acc = axes[1]

width = 0.35  # Bar width
x = np.arange(len(categories))  # the label locations

# Colors for bars
colors = {'tpu': '#008969', 'cocos': '#63cc3d'}

# Bar plot for cycles vs TPU performance
bars = {}
fig.subplots_adjust(wspace=0, hspace=0.5)

for i in range(len(categories)):
    # Plot cycle data
    ax.bar(x[i] + width / 2, cocos_data[i], width, label="COCOSSim", color=colors['cocos'], edgecolor='black', linewidth=1.5)
    
    # Plot TPU data
    ax.bar(x[i] - width / 2, tpu_data[i], width, label="TPU v3", color=colors['tpu'], edgecolor='black', linewidth=1.5)

    # Plot relative error for accuracy plot
    ax_acc.bar(x[i] + width / 2, 100 * abs(cocos_data[i] - tpu_data[i]) / tpu_data[i], width, color=colors['cocos'], edgecolor='black', linewidth=1.5)
    ax_acc.bar(x[i] - width / 2, 100 * abs(tpu_data[i] - tpu_data[i]) / tpu_data[i], width, color=colors['tpu'], edgecolor='black', linewidth=1.5)

# Set the labels, title, and legend
ax.set_ylabel('Latency (us)')
ax_acc.set_ylabel('Relative Error (%)')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=90, ha='right')
ax_acc.set_yticks([0, 25, 50, 75, 100])
ax_acc.set_xticks(x)

# Legends and plot details
ax.legend(['COCOSSim', 'TPU V3'], loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)

ax_acc.tick_params(bottom=False)

# Display the plot
plt.tight_layout()
plt.savefig('cocossim_vs_tpu_comparison.png', bbox_inches='tight', pad_inches=0.05)
plt.show()
