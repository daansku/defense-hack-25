import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import time

# Read data from JSON file
with open('server/nodes.json', 'r') as file:
    data = json.load(file)
    nodes = data['nodes']
    friendly_positions = data['friendly_positions']

# Extract coordinates and labels for nodes
x_coords_nodes = [node["coordinates"]["x"] for node in nodes]
y_coords_nodes = [node["coordinates"]["y"] for node in nodes]
labels_nodes = [node["location_name"] for node in nodes]

# Extract coordinates and labels for friendly positions
x_coords_friendly = [pos["coordinates"]["x"] for pos in friendly_positions]
y_coords_friendly = [pos["coordinates"]["y"] for pos in friendly_positions]
labels_friendly = [pos["name"] for pos in friendly_positions]

# Combine all coordinates for interaction
all_x = x_coords_nodes + x_coords_friendly
all_y = y_coords_nodes + y_coords_friendly
all_labels = labels_nodes + labels_friendly
num_nodes = len(x_coords_nodes)
num_friendly = len(x_coords_friendly)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.2)

# Initial colors
node_colors = ['skyblue'] * num_nodes
friendly_colors = ['limegreen'] * num_friendly

# Create scatter plots
scatter_nodes = ax.scatter(x_coords_nodes, y_coords_nodes, s=200, c=node_colors, 
                           edgecolors='navy', linewidths=2, alpha=0.7, label='Nodes')
scatter_friendly = ax.scatter(x_coords_friendly, y_coords_friendly, s=200, c=friendly_colors, 
                              edgecolors='darkgreen', linewidths=2, alpha=0.7, label='Friendly Positions')

ax.legend(loc='upper right')

# Add labels
annotations_nodes = []
for i, label in enumerate(labels_nodes):
    ann = ax.annotate(label, (x_coords_nodes[i], y_coords_nodes[i]), 
                     textcoords="offset points", xytext=(0,10), 
                     ha='center', fontsize=10, fontweight='bold')
    annotations_nodes.append(ann)

annotations_friendly = []
for i, label in enumerate(labels_friendly):
    ann = ax.annotate(label, (x_coords_friendly[i], y_coords_friendly[i]),
                     textcoords="offset points", xytext=(0,10),
                     ha='center', fontsize=10, fontweight='bold', color='green')
    annotations_friendly.append(ann)

# Customize the plot
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_title('Location Nodes Graph (Click points to select, then use buttons to change color)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.margins(0.2)

# Track selected points
selected_items = set()

# Available colors
color_options = ['skyblue', 'limegreen', 'lightcoral', 'gold', 'plum', 'orange', 'pink', 'cyan']

def on_click(event):
    """Handle click events on the plot to select points"""
    if event.inaxes == ax:
        # Find the closest point to the click
        distances = [(all_x[i] - event.xdata)**2 + (all_y[i] - event.ydata)**2 
                     for i in range(len(all_x))]
        closest_idx = np.argmin(distances) # int
        
        # Only select if click is close enough (within 0.3 units)
        if distances[closest_idx] < 0.3:
            if closest_idx in selected_items:
                selected_items.remove(closest_idx)
            else:
                selected_items.add(closest_idx)
            update_display()

def update_display():
    """Update the scatter plots with current colors and selection"""
    # Update node colors and edges
    node_edge_colors = ['red' if i in selected_items else 'navy' for i in range(num_nodes)]
    node_edge_widths = [3 if i in selected_items else 2 for i in range(num_nodes)]
    scatter_nodes.set_facecolors(node_colors)
    scatter_nodes.set_edgecolors(node_edge_colors)
    scatter_nodes.set_linewidths(node_edge_widths)
    
    # Update friendly position colors and edges
    friendly_edge_colors = ['red' if (i + num_nodes) in selected_items else 'darkgreen' 
                            for i in range(num_friendly)]
    friendly_edge_widths = [3 if (i + num_nodes) in selected_items else 2 
                           for i in range(num_friendly)]
    scatter_friendly.set_facecolors(friendly_colors)
    scatter_friendly.set_edgecolors(friendly_edge_colors)
    scatter_friendly.set_linewidths(friendly_edge_widths)
    
    fig.canvas.draw_idle()

def change_color_with_index(index):
    
    # Only select if click is close enough (within 0.3 units)
    if index in selected_items:
        selected_items.remove(index)
    else:
        selected_items.add(index)
    update_display()

# # Create color buttons
# button_axes = []
# buttons = []
# button_width = 0.08
# button_height = 0.04
# start_x = 0.1

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()

print("qweq")