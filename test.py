#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# The modified link to the Google Sheets CSV export
google_sheet_url = 'https://docs.google.com/spreadsheets/d/1e8xUe2WUkDsmTg_lQ3L9a0uAFG1sb3lPmLtFDDwdwts/export?format=csv&id=1e8xUe2WUkDsmTg_lQ3L9a0uAFG1sb3lPmLtFDDwdwts&gid=493058242'

# Load the data into a DataFrame directly from the CSV export link
player_stats = pd.read_csv(google_sheet_url)



# In[2]:


import panel as pn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os
from PIL import Image  # Add this import statement
import numpy as np
import requests
from io import BytesIO


# Initialize Panel extension
pn.extension()

# Title and Introduction
header = pn.pane.Markdown("## Player Statistics Dashboard presented by Minh Trinh, collaborated with Professor Eren Bilen")
intro = pn.pane.Markdown("""
Welcome to the Complex Tennis Player Statistics Interactive Dashboard!
This interactive dashboard illustrates a comprehensive overview of individual player statistics in Grand Slam 
events from 2019 to 2023. You can compare the selected player’s performance against the averages of other
elite and non-elite players, as well as visualize their serve distribution.""")
comparison_table_title = pn.pane.Markdown("""
### Comparison Table: Player vs Elite vs Non-Elite Statistics  
""")
# Sidebar Dropdown for Player Selection
selected_player_widget = pn.widgets.Select(
    name="Select Player", 
    options=list(player_stats['player_name'].unique()),  # Convert to list
    value=player_stats['player_name'].iloc[0],
    width=200,
    sizing_mode="stretch_width"
)

# Calculate Average Stats for Elite and Non-Elite Players
elite_avg_stats = player_stats[player_stats["elite_numeric"] == 1].mean(numeric_only=True)
non_elite_avg_stats = player_stats[player_stats["elite_numeric"] == 0].mean(numeric_only=True)

# Function to create comparison table
def create_comparison_table(selected_player):
    player_data = player_stats[player_stats["player_name"] == selected_player].iloc[0]
    
    comparison_table = pd.DataFrame({
        "Metric": [
            "Grand Slam Matches Win%","Avg 1st Serve Speed (KM/H)", "Avg 2nd Serve Speed (KM/H)", "Matches Played", 
            "Aces per Match", "Double Faults per Match",
            "Not Deep Return per Match", "Deep Return per Match", "1st Serve In per Match",
            "1st Serve Won per Match", "2nd Serve Won per Match", "Break Points Saved per Match",
            "Break Points Faced per Match", "Break Points Saved Percentage", "Deficit per Match",
            "Forehand Winners per Match", "Backhand Winners per Match", "0-4 Rallies Win %", "5-8 Rallies Win %"
            , "9+ Rallies Win %"
        ],
        f"{selected_player}'s Stats": [
            player_data["Match_Win_Percentage"],player_data["Avg_1st_ServeSpeed"], player_data["Avg_2nd_ServeSpeed"], player_data["match_played"],
            player_data["avg_ace"], player_data["avg_df"],
            player_data["ND_per_match"], player_data["D_per_match"], player_data["1stIn_per_match"],
            player_data["1stWon_per_match"], player_data["2ndWon_per_match"], player_data["bpSaved_per_match"],
            player_data["bpFaced_per_match"], player_data["bpSaved%"], player_data["avg_deficit"],
            player_data["avg_Forehand"], player_data["avg_Backhand"],player_data["0-4_%"],
            player_data["5-8_%"],player_data["9+_%"]
        ],
        "Elite Players' Average": [
            elite_avg_stats["Match_Win_Percentage"],elite_avg_stats["Avg_1st_ServeSpeed"], elite_avg_stats["Avg_2nd_ServeSpeed"], elite_avg_stats["match_played"],
            elite_avg_stats["avg_ace"], elite_avg_stats["avg_df"],
            elite_avg_stats["ND_per_match"], elite_avg_stats["D_per_match"], elite_avg_stats["1stIn_per_match"],
            elite_avg_stats["1stWon_per_match"], elite_avg_stats["2ndWon_per_match"], elite_avg_stats["bpSaved_per_match"],
            elite_avg_stats["bpFaced_per_match"], elite_avg_stats["bpSaved%"], elite_avg_stats["avg_deficit"],
            elite_avg_stats["avg_Forehand"], elite_avg_stats["avg_Backhand"],elite_avg_stats["0-4_%"],
            elite_avg_stats["5-8_%"],elite_avg_stats["9+_%"]
        ],
        "Non-Elite Players' Average": [
            non_elite_avg_stats["Match_Win_Percentage"],non_elite_avg_stats["Avg_1st_ServeSpeed"], non_elite_avg_stats["Avg_2nd_ServeSpeed"], non_elite_avg_stats["match_played"],
            non_elite_avg_stats["avg_ace"], non_elite_avg_stats["avg_df"],
            non_elite_avg_stats["ND_per_match"], non_elite_avg_stats["D_per_match"], non_elite_avg_stats["1stIn_per_match"],
            non_elite_avg_stats["1stWon_per_match"], non_elite_avg_stats["2ndWon_per_match"], non_elite_avg_stats["bpSaved_per_match"],
            non_elite_avg_stats["bpFaced_per_match"], non_elite_avg_stats["bpSaved%"], non_elite_avg_stats["avg_deficit"],
           non_elite_avg_stats["avg_Forehand"], non_elite_avg_stats["avg_Backhand"], non_elite_avg_stats["0-4_%"],
            non_elite_avg_stats["5-8_%"],non_elite_avg_stats["9+_%"]
        ]
    })
    
    # Ensure that values are numeric and format with 2 decimals without trailing zeros
    for col in comparison_table.columns[1:]:
        comparison_table[col] = comparison_table[col].apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))
    
    return comparison_table

def highlight_values(row):
    """
    Apply color formatting to highlight the highest and lowest values in a row.
    Green for the largest value, red for the smallest value.

    Args:
        row (pd.Series): A row of numeric values.

    Returns:
        list: A list of CSS styles for each value in the row.
    """
    # Convert row to numeric values, ignoring non-numeric values
    numeric_values = pd.to_numeric(row, errors='coerce')

    # Identify max and min values (ignoring NaNs)
    max_val = numeric_values.max()
    min_val = numeric_values.min()

    # Apply colors based on max and min values
    colors = [
        "color: green;" if value == max_val else
        "color: red;" if value == min_val else
        ""
        for value in numeric_values
    ]
    return colors
# Function to update the dashboard dynamically
def update_dashboard(event):
    selected_player = selected_player_widget.value
    comparison_table = create_comparison_table(selected_player)
    
    # Apply styling to the table
    styled_table = (
        comparison_table.set_index("Metric")
        .style.apply(highlight_values, axis=1, subset=[f"{selected_player}'s Stats", "Elite Players' Average", "Non-Elite Players' Average"])
    )
    
    comparison_table_pane.object = styled_table

# Create initial comparison table for the first player
comparison_table_pane = pn.pane.DataFrame(create_comparison_table(selected_player_widget.value), width=600, height=200)

# Watch the player selection to update the table
selected_player_widget.param.watch(update_dashboard, 'value')


# Watch the player selection to update the table
selected_player_widget.param.watch(update_dashboard, 'value')


####################################################################################################################

# Define court regions
regions_ctl = {
    'W': {'x_min': 14, 'x_max': 16.25, 'y_min': 13, 'y_max': 18, 'color': '#d7d7d7'},
    'BW': {'x_min': 16.25, 'x_max': 18.5, 'y_min': 13, 'y_max': 18, 'color': '#1f1f1f'},
    'B': {'x_min': 18.5, 'x_max': 20.75, 'y_min': 13, 'y_max': 18, 'color': '#141414'},
    'BC': {'x_min': 20.75, 'x_max': 23, 'y_min': 13, 'y_max': 18, 'color': '#2d2d2d'},
    'C': {'x_min': 23, 'x_max': 25, 'y_min': 13, 'y_max': 18, 'color': '#484848'}
}
regions_nctl = {
    'W': {'x_min': 14, 'x_max': 16.25, 'y_min': 18, 'y_max': 23, 'color': '#4d4d4d'},
    'BW': {'x_min': 16.25, 'x_max': 18.5, 'y_min': 18, 'y_max': 23, 'color': '#a7a7a7'},
    'B': {'x_min': 18.5, 'x_max': 20.75, 'y_min': 18, 'y_max': 23, 'color': '#636363'},
    'BC': {'x_min': 20.75, 'x_max': 23, 'y_min': 18, 'y_max': 23, 'color': '#888888'},
    'C': {'x_min': 23, 'x_max': 25, 'y_min': 18, 'y_max': 23, 'color': '#ffffff'}
}

# Define label positions
label_positions = {
    'NCTL_C': {'x': 45, 'y': 35},
    'NCTL_BC': {'x': 40, 'y': 45},
    'NCTL_B': {'x': 25, 'y': 45},
    'NCTL_BW': {'x': 15, 'y': 35},
    'NCTL_W': {'x': 5, 'y': 45},
    'CTL_C': {'x': 45, 'y': 15},
    'CTL_BC': {'x': 42, 'y': 5},
    'CTL_B': {'x': 22, 'y': 7},
    'CTL_BW': {'x': 7, 'y': 5},
    'CTL_W': {'x': 5, 'y': 15}
}

# Function to prepare serve data for a player
def prepare_player_serve_data(player_name):
    player_data = player_stats[player_stats['player_name'] == player_name]
    serve_data = []
    for col in player_data.columns:
        if col.startswith("CTL") or col.startswith("NCTL"):
            serve_depth, serve_width = col.split("_")
            count = player_data[col].values[0]
            serve_data.append({
                "ServeDepth": serve_depth,
                "ServeWidth": serve_width,
                "count": count
            })
    serve_df = pd.DataFrame(serve_data)
    total_shots = serve_df['count'].sum()
    serve_df['percentage'] = (serve_df['count'] / total_shots) * 100
    return serve_df

# Function to add labels
def add_labels_for_regions(data_counts, label_positions, region_type, regions):
    for _, row in data_counts.iterrows():
        serve_width = row['ServeWidth']
        count = row['count']
        percentage = row['percentage']

        # Construct the key dynamically
        label_key = f"{region_type}_{serve_width}"

        # Check if label_key exists in label_positions
        if label_key in label_positions:
            x = label_positions[label_key]['x']
            y = label_positions[label_key]['y']
        else:
            raise ValueError(f"Label position for '{label_key}' not defined in label_positions.")

        # Label content
        label = f"{region_type} {serve_width}\n{count} shots\n({percentage:.2f}%)"
        
        # Draw line to region center
        region_center_x = (regions[serve_width]['x_min'] + regions[serve_width]['x_max']) / 2
        region_center_y = (regions[serve_width]['y_min'] + regions[serve_width]['y_max']) / 2
        plt.plot([x, region_center_x], [y, region_center_y], color='white', lw=1)

        # Add label background box
        label_box = patches.FancyBboxPatch(
            (x - 6, y - 3.5), 12, 7, boxstyle="round,pad=0.3",
            edgecolor='none', facecolor=regions[serve_width]['color'], alpha=0.6
        )
        plt.gca().add_patch(label_box)
        
        # Add text
        plt.text(x, y, label, fontsize=9, ha='center', va='center', color='black')

# Function to dynamically assign colors based on serve percentages
def assign_dynamic_colors(regions, data_counts):
    """
    Assign colors to regions dynamically based on serve percentages, using a seismic color scale.

    Args:
        regions (dict): Dictionary of regions (CTL or NCTL).
        data_counts (pd.DataFrame): Serve data with percentage for each region.

    Returns:
        dict: Updated regions with dynamically assigned colors.
    """
    # Normalize percentages between -1 and 1 for seismic color mapping
    percentages = data_counts['percentage'].values
    normalized = 2 * ((percentages - percentages.min()) / (percentages.max() - percentages.min())) - 1 if len(percentages) > 1 else [0] * len(percentages)

    # Map normalized values to the seismic color scale
    for region, norm_val in zip(data_counts['ServeWidth'], normalized):
        if norm_val < 0:
            # Blue to white (low values): interpolate between Blue (#0000FF) and White (#FFFFFF)
            r = int(255 * (1 + norm_val))  # Increase red
            g = int(255 * (1 + norm_val))  # Increase green
            b = 255  # Blue remains max
        else:
            # White to red (high values): interpolate between White (#FFFFFF) and Red (#FF0000)
            r = 255  # Red remains max
            g = int(255 * (1 - norm_val))  # Decrease green
            b = int(255 * (1 - norm_val))  # Decrease blue

        color = f'#{r:02x}{g:02x}{b:02x}'  # Seismic hex color
        regions[region]['color'] = color

    return regions



def load_court_image(drive_url):
    try:
        # Extract file ID from the URL
        file_id = drive_url.split('/d/')[1].split('/')[0]
        
        # Create the download URL
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Send HTTP request to download the image
        response = requests.get(download_url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Open the image from the downloaded content
        court_img = Image.open(BytesIO(response.content))
        return np.array(court_img)  # Convert to NumPy array for plotting
    except Exception as e:
        print(f"Error loading image: {e}. Using a white placeholder.")
        return np.ones((500, 500, 3))  # Return white placeholder if an error occurs

# Example usage:
image_url = 'https://drive.google.com/file/d/1vZddjW1NgZpcwJrb5f_llqRaIGBS23cS/view?usp=sharing'
court_img = load_court_image(image_url)



def plot_combined_serve_distribution(player_name):
    """
    Plot the combined serve distribution for the selected player.
    Args:
        player_name (str): Name of the selected player.
    Returns:
        matplotlib.figure.Figure: The plot figure for Panel rendering.
    """
    # Prepare serve data
    serve_df = prepare_player_serve_data(player_name)

    # Separate CTL and NCTL data
    ctl_data = serve_df[serve_df['ServeDepth'] == 'CTL']
    nctl_data = serve_df[serve_df['ServeDepth'] == 'NCTL']

    # Assign dynamic colors based on percentages
    assign_dynamic_colors(regions_ctl, ctl_data)
    assign_dynamic_colors(regions_nctl, nctl_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    court_img = load_court_image(image_url)
    ax.imshow(court_img, extent=[0, 50, 0, 50])
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)

    # Plot CTL regions
    for region, data in regions_ctl.items():
        ax.add_patch(patches.Rectangle(
            (data['x_min'], data['y_min']),
            data['x_max'] - data['x_min'], 
            data['y_max'] - data['y_min'], 
            linewidth=1, edgecolor='black', facecolor=data['color'], alpha=0.5
        ))

    # Plot NCTL regions
    for region, data in regions_nctl.items():
        ax.add_patch(patches.Rectangle(
            (data['x_min'], data['y_min']),
            data['x_max'] - data['x_min'], 
            data['y_max'] - data['y_min'], 
            linewidth=1, edgecolor='black', facecolor=data['color'], alpha=0.5
        ))

    # Add labels for CTL and NCTL regions
    add_labels_for_regions(ctl_data, label_positions, 'CTL', regions_ctl)
    add_labels_for_regions(nctl_data, label_positions, 'NCTL', regions_nctl)

    # Titles and axis labels
    ax.set_title(f'Serve Distribution by {player_name}', fontsize=18, weight='bold')
    ax.set_xlabel('Serve Width', fontsize=15)
    ax.set_ylabel('Serve Depth', fontsize=15, rotation=0, labelpad=40)
    return fig

# Updated: Panel integration for serve distribution
serve_distribution_pane = pn.pane.Matplotlib(
    plot_combined_serve_distribution(selected_player_widget.value), 
    width=800, height=600
)

# Updated: Watch player selection for serve graph
def update_serve_graph(event):
    serve_distribution_pane.object = plot_combined_serve_distribution(selected_player_widget.value)

selected_player_widget.param.watch(update_serve_graph, 'value')

# Updated: Create comparison table dynamically
comparison_table_pane = pn.pane.DataFrame(
    create_comparison_table(selected_player_widget.value), 
    width=600, height=400
)

# Updated: Dynamic dashboard layout
template = pn.template.FastListTemplate(
    title='Complex Professional Tennis Player Statistics in Grand Slams from 2019-2023', 
    sidebar=[
        pn.pane.Markdown("# Complex Player Statistics Interactive Dashboard"), 
        pn.pane.Markdown("### This dashboard is presented by Minh Trinh in collaboration with professor Eren Bilen. \n Contact: quangminh711@gmail.com"),
        intro,
        selected_player_widget
    ],
    main=[
        pn.Row(
            pn.Column(width=150),
            pn.Column(comparison_table_title, comparison_table_pane, margin=(0, 25), width=300)
        ),
        pn.Row(
            serve_distribution_pane
        )
    ],
    accent_base_color="#90EE90",  
    header_background="#90EE90", 
    sidebar_background="#f0f0f0",
    )


template.servable()
template.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script test.ipynb')


# In[ ]:




