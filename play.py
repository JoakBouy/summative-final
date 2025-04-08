import time
import numpy as np
from stable_baselines3 import DQN, PPO
from hospital_assistant_env import HospitalAssistantEnv, CellType
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import io
from PIL import Image

def print_grid(grid):
    """Pretty print the grid with symbols for better visualization"""
    symbols = {
        CellType.EMPTY: '.',
        CellType.AGENT: 'R',
        CellType.PATIENT: 'P',
        CellType.EMERGENCY: 'E',
        CellType.MEDICINE: 'M',
        CellType.OBSTACLE: '#',
        CellType.VISITED_PATIENT: 'p',
        CellType.VISITED_EMERGENCY: 'e',
        CellType.VISITED_MEDICINE: 'm',
    }
    for row in grid:
        print(''.join([symbols.get(cell, '?') for cell in row]))

def grid_to_rgb(grid):
    """Convert grid to RGB values for visualization"""
    # Define colors for each cell type
    colors = {
        CellType.EMPTY: [1.0, 1.0, 1.0],  # White
        CellType.AGENT: [0.0, 0.0, 1.0],  # Blue
        CellType.PATIENT: [1.0, 0.6, 0.6],  # Light Red
        CellType.EMERGENCY: [1.0, 0.0, 0.0],  # Red
        CellType.MEDICINE: [0.0, 0.8, 0.0],  # Green
        CellType.OBSTACLE: [0.5, 0.5, 0.5],  # Gray
        CellType.VISITED_PATIENT: [0.7, 0.7, 0.7],  # Light Gray
        CellType.VISITED_EMERGENCY: [0.7, 0.5, 0.5],  # Light Brown
        CellType.VISITED_MEDICINE: [0.5, 0.7, 0.5],  # Light Green
    }
    
    # Convert grid to RGB values
    height, width = grid.shape
    rgb_grid = np.zeros((height, width, 3))
    
    for i in range(height):
        for j in range(width):
            cell_type = grid[i, j]
            rgb_grid[i, j] = colors.get(cell_type, [0, 0, 0])
    
    return rgb_grid

def create_legend():
    """Create a legend for the visualization"""
    legend_elements = [
        mpatches.Patch(color=[0.0, 0.0, 1.0], label='Robot'),
        mpatches.Patch(color=[1.0, 0.6, 0.6], label='Patient'),
        mpatches.Patch(color=[1.0, 0.0, 0.0], label='Emergency'),
        mpatches.Patch(color=[0.0, 0.8, 0.0], label='Medicine'),
        mpatches.Patch(color=[0.5, 0.5, 0.5], label='Obstacle'),
        mpatches.Patch(color=[0.7, 0.7, 0.7], label='Visited Patient'),
        mpatches.Patch(color=[0.7, 0.5, 0.5], label='Visited Emergency'),
        mpatches.Patch(color=[0.5, 0.7, 0.5], label='Visited Medicine')
    ]
    return legend_elements

# Choose which model to run (uncomment the one you want to use)
model_type = "PPO"  # Options: "DQN" or "PPO"
model_path = "hospital_assistant_ppo_model"  # Change this path as needed

# Initialize the environment
env = HospitalAssistantEnv(grid_size=8, max_steps=200)

# Load the trained model
if model_type == "DQN":
    model = DQN.load(model_path)
else:  # PPO
    try:
        # Try loading normally first
        model = PPO.load(model_path)
    except TypeError:
        # If that fails, try with custom_objects to handle version differences
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "use_sde": False
        }
        model = PPO.load(model_path, custom_objects=custom_objects)

# Reset the environment for simulation
observation = env.reset()[0]
done = False
truncated = False
total_reward = 0
step_count = 0

print(f"Starting Hospital Assistant simulation using {model_type} model...\n")
print("Initial state:")
print_grid(observation["grid"])
print(f"Has medicine: {observation['inventory'][0]}")
print(f"Remaining steps: {observation['remaining_steps'][0]}")

# Action mapping for better understanding
action_names = ["Up", "Down", "Left", "Right", "Interact"]

# For GIF creation
frames = []
fig, ax = plt.subplots(figsize=(10, 8))

# Create the first frame
rgb_grid = grid_to_rgb(observation["grid"])
img = ax.imshow(rgb_grid)
title = ax.set_title(f"Step: 0 | Reward: 0 | Has Medicine: {bool(observation['inventory'][0])}")
legend_elements = create_legend()
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid lines for clarity
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2, alpha=0.3)
ax.set_xticks(np.arange(-.5, len(observation["grid"]), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(observation["grid"]), 1), minor=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
fig.tight_layout()

# Save the first frame
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
frames.append(Image.open(buf))

# Run the agent using the trained model (simulation loop)
for step in range(200):  # Run for a maximum of 100 steps
    step_count = step + 1
    if done or truncated:
        break
        
    # Get action from the trained model
    action, _states = model.predict(observation, deterministic=True)
    
    # Take the action in the environment
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    # Print status information
    print(f"\nStep {step + 1}:")
    print(f"Action: {action_names[action]} ({action})")
    print(f"Reward: {reward:.1f}, Total: {total_reward:.1f}")
    print(f"Has medicine: {observation['inventory'][0]}")
    print(f"Progress: {info['visited_patients']}/{len(env.patient_positions)} patients, "
          f"{info['visited_emergencies']}/{len(env.emergency_positions)} emergencies")
    
    # Render the environment
    env.render()
    
    # Update the plot for GIF
    rgb_grid = grid_to_rgb(observation["grid"])
    img.set_array(rgb_grid)
    title.set_text(f"Step: {step+1} | Total Reward: {total_reward:.1f} | Has Medicine: {bool(observation['inventory'][0])}")
    
    # Save this frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf))
    
    time.sleep(0.5)  # Slow down visualization

# Save the frames as GIF
output_file = f"hospital_assistant_{model_type}_simulation.gif"
frames[0].save(
    output_file,
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=200,  # milliseconds per frame
    loop=0  # 0 means loop indefinitely
)

# Final simulation summary
print("\n=== Simulation Complete ===")
print(f"Steps taken: {step_count}")
print(f"Final score: {total_reward:.1f}")
print(f"Simulation saved as: {output_file}")

if done and not truncated:
    print("SUCCESS: All patients and emergencies were handled efficiently!")
    message = "Mission Complete - Excellent Healthcare Service!"
elif truncated:
    print("TIMEOUT: Could not complete all tasks within the time limit.")
    message = "Shift Ended - Some Patients Still Waiting"
else:
    print("INCOMPLETE: Simulation ended before completing all tasks.")
    message = "Incomplete Rounds"

print(f"\nFinal message: {message}")
env.close()

# Close the matplotlib figure used for simulation
plt.close(fig)

