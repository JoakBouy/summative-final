import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from hospital_assistant_env import HospitalAssistantEnv

# Create the environment
env = HospitalAssistantEnv(grid_size=8, max_steps=100)

# Optional: Check if the environment follows Gymnasium's API
check_env(env)

# Define the DQN model
model = DQN(
    "MultiInputPolicy",  # Policy for dictionary observations
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.2,  # Fraction of entire training period over which exploration rate is reduced
    exploration_initial_eps=1.0,  # Initial value of random action probability
    exploration_final_eps=0.05,  # Final value of random action probability
    target_update_interval=100,  # Update frequency of the target network
    tensorboard_log="./hospital_assistant_dqn_tensorboard/",  # Log directory for TensorBoard
    verbose=1,
)

# Train the model
print("Training the DQN model...")
model.learn(total_timesteps=200000)
print("DQN training complete!")

# Save the model
model_path = "hospital_assistant_dqn_model"
model.save(model_path)
print(f"Model saved as '{model_path}'")

# Close the environment
env.close()