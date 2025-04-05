import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from hospital_assistant_env import HospitalAssistantEnv

# Create the environment
def make_env():
    return HospitalAssistantEnv(grid_size=8, max_steps=100)

# Optional: Check if the environment follows Gymnasium's API
check_env(make_env())

# Vectorize the environment for PPO (which works better with vectorized environments)
env = DummyVecEnv([make_env])
env = VecMonitor(env, "hospital_assistant_logs/")

# Define the PPO model
model = PPO(
    "MultiInputPolicy",  # Policy for dictionary observations
    env,
    learning_rate=3e-4,
    n_steps=2048,        # Number of steps to run for each environment per update
    batch_size=64,
    n_epochs=10,         # Number of epoch when optimizing the surrogate loss
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    clip_range=0.2,      # Clipping parameter for PPO
    ent_coef=0.01,       # Entropy coefficient for exploration
    tensorboard_log="./hospital_assistant_ppo_tensorboard/",  # Log directory for TensorBoard
    verbose=1,
)

# Train the model
print("Training the PPO model...")
model.learn(total_timesteps=300000)
print("PPO training complete!")

# Save the model
model_path = "hospital_assistant_ppo_model"
model.save(model_path)
print(f"Model saved as '{model_path}'")

# Close the environment
env.close()