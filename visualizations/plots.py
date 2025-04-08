import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from hospital_assistant_env import HospitalAssistantEnv
import pandas as pd
import os
import glob
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Function to load tensorboard logs for plotting
def get_tensorboard_data(log_dir, scalar_name="rollout/ep_rew_mean"):
    """Extract data from tensorboard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get data for specific scalar
    if scalar_name in event_acc.scalars.Keys():
        data = event_acc.Scalars(scalar_name)
        steps = [x.step for x in data]
        values = [x.value for x in data]
        return steps, values
    else:
        print(f"Scalar {scalar_name} not found in logs. Available scalars: {event_acc.scalars.Keys()}")
        return [], []

# Create environments for evaluation
def make_env():
    return Monitor(HospitalAssistantEnv(grid_size=8, max_steps=100))

# Load the trained models
try:
    dqn_model = DQN.load("hospital_assistant_dqn_model")
    print("DQN model loaded successfully")
except Exception as e:
    print(f"Error loading DQN model: {e}")
    dqn_model = None

try:
    ppo_model = PPO.load("hospital_assistant_ppo_model")
    print("PPO model loaded successfully")
except Exception as e:
    print(f"Error loading PPO model: {e}")
    ppo_model = None

# Plot training curves if available
plt.figure(figsize=(15, 6))

# DQN Training Curve
dqn_log_dir = "./hospital_assistant_dqn_tensorboard"
if os.path.exists(dqn_log_dir):
    # Find most recent log directory
    log_dirs = glob.glob(f"{dqn_log_dir}/DQN_*")
    if log_dirs:
        latest_log = max(log_dirs, key=os.path.getmtime)
        steps, rewards = get_tensorboard_data(latest_log, "rollout/ep_rew_mean")
        if steps:
            plt.subplot(1, 2, 1)
            plt.plot(steps, rewards)
            plt.title("DQN Training Rewards")
            plt.xlabel("Training Steps")
            plt.ylabel("Mean Episode Reward")
            plt.grid(True)
        
        # Get training loss if available
        loss_steps, losses = get_tensorboard_data(latest_log, "train/loss")
        if loss_steps:
            plt.subplot(1, 2, 2)
            plt.plot(loss_steps, losses)
            plt.title("DQN Training Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Loss")
            plt.grid(True)
    else:
        print("No DQN training logs found")
else:
    print(f"DQN log directory {dqn_log_dir} not found")

plt.tight_layout()
plt.savefig("dqn_training_curves.png")
plt.close()

# PPO Training Curve
plt.figure(figsize=(15, 6))
ppo_log_dir = "./hospital_assistant_ppo_tensorboard"
if os.path.exists(ppo_log_dir):
    # Find most recent log directory
    log_dirs = glob.glob(f"{ppo_log_dir}/PPO_*")
    if log_dirs:
        latest_log = max(log_dirs, key=os.path.getmtime)
        steps, rewards = get_tensorboard_data(latest_log, "rollout/ep_rew_mean")
        if steps:
            plt.subplot(1, 3, 1)
            plt.plot(steps, rewards)
            plt.title("PPO Training Rewards")
            plt.xlabel("Training Steps")
            plt.ylabel("Mean Episode Reward")
            plt.grid(True)
        
        # Get policy loss
        policy_steps, policy_losses = get_tensorboard_data(latest_log, "train/policy_loss")
        if policy_steps:
            plt.subplot(1, 3, 2)
            plt.plot(policy_steps, policy_losses)
            plt.title("PPO Policy Loss")
            plt.xlabel("Training Steps")
            plt.ylabel("Policy Loss")
            plt.grid(True)
        
        # Get entropy
        entropy_steps, entropy = get_tensorboard_data(latest_log, "train/entropy")
        if entropy_steps:
            plt.subplot(1, 3, 3)
            plt.plot(entropy_steps, entropy)
            plt.title("PPO Policy Entropy")
            plt.xlabel("Training Steps")
            plt.ylabel("Entropy")
            plt.grid(True)
    else:
        print("No PPO training logs found")
else:
    print(f"PPO log directory {ppo_log_dir} not found")

plt.tight_layout()
plt.savefig("ppo_training_curves.png")
plt.close()

# Run comparative evaluation of models
num_episodes = 20
models = {"DQN": dqn_model, "PPO": ppo_model}
model_results = {}

for model_name, model in models.items():
    if model is None:
        print(f"Skipping evaluation for {model_name} (model not loaded)")
        continue
        
    print(f"\nEvaluating {model_name} model over {num_episodes} episodes...")
    
    # Prepare metrics storage
    episode_rewards = []
    episode_steps = []
    success_count = 0
    visited_patients_pct = []
    visited_emergencies_pct = []
    
    # Run evaluation episodes
    for ep in range(num_episodes):
        env_eval = make_env()
        observation_eval = env_eval.reset()[0]
        total_reward_eval = 0
        steps_eval = 0
        done_eval = False
        truncated_eval = False
        
        while not (done_eval or truncated_eval):
            action_eval, _ = model.predict(observation_eval, deterministic=True)
            observation_eval, reward_eval, done_eval, truncated_eval, info_eval = env_eval.step(action_eval)
            total_reward_eval += reward_eval
            steps_eval += 1
        
        # Record metrics
        episode_rewards.append(total_reward_eval)
        episode_steps.append(steps_eval)
        
        # Check if all tasks were completed (success)
        if done_eval and not truncated_eval:
            success_count += 1
        
        # Calculate percentage of patients and emergencies visited
        visited_patients_pct.append(100 * info_eval['visited_patients'] / max(1, len(env_eval.patient_positions)))
        visited_emergencies_pct.append(100 * info_eval['visited_emergencies'] / max(1, len(env_eval.emergency_positions)))
        
        env_eval.close()
    
    # Compile results
    model_results[model_name] = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_steps": np.mean(episode_steps),
        "std_steps": np.std(episode_steps),
        "success_rate": 100 * success_count / num_episodes,
        "mean_patients_visited": np.mean(visited_patients_pct),
        "mean_emergencies_visited": np.mean(visited_emergencies_pct),
        "rewards": episode_rewards,
        "steps": episode_steps
    }
    
    print(f"{model_name} Results:")
    print(f"  Mean reward: {model_results[model_name]['mean_reward']:.2f} ± {model_results[model_name]['std_reward']:.2f}")
    print(f"  Mean steps: {model_results[model_name]['mean_steps']:.2f} ± {model_results[model_name]['std_steps']:.2f}")
    print(f"  Success rate: {model_results[model_name]['success_rate']:.1f}%")
    print(f"  Patients visited: {model_results[model_name]['mean_patients_visited']:.1f}%")
    print(f"  Emergencies handled: {model_results[model_name]['mean_emergencies_visited']:.1f}%")

# Compare models with bar charts
if len(model_results) > 0:
    plt.figure(figsize=(16, 12))
    
    # Define plotting positions and bar width
    bar_width = 0.35
    index = np.arange(len(model_results))
    model_names = list(model_results.keys())
    
    # Plot mean rewards
    plt.subplot(2, 2, 1)
    reward_means = [model_results[model]['mean_reward'] for model in model_names]
    reward_stds = [model_results[model]['std_reward'] for model in model_names]
    plt.bar(index, reward_means, bar_width, yerr=reward_stds, capsize=7)
    plt.xlabel('Model')
    plt.ylabel('Mean Reward')
    plt.title('Reward Comparison')
    plt.xticks(index, model_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot mean episode lengths
    plt.subplot(2, 2, 2)
    steps_means = [model_results[model]['mean_steps'] for model in model_names]
    steps_stds = [model_results[model]['std_steps'] for model in model_names]
    plt.bar(index, steps_means, bar_width, yerr=steps_stds, capsize=7)
    plt.xlabel('Model')
    plt.ylabel('Mean Episode Length')
    plt.title('Episode Length Comparison')
    plt.xticks(index, model_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot success rate
    plt.subplot(2, 2, 3)
    success_rates = [model_results[model]['success_rate'] for model in model_names]
    plt.bar(index, success_rates, bar_width)
    plt.xlabel('Model')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate Comparison')
    plt.xticks(index, model_names)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot patients and emergencies handled
    plt.subplot(2, 2, 4)
    x = np.arange(len(model_names))
    width = 0.3
    
    patients = [model_results[model]['mean_patients_visited'] for model in model_names]
    emergencies = [model_results[model]['mean_emergencies_visited'] for model in model_names]
    
    plt.bar(x - width/2, patients, width, label='Patients')
    plt.bar(x + width/2, emergencies, width, label='Emergencies')
    
    plt.xlabel('Model')
    plt.ylabel('Tasks Completed (%)')
    plt.title('Task Completion Rate')
    plt.xticks(x, model_names)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
    
    # Plot episode rewards over time
    plt.figure(figsize=(12, 5))
    
    for model_name in model_names:
        plt.plot(range(1, len(model_results[model_name]["rewards"])+1), 
                model_results[model_name]["rewards"], 
                marker='o', label=model_name)
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards per Model')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("episode_rewards_comparison.png")
    plt.show()

# Save results to file for future reference
with open("model_evaluation_results.json", "w") as f:
    # Convert numpy values to native Python types for JSON serialization
    serializable_results = {}
    for model_name, results in model_results.items():
        serializable_results[model_name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else 
               float(v) if isinstance(v, np.floating) else v
            for k, v in results.items()
        }
    json.dump(serializable_results, f, indent=4)

print("\nPlots and results saved to disk.")