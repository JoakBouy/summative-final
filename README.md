# Hospital Assistant AI 🏥+🤖

## Mission Statement
**"To revolutionize healthcare delivery through artificial intelligence, creating intelligent systems that enhance patient care, optimize medical workflows, and support healthcare professionals in delivering timely, effective treatment."**

## Overview
This project implements a Reinforcement Learning (RL) agent that navigates a hospital environment to assist patients, handle emergencies, and manage medicine distribution. The system demonstrates how AI can optimize healthcare logistics and decision-making in time-sensitive scenarios.

## Key Features
- **Smart Pathfinding**: AI learns optimal routes through complex hospital layouts
- **Priority Management**: Automatically triages patients based on urgency
- **Inventory Control**: Tracks and manages medicine supplies
- **Adaptive Learning**: Improves performance through continuous training
- **Visual Analytics**: Provides intuitive visualization of AI decision-making

## Technical Components
| Component | Description | Technology |
|-----------|-------------|------------|
| **Environment** | Custom hospital simulation with patients, emergencies, and medicine | Gymnasium, Pygame |
| **AI Models** | Dual RL approaches for comparison | PPO, DQN (Stable Baselines3) |
| **Training** | Optimized learning pipelines | TensorBoard, Vectorized Environments |
| **Evaluation** | Comprehensive performance metrics | Matplotlib, GIF generation |

## Installation
```bash
# Clone repository
git clone https://github.com/JoakBouy/summative-final.git

# Create virtual environment
python -m venv healthcare-ai
source healthcare-ai/bin/activate  # Linux/Mac
.\healthcare-ai\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train PPO model
python ppo_training.py

# Train DQN model
python dqn_training.py

# Generate visualization
python play.py

```

