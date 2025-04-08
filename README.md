# 🏥 Hospital Assistant AI

## Project Overview
An intelligent healthcare assistant robot powered by reinforcement learning that navigates hospital environments, prioritizes patient care, responds to emergencies, and manages medical resources efficiently.

![Hospital Assistant Demo](https://github.com/JoakBouy/summative-final/raw/main/assets/demo.gif)

## 🚀 Features
- **Intelligent Navigation**: Optimized pathfinding through complex hospital layouts
- **Dynamic Priority Management**: Automatic triaging of patients based on urgency
- **Resource Management**: Smart medicine collection and administration
- **Multi-objective Optimization**: Balances competing healthcare priorities
- **Comparative AI Research**: Analysis of DQN vs PPO algorithms for healthcare applications

## 🔍 Environment
The simulation takes place in an 8×8 grid hospital environment with:
- Regular patients requiring care
- Emergency cases needing immediate attention
- Medicine cabinets for resource collection
- Obstacles restricting movement
- Time constraints simulating real-world pressure

| Cell Type | Description |
|-----------|-------------|
| Empty | Navigable space |
| Patient | Regular care required |
| Emergency | High-priority case |
| Medicine Cabinet | Resource collection point |
| Obstacle | Non-navigable area |
| Visited | Already attended location |

## 🧠 AI Implementation
This project implements and compares two state-of-the-art reinforcement learning approaches:

### Deep Q-Network (DQN)
- Multi-input network architecture for processing complex state representations
- Experience replay buffer (100,000 transitions) for efficient learning
- Target network updates every 100 steps for stable training
- Annealed exploration strategy (1.0 → 0.05)

### Proximal Policy Optimization (PPO)
- Actor-critic architecture with shared feature extraction
- Generalized Advantage Estimation (GAE) with λ=0.95
- Clipped surrogate objective function (clip_range=0.2)
- Multiple optimization epochs (n=10) per batch

## 📊 Performance Comparison

| Metric | DQN | PPO |
|--------|-----|-----|
| Peak Reward | ~48 (early) | ~140 (sustained) |
| Training Stability | High volatility | Consistent improvement |
| Steps to Convergence | 60,000 (unstable) | 400,000 (stable) |
| Emergency Handling | Adequate | Superior |
| Generalization | Limited | Robust |

![Performance Comparison](https://github.com/JoakBouy/summative-final/raw/main/assets/performance_comparison.png)

## 🛠️ Installation & Usage

```bash
# Clone repository
git clone https://github.com/JoakBouy/summative-final.git
cd summative-final

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training Models
```bash
# Train PPO model
python train_ppo.py

# Train DQN model
python train_dqn.py
```

### Running Visualization
```bash
# Visualize trained agent performance
python visualize.py --model ppo  # or --model dqn
```

## 📝 Key Findings

- **Algorithm Suitability**: PPO demonstrated superior performance for complex healthcare scenarios with multiple competing objectives
- **Training Stability**: PPO showed consistent improvement without dramatic fluctuations thanks to its clipped surrogate objective
- **Long-term Planning**: PPO proved more effective at optimizing action sequences toward high-value goals requiring multiple preparatory steps
- **Credit Assignment**: DQN struggled with temporal credit assignment for delayed rewards requiring sequences of unrewarded actions

## 🔮 Future Work

1. **Hierarchical Reinforcement Learning**: Better management of multi-step tasks
2. **Prioritized Experience Replay**: Address sparse reward challenges for value-based methods
3. **Curriculum Learning**: Gradually increase environmental complexity for more robust agents
4. **Multi-agent Coordination**: Extend to team-based scenarios reflecting real-world hospital staffing
5. **Recurrent Network Integration**: Better adaptation to changing priorities and temporal patterns

## 📚 Resources

- [Project Video Demo](https://drive.google.com/file/d/1H9cMMtbXJP61-8g0-GSKzUmmefEj0G9-/view?usp=sharing)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 👤 Author
Joak Buoy Gai - African Leadership University

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.