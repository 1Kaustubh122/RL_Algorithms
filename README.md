# rlx-core

**Modular, reproducible, research-grade Reinforcement Learning repository built for robotics-centric deployment.**  
Covers tabular, deep, model-based, meta, and multi-agent RL across 13 structured phases with clean implementations and minimal dependencies.

---

## 🧠 Overview

`rlx-core` is a curriculum-aligned RL repository designed for robotics applications, focusing on low-level reproducibility and high-level extensibility.  
It builds foundational algorithms on grid-based environments, progressing to full-stack deep RL and world model agents.

---

## 🗂️ Structure

<pre>
rlx-core/
├── phase_01_tabular/             # Bandits, MDP, MC, TD, n-step, Planning
├── phase_02_function_approx/     # Tile/Coarse Coding, Neural Approx
├── phase_03_dqn/                 # DQN, DDQN, Dueling, Rainbow, PER
├── phase_04_pg/                  # REINFORCE, A2C, PPO, TRPO
├── phase_05_actor_critic/        # DDPG, TD3, SAC
├── phase_06_model_based/         # Dyna-Q, MBPO, PETS, PlaNet
├── phase_07_dreamer/             # DreamerV1, V2, V3
├── phase_08_world_models/        # MuZero, SimPLe
├── phase_09_meta_rl/             # MAML, RL²
├── phase_10_hierarchical_rl/     # Options, Feudal Networks
├── phase_11_offline_rl/          # BC, DAgger, CQL, BRAC
├── phase_12_multi_agent_rl/      # I-DQN, QMIX, MADDPG
├── phase_13_exploration/         # ICM, RND, NGU, Go-Explore
├── docs/                         # Papers, Diagrams
</pre>

---

## 🔍 Key Features

- **GridWorld base environments** for all tabular and early-phase implementations  
- **DM Control + OpenAI Gym support** in deep RL phases  
- **Minimal dependencies** for maximal portability and hardware control  
- **No frameworks like RLlib or Stable-Baselines** to preserve full control and transparency  
- **Strict separation by algorithmic paradigm and training methodology**  
- **Prepared for robotics applications**: real-time loop awareness, model-based planning, meta-RL generalization, sim-to-real portability  

---

## 📚 Docs

- `docs/papers/`: Canonical papers for each algorithm (DQN, PPO, DreamerV3, MuZero, etc.)  
- `docs/diagrams/`: Architecture visualizations and execution flow  

---

## 🛠️ Dependencies

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- NumPy, Matplotlib  
- For model-based/Dreamer phases: `torch.distributions`, `opencv-python`, `imageio`  
- For visualizations: `seaborn`, `tensorboard`  

---

## 🚧 Notes

- Inverse RL (AIRL, GAIL) is **not included** in this version. May be added in a separate repository later  
- Focus is on **RL for control**, not academic benchmarking  
- All code is **from scratch**, no third-party abstractions  
- Stable commit points will be **tagged per completed phase**  

---

## 📜 License

MIT License. Free to use, modify, and distribute with attribution.
