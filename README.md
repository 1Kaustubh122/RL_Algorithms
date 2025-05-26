```markdown
# rlx-core

**Modular, reproducible, research-grade Reinforcement Learning repository built for robotics-centric deployment.**  
Covers tabular, deep, model-based, meta, and multi-agent RL across 13 structured phases with clean implementations and minimal dependencies.

---

## 🧠 Overview

`rlx-core` is a curriculum-aligned RL repository designed for robotics applications, focusing on low-level reproducibility and high-level extensibility.  
It builds foundational algorithms on grid-based environments, progressing to full-stack deep RL and world model agents.

---

## 🗂️ Structure

```

rlx-core/
├── phase\_01\_tabular/             # Bandits, TD, MC, SARSA, Q-Learning, Planning
├── phase\_02\_function\_approx/     # Tile/Coarse Coding, Neural Approx
├── phase\_03\_dqn/                 # DQN, DDQN, Dueling, Rainbow, PER
├── phase\_04\_pg/                  # REINFORCE, A2C, PPO, TRPO
├── phase\_05\_actor\_critic/        # DDPG, TD3, SAC
├── phase\_06\_model\_based/         # Dyna-Q, MBPO, PETS, PlaNet
├── phase\_07\_dreamer/             # DreamerV1, V2, V3
├── phase\_08\_world\_models/        # MuZero, SimPLe
├── phase\_09\_meta\_rl/             # MAML, RL²
├── phase\_10\_hierarchical\_rl/     # Options, Feudal Networks
├── phase\_11\_offline\_rl/          # BC, DAgger, CQL, BRAC
├── phase\_12\_multi\_agent\_rl/      # I-DQN, QMIX, MADDPG
├── phase\_13\_exploration/         # ICM, RND, NGU, Go-Explore
├── docs/                         # Papers, Diagrams

```

---

## 🔍 Key Features

- **GridWorld base environments** for all tabular and early-phase implementations.
- **DM Control + OpenAI Gym support** in deep RL phases.
- **Minimal dependencies** for maximal portability and hardware control.
- **No frameworks like RLlib or Stable-Baselines** to preserve full control and transparency.
- **Strict separation by algorithmic paradigm and training methodology.**
- **Prepared for robotics applications**: Real-time loop awareness, model-based planning, meta-RL generalization, and sim-to-real portability.

---

## 📚 Docs

- `docs/papers/`: Canonical papers for each algorithm (DQN, PPO, DreamerV3, MuZero, etc.)
- `docs/diagrams/`: Architecture visualizations and execution flow

---

## 🚧 Notes

- Inverse RL (AIRL, GAIL) is not included in this version. It may be split into a separate repository later.
- Focus is on **RL for control**, not pure academic benchmarking. Robotics deployment is the goal.
- All code is being written from scratch, module-by-module. No third-party abstractions.
- Stable commit points will be tagged per completed phase.

---

## 📜 License

MIT License. Free to use, modify, distribute with attribution.

```
