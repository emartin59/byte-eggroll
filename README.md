# Byte-Eggroll: Cultural Transmission on TPUs

A massively parallel ALife simulation using JAX and EGGROLL (Low-Rank Evolution Strategies) to study the emergence of cultural transmission in multi-agent systems.

## 🚀 Overview
Byte-Eggroll simulates "species" of agents controlled by GRU (Gated Recurrent Unit) neural networks. These agents inhabit a 2D grid where they must:
- **Forage:** Locate and consume food to maintain energy.
- **Communicate:** Emit and receive audio tokens.
- **Transmit:** Write persistent marks to the grid that future generations can read.

The core goal is to observe **Proto-Cultural Transmission**: a scenario where information left by earlier agents (writing) significantly alters the survival and behavior of later agents.

## 🛠 Architecture
- **Engine:** JAX-native world engine using `jax.lax.scan` for O(1) TPU step-time.
- **Optimization:** EGGROLL ES. This avoids backpropagation and allows for non-differentiable world logic.
- **Parallelism:** `jax.pmap` distributes the population across 8 TPU cores (v3-8 or v5e-8).
- **Policy:** A Gated Recurrent Unit (GRU) with vision, audio, and proprioceptive inputs.

## 📊 Inputs & Outputs
- **Vision:** 15x15 local grid view (8 channels: food, agents, species, writing, etc.).
- **Audio:** 3-tick buffer of 27-dimensional communication tokens.
- **Proprioception:** Energy level and species ID.
- **Actions:** Move (5), Speak (27), Write (27), and Inventory/Misc (4).

## 🏃 Getting Started
1. Open a Kaggle Notebook or Google Cloud TPU VM.
2. Select **TPU v3-8** or **v5e-8** as the accelerator.
3. Run the script. The simulation will JIT-compile (taking ~1-2 minutes) and then begin printing generation logs.
