# %% [markdown]
# # EGGROLL-Accelerated Byte-Multi-Agent: Cultural Transmission Experiment
#
# **What this does:**
# Runs a tabula-rasa multi-agent ALife simulation on Kaggle's TPU v5e-8.
# Agents (small GRU networks) live on a 2D byte grid, eat food, speak to each other,
# and write persistent marks. We use EGGROLL-style low-rank ES perturbations
# to optimize all species simultaneously — much faster than naïve ES.
#
# **The scientific bet:**
# Can writing accumulate meaningfully across generations?
# We measure whether agents born later behave differently because of marks
# left by earlier agents — proto-cultural transmission.
#
# **TPU Setup:** Select TPU v5e-8 accelerator in Kaggle Settings before running.

# %% [markdown]
# ## Cell 1: Install dependencies

# %%
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html")
install("flax")
install("optax")

# %% [markdown]
# ## Cell 2: Verify TPU

# %%
import jax
import jax.numpy as jnp

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Device count:", jax.device_count())
assert jax.device_count() >= 4, "Expected at least 4 TPU cores. Check accelerator setting."

# %% [markdown]
# ## Cell 3: Configuration
# All hyperparameters in one place. Edit these to tune the experiment.

# %%
from dataclasses import dataclass

@dataclass
class Config:
    # World
    GRID_W: int = 64
    GRID_H: int = 64
    N_AGENTS: int = 8          # agents per environment
    N_SPECIES: int = 4         # separate policy networks
    EPISODE_TICKS: int = 400

    # Agent architecture
    VISION_R: int = 7          # vision crop radius → (2R+1)^2 = 225 input cells
    HIDDEN_DIM: int = 256      # GRU hidden state  (was 64 in original, scaled up)
    AUDIO_TOKS: int = 27       # speech token vocab (a-z + silence)
    AUDIO_BUF: int = 3         # ticks of audio history
    AUDIO_RANGE: int = 10      # Chebyshev range for hearing

    # Actions
    N_MOVE: int = 5            # stay + 4 directions
    N_SPEAK: int = 27          # speech tokens
    N_WRITE: int = 27          # write tokens (same vocab)
    N_INV: int = 4             # inventory slots

    # Economy
    INIT_ENERGY: float = 100.0
    METABOLIC_COST: float = 0.5
    MOVE_COST: float = 1.0
    SPEAK_COST: float = 0.3
    WRITE_COST: float = 0.5
    FOOD_ENERGY: float = 30.0
    FOOD_DENSITY: float = 0.08  # fraction of grid cells start with food
    SEED_REGROW_PROB: float = 0.02

    # EGGROLL ES
    POP_SIZE: int = 512         # perturbation population per update
    N_ENVS_PER_MEMBER: int = 2  # rollouts per population member
    NOISE_STD: float = 0.05
    EGGROLL_RANK: int = 1       # r=1 is the EGGROLL sweet spot from the paper
    LR: float = 0.01
    LR_DECAY: float = 0.999
    WEIGHT_DECAY: float = 1e-4

    # Training
    N_GENERATIONS: int = 2000
    CHECKPOINT_EVERY: int = 50
    PRINT_EVERY: int = 25

    # Fitness weights
    W_SURVIVAL: float = 1.0
    W_ENERGY: float = 0.5
    W_FINAL_ENERGY: float = 0.1

    # Cultural transmission diagnostics
    CULTURAL_SAMPLE_EVERY: int = 100  # generations between cultural analysis
    WRITING_INFLUENCE_WINDOW: int = 10 # generations to measure writing→behavior

CFG = Config()

# %% [markdown]
# ## Cell 4: World & Grid Utilities

# %%
import numpy as np
from functools import partial

# Byte channel indices in grid tensor
# Grid shape: [H, W, N_CHANNELS]
CH_FOOD    = 0
CH_SEED    = 1
CH_ROCK    = 2
CH_WATER   = 3
CH_WRITE   = 4   # persistent byte (0 = empty, 1-26 = letter a-z)
CH_TOOL    = 5   # tool byte
CH_AGENT   = 6   # agent id +1 (0 = empty)
CH_ENERGY  = 7   # agent energy (scaled 0-255)
N_CHANNELS = 8

def init_world(key: jax.Array) -> jax.Array:
    """Initialize a fresh grid. Returns float32 [H, W, N_CHANNELS]."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    H, W = CFG.GRID_H, CFG.GRID_W
    grid = jnp.zeros((H, W, N_CHANNELS), dtype=jnp.float32)

    # Scatter food
    food_mask = jax.random.bernoulli(k1, CFG.FOOD_DENSITY, (H, W))
    grid = grid.at[:, :, CH_FOOD].set(food_mask.astype(jnp.float32))

    # Scatter seeds (half as common as food)
    seed_mask = jax.random.bernoulli(k2, CFG.FOOD_DENSITY / 2, (H, W))
    grid = grid.at[:, :, CH_SEED].set(seed_mask.astype(jnp.float32))

    # Scatter rocks
    rock_mask = jax.random.bernoulli(k3, 0.05, (H, W))
    grid = grid.at[:, :, CH_ROCK].set(rock_mask.astype(jnp.float32))

    # Scatter water
    water_mask = jax.random.bernoulli(k4, 0.04, (H, W))
    grid = grid.at[:, :, CH_WATER].set(water_mask.astype(jnp.float32))

    return grid


def place_agents(key: jax.Array, grid: jax.Array) -> tuple:
    """Place N_AGENTS on empty cells. Returns (grid, positions [N,2], energies [N])."""
    H, W = CFG.GRID_H, CFG.GRID_W
    # Sample random positions
    keys = jax.random.split(key, CFG.N_AGENTS)
    rows = jax.random.randint(keys[0], (CFG.N_AGENTS,), 1, H - 1)
    cols = jax.random.randint(keys[1], (CFG.N_AGENTS,), 1, W - 1)
    pos = jnp.stack([rows, cols], axis=-1)  # [N, 2]
    energies = jnp.full((CFG.N_AGENTS,), CFG.INIT_ENERGY)

    # Write agent IDs into grid
    def place_one(carry, x):
        g = carry
        i, r, c = x
        g = g.at[r, c, CH_AGENT].set(float(i + 1))
        g = g.at[r, c, CH_ENERGY].set(CFG.INIT_ENERGY / 255.0 * 255.0)
        return g, None

    grid, _ = jax.lax.scan(
        place_one, grid,
        (jnp.arange(CFG.N_AGENTS), rows, cols)
    )
    return grid, pos, energies


def get_vision_crop(grid: jax.Array, row: int, col: int) -> jax.Array:
    """Extract (2R+1)^2 x N_CHANNELS vision crop with torus wrapping."""
    R = CFG.VISION_R
    H, W = CFG.GRID_H, CFG.GRID_W
    rows = (jnp.arange(-R, R + 1) + row) % H
    cols = (jnp.arange(-R, R + 1) + col) % W
    # grid[rows][:, cols] — use advanced indexing
    crop = grid[rows[:, None], cols[None, :], :]  # [2R+1, 2R+1, C]
    return crop.reshape(-1)  # flatten → (2R+1)^2 * C

# %% [markdown]
# ## Cell 5: Agent Policy Network (GRU)
# Each species has its own copy of these weights.
# Input: vision_flat + audio + proprioception
# Output: move logits, speak logits, write logits, inv_select logits

# %%
import flax.linen as nn
from typing import Tuple

VISION_DIM = (2 * CFG.VISION_R + 1) ** 2 * N_CHANNELS  # 225 * 8 = 1800
AUDIO_DIM  = CFG.AUDIO_BUF * CFG.AUDIO_TOKS             # 3 * 27 = 81
PROPR_DIM  = 6   # energy, inv_contents (4 slots), species_id

INPUT_DIM  = VISION_DIM + AUDIO_DIM + PROPR_DIM


class AgentPolicy(nn.Module):
    hidden_dim: int = CFG.HIDDEN_DIM

    @nn.compact
    def __call__(self, obs: jax.Array, hidden: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        obs:    [INPUT_DIM]
        hidden: [hidden_dim]
        returns: (action_logits [N_MOVE + N_SPEAK + N_WRITE + N_INV], new_hidden [hidden_dim])
        """
        # Embed observation
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.tanh(x)

        # GRU step
        # Manual GRU for clarity (Flax GRUCell works fine too)
        gates = nn.Dense(2 * self.hidden_dim)(jnp.concatenate([x, hidden]))
        r, z = jnp.split(nn.sigmoid(gates), 2, axis=-1)
        h_candidate = nn.Dense(self.hidden_dim)(jnp.concatenate([x, r * hidden]))
        h_candidate = jnp.tanh(h_candidate)
        new_hidden = (1 - z) * hidden + z * h_candidate

        # Action heads
        move_logits  = nn.Dense(CFG.N_MOVE)(new_hidden)
        speak_logits = nn.Dense(CFG.N_SPEAK)(new_hidden)
        write_logits = nn.Dense(CFG.N_WRITE)(new_hidden)
        inv_logits   = nn.Dense(CFG.N_INV)(new_hidden)

        all_logits = jnp.concatenate([move_logits, speak_logits, write_logits, inv_logits])
        return all_logits, new_hidden


def init_policy_params(key: jax.Array) -> dict:
    """Initialize one species' policy parameters."""
    model = AgentPolicy()
    dummy_obs    = jnp.zeros(INPUT_DIM)
    dummy_hidden = jnp.zeros(CFG.HIDDEN_DIM)
    params = model.init(key, dummy_obs, dummy_hidden)
    return params


def policy_forward(params: dict, obs: jax.Array, hidden: jax.Array) -> Tuple[jax.Array, jax.Array]:
    model = AgentPolicy()
    return model.apply(params, obs, hidden)

# %% [markdown]
# ## Cell 6: EGGROLL Low-Rank Perturbation
#
# Core math from the paper:
#   E_i = (1/sqrt(r)) * A_i @ B_i^T
#   perturbed_params = base_params + sigma * E_i
#
# We work with flattened parameter vectors for simplicity, then reshape back.

# %%
def flatten_params(params: dict) -> jax.Array:
    """Flatten nested param dict to 1D vector."""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([x.ravel() for x in leaves])


def unflatten_params(flat: jax.Array, template: dict) -> dict:
    """Restore flat vector back to param tree matching template structure."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    shapes = [x.shape for x in leaves]
    sizes  = [x.size  for x in leaves]
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    chunks = jnp.split(flat, splits)
    restored_leaves = [c.reshape(s) for c, s in zip(chunks, shapes)]
    return treedef.unflatten(restored_leaves)


def eggroll_perturb(key: jax.Array, base_flat: jax.Array) -> tuple:
    """
    Generate one EGGROLL rank-r perturbation.
    Returns (perturbed_flat, (A, B)) where E = (1/sqrt(r)) * outer(A, B)
    approximated as additive noise to the full flat vector.

    For rank-1: we generate A [d] and B [1], so E = A * b_scalar,
    which is just a random direction vector scaled by a scalar.
    This is the rank-1 EGGROLL case: highly efficient.
    """
    d = base_flat.shape[0]
    r = CFG.EGGROLL_RANK

    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (d, r))           # [d, r]
    B = jax.random.normal(k2, (r,))             # [r]  (one B vector, rank-1 collapse)

    # E = (1/sqrt(r)) * A @ B  →  shape [d]
    E = (1.0 / jnp.sqrt(float(r))) * (A @ B)

    perturbed = base_flat + CFG.NOISE_STD * E
    return perturbed, (A, B)


def eggroll_antithetic_pair(key: jax.Array, base_flat: jax.Array) -> tuple:
    """
    Returns (pos_flat, neg_flat, A, B) for antithetic sampling.
    Using antithetic pairs halves variance and is standard in ES.
    """
    d = base_flat.shape[0]
    r = CFG.EGGROLL_RANK
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (d, r))
    B = jax.random.normal(k2, (r,))
    E = (1.0 / jnp.sqrt(float(r))) * (A @ B)
    pos_flat = base_flat + CFG.NOISE_STD * E
    neg_flat = base_flat - CFG.NOISE_STD * E
    return pos_flat, neg_flat, A, B


def eggroll_update(base_flat: jax.Array,
                   As: jax.Array,
                   Bs: jax.Array,
                   shaped_fitnesses: jax.Array,
                   lr: float) -> jax.Array:
    """
    Compute the EGGROLL parameter update:
      Δθ = (lr / N) * Σ_i  E_i * f_i
         = (lr / N) * Σ_i  [(1/sqrt(r)) * A_i @ B_i] * f_i
    
    As: [N, d, r]
    Bs: [N, r]
    shaped_fitnesses: [N]  (rank-normalized, antithetic-paired)
    """
    r = CFG.EGGROLL_RANK
    N = shaped_fitnesses.shape[0]

    # E_i = (1/sqrt(r)) * As[i] @ Bs[i]  →  [d]
    # Weighted sum: Σ_i f_i * E_i
    # = (1/sqrt(r)) * Σ_i f_i * As[i] @ Bs[i]
    # Efficient: (diag(f) @ As reshaped) then matmul with Bs

    # As: [N, d, r], Bs: [N, r], f: [N]
    # Σ_i f_i * (As[i] @ Bs[i]) = As.T(weighted) * Bs summed
    # = einsum('i,idr,ir->d', f, As, Bs) but let's do it step by step

    # Step 1: weighted Bs → [N, r]
    fBs = shaped_fitnesses[:, None] * Bs   # [N, r]

    # Step 2: Σ_i A_i[:,j] * (f_i * B_i[j]) for all j → [d]
    # = einsum('idr,ir->d', As, fBs)
    delta = jnp.einsum('idr,ir->d', As, fBs)   # [d]
    delta = delta / (jnp.sqrt(float(r)) * float(N))

    new_flat = base_flat + lr * delta

    # Weight decay
    new_flat = new_flat * (1.0 - CFG.WEIGHT_DECAY)

    return new_flat


def rank_shape_fitnesses(raw_fitnesses: jax.Array) -> jax.Array:
    """
    Rank-based fitness shaping: map to [-0.5, 0.5].
    Standard in ES to reduce sensitivity to outliers.
    raw_fitnesses: [N]
    """
    N = raw_fitnesses.shape[0]
    ranks = jnp.argsort(jnp.argsort(raw_fitnesses))  # double argsort = rank
    shaped = ranks.astype(jnp.float32) / float(N - 1) - 0.5
    return shaped

# %% [markdown]
# ## Cell 7: Environment Step
# One tick of the simulation. Pure JAX, JIT-compilable.

# %%
MOVE_DELTAS = jnp.array([
    [0, 0],   # stay
    [-1, 0],  # up
    [1, 0],   # down
    [0, -1],  # left
    [0, 1],   # right
], dtype=jnp.int32)


def agent_obs(grid: jax.Array, pos: jax.Array,
              audio_buf: jax.Array, energy: float,
              inv: jax.Array, species_id: int) -> jax.Array:
    """Assemble observation vector for one agent."""
    r, c = pos[0], pos[1]
    vision = get_vision_crop(grid, r, c)          # [VISION_DIM]
    audio  = audio_buf.ravel()                    # [AUDIO_DIM]
    propr  = jnp.array([
        energy / CFG.INIT_ENERGY,                 # normalized energy
        inv[0].astype(jnp.float32) / 255.0,
        inv[1].astype(jnp.float32) / 255.0,
        inv[2].astype(jnp.float32) / 255.0,
        inv[3].astype(jnp.float32) / 255.0,
        float(species_id) / float(CFG.N_SPECIES),
    ])
    return jnp.concatenate([vision, audio, propr])  # [INPUT_DIM]


def step_agent(params: dict,
               grid: jax.Array,
               pos: jax.Array,
               hidden: jax.Array,
               energy: float,
               inv: jax.Array,
               audio_buf: jax.Array,
               species_id: int,
               key: jax.Array) -> tuple:
    """
    Run one agent for one tick.
    Returns: (new_grid, new_pos, new_hidden, new_energy, new_inv,
              spoken_token, written_token, alive)
    """
    H, W = CFG.GRID_H, CFG.GRID_W
    alive = energy > 0.0

    obs = agent_obs(grid, pos, audio_buf, energy, inv, species_id)
    logits, new_hidden = policy_forward(params, obs, hidden)

    # Split logits into heads
    move_logits  = logits[:CFG.N_MOVE]
    speak_logits = logits[CFG.N_MOVE: CFG.N_MOVE + CFG.N_SPEAK]
    write_logits = logits[CFG.N_MOVE + CFG.N_SPEAK: CFG.N_MOVE + CFG.N_SPEAK + CFG.N_WRITE]
    inv_logits   = logits[CFG.N_MOVE + CFG.N_SPEAK + CFG.N_WRITE:]

    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Sample actions (greedy during eval, sampled during training)
    move_action  = jax.random.categorical(k1, move_logits)
    speak_token  = jax.random.categorical(k2, speak_logits)
    write_token  = jax.random.categorical(k3, write_logits)
    inv_slot     = jax.random.categorical(k4, inv_logits)

    # Movement
    delta = MOVE_DELTAS[move_action]
    new_r = (pos[0] + delta[0]) % H
    new_c = (pos[1] + delta[1]) % W
    new_pos = jnp.where(alive, jnp.array([new_r, new_c]), pos)

    # Energy costs
    move_cost  = jnp.where(move_action > 0, CFG.MOVE_COST, 0.0)
    speak_cost = jnp.where(speak_token > 0, CFG.SPEAK_COST, 0.0)
    write_cost = jnp.where(write_token > 0, CFG.WRITE_COST, 0.0)
    new_energy = energy - CFG.METABOLIC_COST - move_cost - speak_cost - write_cost

    # Eat food at new position
    food_here = grid[new_r, new_c, CH_FOOD]
    new_energy = new_energy + food_here * CFG.FOOD_ENERGY

    # Remove food, plant seed
    new_grid = grid.at[new_r, new_c, CH_FOOD].set(
        jnp.where(food_here > 0, 0.0, grid[new_r, new_c, CH_FOOD])
    )
    new_grid = new_grid.at[new_r, new_c, CH_SEED].set(
        jnp.where(food_here > 0, 1.0, new_grid[new_r, new_c, CH_SEED])
    )

    # Write to grid (only if agent chose to write, token > 0 = non-silence)
    write_val = jnp.where(write_token > 0, write_token.astype(jnp.float32), grid[new_r, new_c, CH_WRITE])
    new_grid = jnp.where(alive & (write_token > 0),
                         new_grid.at[new_r, new_c, CH_WRITE].set(write_val),
                         new_grid)

    # Update agent channel
    new_grid = new_grid.at[pos[0], pos[1], CH_AGENT].set(0.0)
    new_grid = jnp.where(alive,
                         new_grid.at[new_r, new_c, CH_AGENT].set(float(species_id + 1)),
                         new_grid)

    new_energy = jnp.maximum(new_energy, 0.0)
    new_alive  = new_energy > 0.0

    # If dead, return hidden as zeros (agent respawns next episode)
    new_hidden = jnp.where(new_alive, new_hidden, jnp.zeros_like(new_hidden))

    return (new_grid, new_pos, new_hidden, new_energy, inv,
            speak_token, write_token, new_alive)


def step_world(grid: jax.Array, key: jax.Array) -> jax.Array:
    """Grow seeds stochastically."""
    H, W = CFG.GRID_H, CFG.GRID_W
    grow_mask = jax.random.bernoulli(key, CFG.SEED_REGROW_PROB, (H, W))
    new_food = jnp.minimum(grid[:, :, CH_FOOD] + (grid[:, :, CH_SEED] * grow_mask), 1.0)
    grid = grid.at[:, :, CH_FOOD].set(new_food)
    return grid

# %% [markdown]
# ## Cell 8: Full Episode Rollout
# Run one complete episode for one set of species weights.
# Returns scalar fitness score.

# %%
def run_episode(all_species_params: list,
                species_assignments: jax.Array,
                key: jax.Array) -> tuple:
    """
    Run one full episode.
    all_species_params: list of N_SPECIES param dicts
    species_assignments: [N_AGENTS] int array, which species each agent is
    key: PRNG key

    Returns: (total_fitness, diagnostics_dict)
    """
    k_world, k_agents, k_run = jax.random.split(key, 3)

    # Initialize world
    grid = init_world(k_world)
    grid, positions, energies = place_agents(k_agents, grid)

    # Initialize hidden states and audio buffers
    hiddens    = jnp.zeros((CFG.N_AGENTS, CFG.HIDDEN_DIM))
    inventories = jnp.zeros((CFG.N_AGENTS, CFG.N_INV), dtype=jnp.uint8)
    audio_bufs  = jnp.zeros((CFG.N_AGENTS, CFG.AUDIO_BUF, CFG.AUDIO_TOKS))

    # Accumulators
    survival_ticks = jnp.zeros(CFG.N_AGENTS)
    energy_earned  = jnp.zeros(CFG.N_AGENTS)
    writing_counts = jnp.zeros(CFG.N_AGENTS)
    speech_counts  = jnp.zeros(CFG.N_AGENTS)

    keys_run = jax.random.split(k_run, CFG.EPISODE_TICKS)

    for tick in range(CFG.EPISODE_TICKS):
        tick_key = keys_run[tick]
        agent_keys = jax.random.split(tick_key, CFG.N_AGENTS + 1)
        world_key  = agent_keys[-1]

        spoken_tokens = jnp.zeros(CFG.N_AGENTS, dtype=jnp.int32)

        for agent_i in range(CFG.N_AGENTS):
            sp_id   = int(species_assignments[agent_i])
            params  = all_species_params[sp_id]
            alive   = energies[agent_i] > 0.0

            (grid, new_pos, new_hidden, new_energy, new_inv,
             speak_tok, write_tok, still_alive) = step_agent(
                params=params,
                grid=grid,
                pos=positions[agent_i],
                hidden=hiddens[agent_i],
                energy=energies[agent_i],
                inv=inventories[agent_i],
                audio_buf=audio_bufs[agent_i],
                species_id=sp_id,
                key=agent_keys[agent_i]
            )

            positions    = positions.at[agent_i].set(new_pos)
            hiddens      = hiddens.at[agent_i].set(new_hidden)
            energies     = energies.at[agent_i].set(new_energy)
            inventories  = inventories.at[agent_i].set(new_inv)
            spoken_tokens = spoken_tokens.at[agent_i].set(speak_tok)

            # Diagnostics
            survival_ticks = survival_ticks.at[agent_i].add(still_alive.astype(jnp.float32))
            energy_earned  = energy_earned.at[agent_i].add(jnp.maximum(new_energy - energies[agent_i], 0.0))
            writing_counts = writing_counts.at[agent_i].add((write_tok > 0).astype(jnp.float32))
            speech_counts  = speech_counts.at[agent_i].add((speak_tok > 0).astype(jnp.float32))

        # Build audio buffers: shift and insert new tokens
        new_audio = jnp.zeros((CFG.N_AGENTS, CFG.AUDIO_BUF, CFG.AUDIO_TOKS))
        for agent_i in range(CFG.N_AGENTS):
            pos_i = positions[agent_i]
            for agent_j in range(CFG.N_AGENTS):
                if agent_i == agent_j:
                    continue
                pos_j = positions[agent_j]
                dist  = jnp.maximum(
                    jnp.abs(pos_i[0] - pos_j[0]),
                    jnp.abs(pos_i[1] - pos_j[1])
                )
                in_range = dist <= CFG.AUDIO_RANGE
                tok_j    = spoken_tokens[agent_j]
                one_hot  = jax.nn.one_hot(tok_j, CFG.AUDIO_TOKS)
                new_audio = new_audio.at[agent_i, 0].add(
                    jnp.where(in_range, one_hot, jnp.zeros(CFG.AUDIO_TOKS))
                )

        # Shift audio buffer (newest at index 0)
        audio_bufs = jnp.concatenate([
            new_audio[:, :1, :],
            audio_bufs[:, :CFG.AUDIO_BUF - 1, :]
        ], axis=1)

        # World step (seed regrowth)
        grid = step_world(grid, world_key)

    # Fitness: survival + energy
    fitness = (
        CFG.W_SURVIVAL    * jnp.sum(survival_ticks) +
        CFG.W_ENERGY      * jnp.sum(energy_earned) +
        CFG.W_FINAL_ENERGY * jnp.sum(energies)
    )

    # Writing coverage: fraction of cells with marks
    writing_coverage = jnp.mean(grid[:, :, CH_WRITE] > 0)

    diagnostics = {
        "fitness":          float(fitness),
        "mean_survival":    float(jnp.mean(survival_ticks)),
        "mean_speech":      float(jnp.mean(speech_counts)),
        "speech_diversity": float(len(jnp.unique(spoken_tokens))),
        "mean_writing":     float(jnp.mean(writing_counts)),
        "writing_coverage": float(writing_coverage),
        "final_grid":       grid,  # for cultural analysis
    }

    return fitness, diagnostics

# %% [markdown]
# ## Cell 9: Training Loop
# The main ES loop with EGGROLL updates.

# %%
import time
import pickle
import os

def compute_fitness_for_member(species_flat_perturbed: list,
                                species_templates: list,
                                species_assignments: jax.Array,
                                key: jax.Array) -> float:
    """Reconstruct params from flat vectors and run episode."""
    params_list = [
        unflatten_params(flat, template)
        for flat, template in zip(species_flat_perturbed, species_templates)
    ]
    fitness, _ = run_episode(params_list, species_assignments, key)
    return fitness


def training_loop():
    print("="*60)
    print("EGGROLL Byte-Multi-Agent: Cultural Transmission Experiment")
    print("="*60)
    print(f"TPU cores: {jax.device_count()}")
    print(f"Grid: {CFG.GRID_W}x{CFG.GRID_H}, Agents: {CFG.N_AGENTS}, Species: {CFG.N_SPECIES}")
    print(f"Pop size: {CFG.POP_SIZE}, EGGROLL rank: {CFG.EGGROLL_RANK}")
    print(f"Policy hidden dim: {CFG.HIDDEN_DIM}, Params/species: ~{INPUT_DIM * CFG.HIDDEN_DIM * 4:,}")
    print()

    # ── Initialize keys and params ──────────────────────────────────────
    master_key = jax.random.PRNGKey(42)
    k_init, master_key = jax.random.split(master_key)
    init_keys = jax.random.split(k_init, CFG.N_SPECIES)

    species_params   = [init_policy_params(init_keys[s]) for s in range(CFG.N_SPECIES)]
    species_templates = species_params.copy()
    species_flat     = [flatten_params(p) for p in species_params]

    param_dims = [f.shape[0] for f in species_flat]
    print(f"Parameter dimensions per species: {param_dims}")
    print(f"Total parameters: {sum(param_dims):,}")
    print()

    lr = CFG.LR

    # Fixed species assignments: rotate through species
    species_assignments = jnp.array(
        [i % CFG.N_SPECIES for i in range(CFG.N_AGENTS)],
        dtype=jnp.int32
    )

    # ── Logging ──────────────────────────────────────────────────────────
    history = {
        "fitness":           [],
        "mean_survival":     [],
        "mean_speech":       [],
        "speech_diversity":  [],
        "mean_writing":      [],
        "writing_coverage":  [],
        "generation":        [],
        # Cultural transmission metrics
        "cultural_influence": [],  # writing→behavior correlation across generations
    }

    # Writing fingerprint from previous cultural window
    prev_writing_grid = None
    cultural_influence_scores = []

    # ── Main loop ────────────────────────────────────────────────────────
    for gen in range(CFG.N_GENERATIONS):
        t0 = time.time()

        k_gen, master_key = jax.random.split(master_key)
        pop_keys = jax.random.split(k_gen, CFG.POP_SIZE)

        # Collect fitnesses for all population members
        # For each member: generate antithetic pair for each species, run 2 envs
        all_fitnesses_pos = []
        all_fitnesses_neg = []
        all_As = [[] for _ in range(CFG.N_SPECIES)]
        all_Bs = [[] for _ in range(CFG.N_SPECIES)]

        half_pop = CFG.POP_SIZE // 2

        for member in range(half_pop):
            k_m = pop_keys[member]
            k_env1, k_env2, k_pert = jax.random.split(k_m, 3)

            # Generate antithetic perturbations for each species
            pos_flats = []
            neg_flats = []
            pert_keys = jax.random.split(k_pert, CFG.N_SPECIES)

            for s in range(CFG.N_SPECIES):
                pos_f, neg_f, A, B = eggroll_antithetic_pair(pert_keys[s], species_flat[s])
                pos_flats.append(pos_f)
                neg_flats.append(neg_f)
                all_As[s].append(A)
                all_Bs[s].append(B)

            # Evaluate pos perturbation
            f_pos = 0.0
            for env_idx in range(CFG.N_ENVS_PER_MEMBER):
                k_ev = jax.random.fold_in(k_env1, env_idx)
                f_pos += compute_fitness_for_member(
                    pos_flats, species_templates, species_assignments, k_ev
                )
            f_pos /= CFG.N_ENVS_PER_MEMBER

            # Evaluate neg perturbation
            f_neg = 0.0
            for env_idx in range(CFG.N_ENVS_PER_MEMBER):
                k_ev = jax.random.fold_in(k_env2, env_idx)
                f_neg += compute_fitness_for_member(
                    neg_flats, species_templates, species_assignments, k_ev
                )
            f_neg /= CFG.N_ENVS_PER_MEMBER

            all_fitnesses_pos.append(f_pos)
            all_fitnesses_neg.append(f_neg)

        # Combine antithetic fitnesses: shaped by relative rank
        # For antithetic pairs, fitness signal = f_pos - f_neg (sign only, from paper)
        all_fitnesses_pos = jnp.array(all_fitnesses_pos)
        all_fitnesses_neg = jnp.array(all_fitnesses_neg)

        # Rank-shape on combined pool
        combined = jnp.concatenate([all_fitnesses_pos, all_fitnesses_neg])
        combined_shaped = rank_shape_fitnesses(combined)
        shaped_pos = combined_shaped[:half_pop]
        shaped_neg = combined_shaped[half_pop:]

        # Antithetic signal: positive perturbation scored positively, negative negatively
        antithetic_signal = shaped_pos - shaped_neg   # [half_pop]

        # EGGROLL update for each species
        for s in range(CFG.N_SPECIES):
            As = jnp.stack(all_As[s])  # [half_pop, d, r]
            Bs = jnp.stack(all_Bs[s])  # [half_pop, r]
            species_flat[s] = eggroll_update(
                species_flat[s], As, Bs, antithetic_signal, lr
            )

        # Decay LR
        lr = lr * CFG.LR_DECAY

        # ── Diagnostics ──────────────────────────────────────────────────
        if gen % CFG.PRINT_EVERY == 0:
            # Run one clean eval episode
            k_eval, master_key = jax.random.split(master_key)
            eval_params = [unflatten_params(f, t)
                           for f, t in zip(species_flat, species_templates)]
            eval_fitness, eval_diag = run_episode(
                eval_params, species_assignments, k_eval
            )

            elapsed = time.time() - t0
            print(f"Gen {gen:4d} | "
                  f"Fitness: {eval_fitness:8.1f} | "
                  f"Survival: {eval_diag['mean_survival']:5.1f} | "
                  f"Speech: {eval_diag['mean_speech']:4.1f} "
                  f"(div={eval_diag['speech_diversity']:.0f}) | "
                  f"Writing: {eval_diag['mean_writing']:4.1f} "
                  f"(cov={eval_diag['writing_coverage']:.3f}) | "
                  f"LR: {lr:.5f} | "
                  f"t: {elapsed:.1f}s")

            history["fitness"].append(float(eval_fitness))
            history["mean_survival"].append(float(eval_diag["mean_survival"]))
            history["mean_speech"].append(float(eval_diag["mean_speech"]))
            history["speech_diversity"].append(float(eval_diag["speech_diversity"]))
            history["mean_writing"].append(float(eval_diag["mean_writing"]))
            history["writing_coverage"].append(float(eval_diag["writing_coverage"]))
            history["generation"].append(gen)

            # ── Cultural transmission measurement ────────────────────────
            # Compare current writing grid to N generations ago.
            # If agents today are spending more time near old marks,
            # that's evidence writing is influencing behavior.
            current_grid = eval_diag["final_grid"]
            current_writing = current_grid[:, :, CH_WRITE]

            if prev_writing_grid is not None:
                # Simple metric: overlap between old writing pattern and
                # current agent position distribution
                old_mark_mask  = (prev_writing_grid > 0).astype(jnp.float32)
                agent_heatmap  = (current_grid[:, :, CH_AGENT] > 0).astype(jnp.float32)

                # Normalize both
                old_norm   = old_mark_mask / (jnp.sum(old_mark_mask) + 1e-8)
                agent_norm = agent_heatmap / (jnp.sum(agent_heatmap) + 1e-8)

                # KL divergence of agent positions w.r.t. old writing (lower = more attracted)
                # Use simple dot product (higher = more overlap = cultural influence)
                influence = float(jnp.sum(old_norm * agent_norm))
                cultural_influence_scores.append(influence)
                history["cultural_influence"].append(influence)
                print(f"         Cultural influence (writing→agent overlap): {influence:.6f}")
            else:
                history["cultural_influence"].append(0.0)

            # Update writing fingerprint every CULTURAL_SAMPLE_EVERY gens
            if gen % CFG.CULTURAL_SAMPLE_EVERY == 0:
                prev_writing_grid = current_writing

        # ── Checkpoint ───────────────────────────────────────────────────
        if gen % CFG.CHECKPOINT_EVERY == 0 and gen > 0:
            ckpt_path = f"/kaggle/working/checkpoint_gen{gen:04d}.pkl"
            ckpt_data = {
                "gen":              gen,
                "species_flat":     [np.array(f) for f in species_flat],
                "species_templates": species_templates,
                "history":          history,
                "config":           CFG,
            }
            with open(ckpt_path, "wb") as fh:
                pickle.dump(ckpt_data, fh)
            print(f"         ✓ Checkpoint saved: {ckpt_path}")

    print("\nTraining complete.")
    return species_flat, species_templates, history


# %% [markdown]
# ## Cell 10: Visualization & Cultural Analysis

# %%
def ascii_render(grid: jax.Array, label: str = "") -> None:
    """Print an ASCII snapshot of the world grid."""
    H, W = CFG.GRID_H, CFG.GRID_W
    g = np.array(grid)
    if label:
        print(f"\n{'─'*W}")
        print(f"  {label}")
        print(f"{'─'*W}")
    for r in range(H):
        row_str = ""
        for c in range(W):
            if g[r, c, CH_AGENT] > 0:
                sp_id = int(g[r, c, CH_AGENT]) - 1
                row_str += "ABCD"[sp_id % 4]
            elif g[r, c, CH_WRITE] > 0:
                tok = int(g[r, c, CH_WRITE])
                row_str += chr(ord('a') + tok - 1) if 1 <= tok <= 26 else '?'
            elif g[r, c, CH_FOOD] > 0:
                row_str += '*'
            elif g[r, c, CH_ROCK] > 0:
                row_str += '#'
            elif g[r, c, CH_WATER] > 0:
                row_str += '~'
            elif g[r, c, CH_SEED] > 0:
                row_str += '.'
            else:
                row_str += ' '
        print(row_str)
    print()


def plot_training_curves(history: dict) -> None:
    """Plot training metrics with matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for Kaggle

        gens = history["generation"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("EGGROLL Byte-Multi-Agent Training", fontsize=14)

        axes[0,0].plot(gens, history["fitness"])
        axes[0,0].set_title("Total Fitness"); axes[0,0].set_xlabel("Generation")

        axes[0,1].plot(gens, history["mean_survival"])
        axes[0,1].set_title("Mean Survival Ticks"); axes[0,1].set_xlabel("Generation")

        axes[0,2].plot(gens, history["mean_speech"], label="count")
        axes[0,2].set_title("Mean Speech Activity"); axes[0,2].set_xlabel("Generation")

        axes[1,0].plot(gens, history["speech_diversity"])
        axes[1,0].set_title("Speech Token Diversity"); axes[1,0].set_xlabel("Generation")

        axes[1,1].plot(gens, history["writing_coverage"])
        axes[1,1].set_title("Writing Grid Coverage"); axes[1,1].set_xlabel("Generation")

        if any(x > 0 for x in history["cultural_influence"]):
            axes[1,2].plot(gens, history["cultural_influence"])
            axes[1,2].set_title("Cultural Influence Score\n(writing→agent overlap)")
            axes[1,2].set_xlabel("Generation")
        else:
            axes[1,2].text(0.5, 0.5, "Cultural influence\n(builds over training)",
                           ha='center', va='center', transform=axes[1,2].transAxes)

        plt.tight_layout()
        plot_path = "/kaggle/working/training_curves.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"Plot saved: {plot_path}")

    except ImportError:
        print("matplotlib not available, skipping plot.")


def analyze_cultural_transmission(history: dict) -> None:
    """
    Print a summary of cultural transmission evidence.
    Key question: does cultural_influence trend upward over training?
    An upward trend means agents increasingly cluster near old writing marks —
    evidence that writing is shaping behavior across generations.
    """
    ci = [x for x in history["cultural_influence"] if x > 0]
    if len(ci) < 2:
        print("Not enough data for cultural transmission analysis yet.")
        return

    first_half = ci[:len(ci)//2]
    second_half = ci[len(ci)//2:]
    early_mean = np.mean(first_half)
    late_mean  = np.mean(second_half)

    print("\n" + "="*60)
    print("CULTURAL TRANSMISSION ANALYSIS")
    print("="*60)
    print(f"Early training influence:  {early_mean:.6f}")
    print(f"Late training influence:   {late_mean:.6f}")
    print(f"Change:                    {(late_mean - early_mean) / (early_mean + 1e-10) * 100:.1f}%")

    if late_mean > early_mean * 1.1:
        print("\n✓ POSITIVE SIGNAL: Agents increasingly cluster near old writing marks.")
        print("  This is evidence of proto-cultural transmission —")
        print("  behavior is being shaped by marks left by previous agents.")
    elif late_mean > early_mean:
        print("\n~ WEAK SIGNAL: Slight increase in cultural influence.")
        print("  Run more generations or increase population size for stronger signal.")
    else:
        print("\n✗ NO SIGNAL YET: No evidence of writing influencing agent behavior.")
        print("  This is normal early in training. Keep running.")

    print()
    print("Speech token diversity trend:")
    sd = history["speech_diversity"]
    if len(sd) >= 4:
        print(f"  Early: {np.mean(sd[:len(sd)//4]):.1f} unique tokens")
        print(f"  Late:  {np.mean(sd[-len(sd)//4:]):.1f} unique tokens")
    print()

# %% [markdown]
# ## Cell 11: Run Everything

# %%
if __name__ == "__main__":
    # ── Quick smoke-test first (optional, comment out to go straight to full run) ──
    print("Running 2-tick smoke test to verify env + policy work...")
    test_key = jax.random.PRNGKey(0)
    test_params = [init_policy_params(jax.random.fold_in(test_key, s))
                   for s in range(CFG.N_SPECIES)]
    test_assignments = jnp.array([i % CFG.N_SPECIES for i in range(CFG.N_AGENTS)], dtype=jnp.int32)

    # Temporarily shorten episode
    original_ticks = CFG.EPISODE_TICKS
    CFG.EPISODE_TICKS = 2
    test_fitness, test_diag = run_episode(test_params, test_assignments, test_key)
    CFG.EPISODE_TICKS = original_ticks

    print(f"  Smoke test OK. Fitness={test_fitness:.2f}, "
          f"writing_coverage={test_diag['writing_coverage']:.4f}")
    ascii_render(test_diag["final_grid"], label="Initial world (2 ticks)")

    # ── Full training run ──────────────────────────────────────────────────
    print("\nStarting full training...\n")
    species_flat, species_templates, history = training_loop()

    # ── Final eval & visualization ─────────────────────────────────────────
    print("\nRunning final evaluation episode...")
    final_params = [unflatten_params(f, t)
                    for f, t in zip(species_flat, species_templates)]
    final_key = jax.random.PRNGKey(999)
    final_fitness, final_diag = run_episode(final_params, test_assignments, final_key)
    ascii_render(final_diag["final_grid"], label=f"Final world (gen {CFG.N_GENERATIONS})")

    print(f"\nFinal fitness: {final_fitness:.1f}")
    print(f"Final writing coverage: {final_diag['writing_coverage']:.4f}")
    print(f"Final speech activity: {final_diag['mean_speech']:.2f} utterances/agent/episode")

    plot_training_curves(history)
    analyze_cultural_transmission(history)

    # Save final results
    results_path = "/kaggle/working/final_results.pkl"
    with open(results_path, "wb") as fh:
        pickle.dump({
            "species_flat": [np.array(f) for f in species_flat],
            "species_templates": species_templates,
            "history": history,
            "config": CFG,
            "final_fitness": float(final_fitness),
            "final_diagnostics": {
                k: float(v) if not hasattr(v, 'shape') else None
                for k, v in final_diag.items()
                if k != "final_grid"
            }
        }, fh)
    print(f"Results saved: {results_path}")

# %% [markdown]
# ## Appendix: Tips & Next Steps
#
# **To speed up:**
# - Increase `POP_SIZE` to 1024+ (EGGROLL scales near-linearly on TPU)
# - Reduce `EPISODE_TICKS` to 200 for faster generations early on
# - Use `jax.lax.scan` to replace the Python for-loops in `run_episode`
#   (see roadmap below — this is the main optimization left on the table)
#
# **Signs of life to watch for (in order):**
# 1. Mean survival climbing above ~50 ticks (agents finding food)
# 2. Speech token diversity > 5 (not just random noise)
# 3. Writing coverage > 0.05 (agents actually marking the grid)
# 4. Cultural influence score trending upward (the key result)
#
# **Roadmap to make this faster with `jax.lax.scan`:**
# The inner loops in `run_episode` (per-tick, per-agent) are Python loops.
# Replacing them with `jax.lax.scan` would JIT-compile the entire episode
# and give ~10-50x speedup. The reason it's written as Python loops here
# is readability — but the data structures are all JAX arrays, so the
# conversion is straightforward once you want to optimize.
#
# **The cultural transmission metric explained:**
# `cultural_influence` measures the spatial overlap between:
# - Writing marks left by agents in a previous measurement window
# - Agent positions in the current window
#
# If agents are visiting cells that were written on previously (rather than
# randomly distributed), that's evidence that writing is acting as a signal
# that attracts or informs later agents — proto-cultural transmission.
# A rising trend over training is the key result we're looking for.
