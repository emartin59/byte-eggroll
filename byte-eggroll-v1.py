# %% [markdown]
# # EGGROLL-Accelerated Byte-Multi-Agent: v2 (TPU Optimized)
# 
# **Scientific Bet:** Can writing accumulate across generations to shape behavior?
# **Optimizations:** # 1. `jax.lax.scan` for tick-loops (replaces Python for-loops).
# 2. `jax.vmap` for agent-parallelism within ticks.
# 3. `jax.pmap` to distribute the population across 8 TPU cores.

# %% [markdown]
# ## Cell 1: Install Dependencies
# Optimized for subprocess flags.

# %%
import subprocess, sys

def install(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(args))

install("jax[tpu]", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html")
install("flax")
install("optax")

# %% [markdown]
# ## Cell 2: Verify TPU

# %%
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import time
import pickle

print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
print("Device count:", jax.device_count())
assert jax.device_count() >= 8, "This optimized version requires all 8 TPU cores."

# %% [markdown]
# ## Cell 3: Configuration

# %%
from dataclasses import dataclass

@dataclass
class Config:
    # World
    GRID_W: int = 64
    GRID_H: int = 64
    N_AGENTS: int = 8
    N_SPECIES: int = 4
    EPISODE_TICKS: int = 400

    # Agent architecture
    VISION_R: int = 7
    HIDDEN_DIM: int = 256
    AUDIO_TOKS: int = 27
    AUDIO_BUF: int = 3
    AUDIO_RANGE: int = 10

    # Actions
    N_MOVE: int = 5
    N_SPEAK: int = 27
    N_WRITE: int = 27
    N_INV: int = 4

    # Economy
    INIT_ENERGY: float = 100.0
    METABOLIC_COST: float = 0.5
    MOVE_COST: float = 1.0
    SPEAK_COST: float = 0.3
    WRITE_COST: float = 0.5
    FOOD_ENERGY: float = 30.0
    FOOD_DENSITY: float = 0.08
    SEED_REGROW_PROB: float = 0.02

    # EGGROLL ES
    POP_SIZE: int = 512
    N_ENVS_PER_MEMBER: int = 2
    NOISE_STD: float = 0.05
    EGGROLL_RANK: int = 1
    LR: float = 0.01
    LR_DECAY: float = 0.999
    WEIGHT_DECAY: float = 1e-4

    # Training
    N_GENERATIONS: int = 2000
    PRINT_EVERY: int = 25
    CHECKPOINT_EVERY: int = 50

    # Fitness weights
    W_SURVIVAL: float = 1.0
    W_ENERGY: float = 0.5
    W_FINAL_ENERGY: float = 0.1

    CULTURAL_SAMPLE_EVERY: int = 100

CFG = Config()

# Channel indices
CH_FOOD, CH_SEED, CH_ROCK, CH_WATER, CH_WRITE, CH_TOOL, CH_AGENT, CH_ENERGY = range(8)
N_CHANNELS = 8

# %% [markdown]
# ## Cell 4: Optimized World Utilities

# %%
def init_world(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    H, W = CFG.GRID_H, CFG.GRID_W
    grid = jnp.zeros((H, W, N_CHANNELS))
    grid = grid.at[:, :, CH_FOOD].set(jax.random.bernoulli(k1, CFG.FOOD_DENSITY, (H, W)))
    grid = grid.at[:, :, CH_SEED].set(jax.random.bernoulli(k2, CFG.FOOD_DENSITY/2, (H, W)))
    grid = grid.at[:, :, CH_ROCK].set(jax.random.bernoulli(k3, 0.05, (H, W)))
    grid = grid.at[:, :, CH_WATER].set(jax.random.bernoulli(k4, 0.04, (H, W)))
    return grid

def place_agents(key, grid):
    H, W = CFG.GRID_H, CFG.GRID_W
    k1, k2 = jax.random.split(key)
    rows = jax.random.randint(k1, (CFG.N_AGENTS,), 1, H-1)
    cols = jax.random.randint(k2, (CFG.N_AGENTS,), 1, W-1)
    pos = jnp.stack([rows, cols], axis=-1)
    energies = jnp.full((CFG.N_AGENTS,), CFG.INIT_ENERGY)

    def place_one(carry, x):
        g, i, r, c = carry, x[0], x[1], x[2]
        # Fixed: Using JAX-native casting to avoid ConcretizationTypeError
        g = g.at[r, c, CH_AGENT].set((i + 1).astype(jnp.float32))
        g = g.at[r, c, CH_ENERGY].set(CFG.INIT_ENERGY)
        return g, None

    grid, _ = jax.lax.scan(place_one, grid, (jnp.arange(CFG.N_AGENTS), rows, cols))
    return grid, pos, energies

@jax.vmap
def get_vision_vmap(grid, pos):
    R = CFG.VISION_R
    H, W = CFG.GRID_H, CFG.GRID_W
    rows = (jnp.arange(-R, R + 1) + pos[0]) % H
    cols = (jnp.arange(-R, R + 1) + pos[1]) % W
    return grid[rows[:, None], cols[None, :], :].reshape(-1)

# %% [markdown]
# ## Cell 5: Policy & EGGROLL Math

# %%
import flax.linen as nn

VISION_DIM = (2 * CFG.VISION_R + 1)**2 * N_CHANNELS
INPUT_DIM = VISION_DIM + (CFG.AUDIO_BUF * CFG.AUDIO_TOKS) + 6

class AgentPolicy(nn.Module):
    @nn.compact
    def __call__(self, obs, hidden):
        x = nn.tanh(nn.Dense(CFG.HIDDEN_DIM)(obs))
        # GRU Logic
        gates = nn.Dense(2 * CFG.HIDDEN_DIM)(jnp.concatenate([x, hidden]))
        r, z = jnp.split(nn.sigmoid(gates), 2, axis=-1)
        h_cand = jnp.tanh(nn.Dense(CFG.HIDDEN_DIM)(jnp.concatenate([x, r * hidden])))
        new_h = (1 - z) * hidden + z * h_cand
        
        logits = nn.Dense(CFG.N_MOVE + CFG.N_SPEAK + CFG.N_WRITE + CFG.N_INV)(new_h)
        return logits, new_h

def flatten_params(params):
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([x.ravel() for x in leaves])

def unflatten_params(flat, template):
    leaves, treedef = jax.tree_util.tree_flatten(template)
    sizes = [x.size for x in leaves]
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    chunks = jnp.split(flat, splits)
    restored = [c.reshape(l.shape) for c, l in zip(chunks, leaves)]
    return treedef.unflatten(restored)

# EGGROLL ES functions
def eggroll_antithetic_pair(key, base_flat):
    d = base_flat.shape[0]
    r = CFG.EGGROLL_RANK
    k1, k2 = jax.random.split(key)
    A, B = jax.random.normal(k1, (d, r)), jax.random.normal(k2, (r,))
    E = (1.0 / jnp.sqrt(float(r))) * (A @ B)
    return base_flat + CFG.NOISE_STD * E, base_flat - CFG.NOISE_STD * E, A, B

def eggroll_update(base_flat, As, Bs, shaped_fitness, lr):
    r = CFG.EGGROLL_RANK
    fBs = shaped_fitness[:, None] * Bs
    delta = jnp.einsum('idr,ir->d', As, fBs) / (jnp.sqrt(float(r)) * float(shaped_fitness.shape[0]))
    return (base_flat + lr * delta) * (1.0 - CFG.WEIGHT_DECAY)

def rank_shape(fitness):
    ranks = jnp.argsort(jnp.argsort(fitness))
    return ranks.astype(jnp.float32) / float(fitness.shape[0] - 1) - 0.5

# %% [markdown]
# ## Cell 6: Optimized Episode Rollout (JIT/Scan Compatible)

# %%
MOVE_DELTAS = jnp.array([[0,0], [-1,0], [1,0], [0,-1], [0,1]])

def step_world_jit(grid, key):
    grow = jax.random.bernoulli(key, CFG.SEED_REGROW_PROB, (CFG.GRID_H, CFG.GRID_W))
    new_food = jnp.minimum(grid[..., CH_FOOD] + (grid[..., CH_SEED] * grow), 1.0)
    return grid.at[..., CH_FOOD].set(new_food)

def run_episode_jit(all_params_flat, species_assignments, key, templates):
    # Initialize
    k_w, k_a, k_run = jax.random.split(key, 3)
    grid = init_world(k_w)
    grid, pos, energies = place_agents(k_a, grid)
    hiddens = jnp.zeros((CFG.N_AGENTS, CFG.HIDDEN_DIM))
    audio_bufs = jnp.zeros((CFG.N_AGENTS, CFG.AUDIO_BUF, CFG.AUDIO_TOKS))
    
    # State carry for lax.scan
    init_state = (grid, pos, energies, hiddens, audio_bufs, 
                  jnp.zeros(CFG.N_AGENTS), jnp.zeros(CFG.N_AGENTS)) # survival, energy_earned
    
    def tick_fn(state, tick_key):
        grid, pos, energies, hiddens, audio_bufs, surv, earned = state
        
        # 1. Observations
        vision = get_vision_vmap(grid, pos)
        propr = jnp.stack([energies/CFG.INIT_ENERGY, 
                           jnp.zeros(CFG.N_AGENTS), jnp.zeros(CFG.N_AGENTS), jnp.zeros(CFG.N_AGENTS), jnp.zeros(CFG.N_AGENTS),
                           species_assignments.astype(jnp.float32)/CFG.N_SPECIES], axis=-1)
        obs = jnp.concatenate([vision, audio_bufs.reshape(CFG.N_AGENTS, -1), propr], axis=-1)
        
        # 2. Policy (Vmapped)
        def agent_policy_apply(p_flat, o, h, temp):
            p_dict = unflatten_params(p_flat, temp)
            return AgentPolicy().apply(p_dict, o, h)
        
        # We assume one dominant set of params for the vectorized call or switch
        # For multi-species optimization, we index params by species_assignments
        member_params = jnp.take(jnp.stack(all_params_flat), species_assignments, axis=0)
        logits, new_hiddens = jax.vmap(agent_policy_apply, in_axes=(0, 0, 0, None))(
            member_params, obs, hiddens, templates[0]
        )
        
        # 3. Actions & Resolution (Simplified Vectorized resolution)
        k_move, k_speak, k_write, k_world = jax.random.split(tick_key, 4)
        move_acts = jax.random.categorical(k_move, logits[:, :5])
        speak_toks = jax.random.categorical(k_speak, logits[:, 5:32])
        write_toks = jax.random.categorical(k_write, logits[:, 32:59])
        
        # Movement & Energy logic
        new_pos = (pos + MOVE_DELTAS[move_acts]) % jnp.array([CFG.GRID_H, CFG.GRID_W])
        alive = energies > 0
        
        # 4. Sequential Grid Updates (via small scan to prevent race conditions)
        def update_grid_seq(g_carry, i):
            idx = i
            p, a, s_tok, w_tok = new_pos[idx], alive[idx], speak_toks[idx], write_toks[idx]
            food = g_carry[p[0], p[1], CH_FOOD]
            
            g = g_carry.at[pos[idx,0], pos[idx,1], CH_AGENT].set(0.0)
            g = jax.lax.cond(a, 
                lambda: g.at[p[0], p[1], CH_AGENT].set((species_assignments[idx]+1).astype(jnp.float32))
                         .at[p[0], p[1], CH_FOOD].set(0.0)
                         .at[p[0], p[1], CH_SEED].set(jnp.where(food > 0, 1.0, g[p[0], p[1], CH_SEED])),
                lambda: g)
            # Writing
            g = jax.lax.cond(a & (w_tok > 0), 
                lambda: g.at[p[0], p[1], CH_WRITE].set(w_tok.astype(jnp.float32)), 
                lambda: g)
            return g, food
            
        grid, food_eaten = jax.lax.scan(update_grid_seq, grid, jnp.arange(CFG.N_AGENTS))
        
        # Energy updates
        costs = CFG.METABOLIC_COST + jnp.where(move_acts>0, CFG.MOVE_COST, 0.) + jnp.where(speak_toks>0, CFG.SPEAK_COST, 0.)
        new_energies = jnp.maximum(0, energies - costs + food_eaten * CFG.FOOD_ENERGY)
        
        # Audio (O(N^2) vectorized distance)
        dist = jnp.max(jnp.abs(new_pos[:, None] - new_pos[None, :]), axis=-1)
        heard = (dist <= CFG.AUDIO_RANGE)[:, :, None] * jax.nn.one_hot(speak_toks, CFG.AUDIO_TOKS)[None, :, :]
        new_audio = jnp.concatenate([heard.sum(axis=1)[:, None, :], audio_bufs[:, :2, :]], axis=1)
        
        grid = step_world_jit(grid, k_world)
        
        new_state = (grid, new_pos, new_energies, new_hiddens, new_audio, 
                     surv + alive, earned + food_eaten * CFG.FOOD_ENERGY)
        return new_state, None

    final_state, _ = jax.lax.scan(tick_fn, init_state, jax.random.split(k_run, CFG.EPISODE_TICKS))
    
    # Fitness Calculation
    surv_ticks, energy_val, final_e = final_state[5], final_state[6], final_state[2]
    fitness = CFG.W_SURVIVAL * jnp.sum(surv_ticks) + CFG.W_ENERGY * jnp.sum(energy_val) + CFG.W_FINAL_ENERGY * jnp.sum(final_e)
    
    diag = {"writing_coverage": jnp.mean(final_state[0][..., CH_WRITE] > 0), "final_grid": final_state[0]}
    return fitness, diag

# %% [markdown]
# ## Cell 7: Multi-Core PMAP Training Loop

# %%
@partial(jax.pmap, in_axes=(0, None, 0, None))
def evaluate_population(pop_params, species_assignments, keys, templates):
    # pop_params: [Cores, Pop/Cores, Species, D]
    def single_eval(p_list, k):
        f1, _ = run_episode_jit(p_list, species_assignments, k, templates)
        return f1
    return jax.vmap(single_eval)(pop_params, keys)

def training_loop():
    master_key = jax.random.PRNGKey(42)
    templates = [flatten_params(AgentPolicy().init(jax.random.PRNGKey(i), jnp.zeros(INPUT_DIM), jnp.zeros(CFG.HIDDEN_DIM))) for i in range(CFG.N_SPECIES)]
    species_flat = jnp.stack(templates)
    
    # Population setup for 8 cores
    n_cores = jax.device_count()
    pop_per_core = (CFG.POP_SIZE // 2) // n_cores # Antithetic pairs
    
    history = {"fitness": [], "writing_coverage": [], "generation": []}
    lr = CFG.LR
    
    for gen in range(CFG.N_GENERATIONS):
        t0 = time.time()
        k_gen, master_key = jax.random.split(master_key)
        
        # Generate perturbations
        pert_keys = jax.random.split(k_gen, CFG.POP_SIZE // 2)
        all_pos, all_neg, all_A, all_B = [], [], [], []
        
        for s in range(CFG.N_SPECIES):
            p, n, A, B = jax.vmap(eggroll_antithetic_pair, in_axes=(0, None))(pert_keys, species_flat[s])
            all_pos.append(p); all_neg.append(n); all_A.append(A); all_B.append(B)
            
        # Reshape for PMAP: [8 cores, pop_per_core, N_SPECIES, D]
        def prep_pop(data_list):
            return jnp.stack(data_list, axis=1).reshape(n_cores, pop_per_core, CFG.N_SPECIES, -1)
        
        pos_pop = prep_pop(all_pos)
        neg_pop = prep_pop(all_neg)
        pmap_keys = jax.random.split(master_key, n_cores)
        
        # Parallel Eval
        assigns = jnp.array([i % CFG.N_SPECIES for i in range(CFG.N_AGENTS)])
        f_pos = evaluate_population(pos_pop, assigns, pmap_keys, templates).ravel()
        f_neg = evaluate_population(neg_pop, assigns, pmap_keys, templates).ravel()
        
        # Update
        shaped = rank_shape(jnp.concatenate([f_pos, f_neg]))
        antithetic_signal = shaped[:CFG.POP_SIZE//2] - shaped[CFG.POP_SIZE//2:]
        
        new_species_flat = []
        for s in range(CFG.N_SPECIES):
            new_f = eggroll_update(species_flat[s], all_A[s], all_B[s], antithetic_signal, lr)
            new_species_flat.append(new_f)
        species_flat = jnp.stack(new_species_flat)
        lr *= CFG.LR_DECAY
        
        if gen % CFG.PRINT_EVERY == 0:
            eval_f, eval_diag = run_episode_jit(species_flat, assigns, jax.random.PRNGKey(gen), templates)
            print(f"Gen {gen:4d} | Fitness: {eval_f:8.1f} | Writing Cov: {eval_diag['writing_coverage']:.4f} | t: {time.time()-t0:.1f}s")
            history["fitness"].append(float(eval_f))
            history["writing_coverage"].append(float(eval_diag["writing_coverage"]))
            history["generation"].append(gen)

    return species_flat, history

# %% [markdown]
# ## Cell 8: Execution

# %%
if __name__ == "__main__":
    print("Starting Optimized TPU Training (v2)...")
    final_params, history = training_loop()
    print("Training Complete.")
