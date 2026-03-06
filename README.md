# Atarax

A pure-JAX library of Atari 2600 game environments designed to **match the
structural RL challenge of each game, not its pixel fidelity**. Each game is
reimplemented from scratch in JAX: branch-free physics, procedural rendering
with distinct entity colours, and motion trails that encode velocity into a
single RGB frame — without ROM loading, C++ dependencies, or hardware artefacts.

Each game's mutable state lives in an `AtariState` pytree, making every
`step(rng, state, action, params) → (obs, state, reward, done, info)` call a
stateless JAX computation that compiles with `jax.jit` and batches with
`jax.vmap` — no Python loops, no host-side control flow, no ROM loading.

Part of the [Envrax suite](https://github.com/Achronus/envrax) — installs
`envrax` automatically.

## Design Philosophy

> **Match the challenge, not the chrome.**

Atarax aims to preserve the eight structural properties that make each Atari
game a meaningful RL benchmark, while discarding hardware artefacts that
have no bearing on the learning problem:

| # | Property | Description |
|---|-----------|-------------|
| 1 | **Exploration difficulty** | Sparse or delayed reward structures that require sustained exploration |
| 2 | **Credit assignment horizon** | Long action-to-outcome delays that stress temporal credit assignment |
| 3 | **Observation complexity** | Multiple moving entities at different spatial scales |
| 4 | **Task diversity** | A variety of sub-goals within a single episode |
| 5 | **Generalisation pressure** | Randomised or procedurally varied states that prevent memorisation |
| 6 | **Partial observability** | Entities that appear, disappear, or occlude each other |
| 7 | **Non-stationarity** | Changing enemy behaviour, speed escalation, or level progression |
| 8 | **Reward density** | Balance between dense shaping signals and sparse terminal rewards |

### Rendering approach

Procedural RGB rendering with **distinct colours per entity type** is
sufficient and preferable to sprite art for RL. A CNN policy requires:

- **Spatial distinctness** — entities must differ from background and each other
- **Positional accuracy** — entity bounding boxes must be at their physics coordinates
- **Colour identity** — entity type must be inferrable from colour alone

**Standard palette** (thematic deviations allowed provided all entities are
distinct within a game):

| Entity type | Colour | RGB |
|-------------|--------|-----|
| Player | Green | `[92, 186, 92]` |
| Enemy | Red | `[213, 80, 80]` |
| Projectile | Yellow | `[255, 255, 100]` |
| Ball | White | `[255, 255, 255]` |
| Pickup / treasure | Blue | `[100, 180, 255]` |
| Wall / structure | Grey | `[140, 140, 140]` |

### Motion trails

Every moving entity draws a faded ghost at its previous position
`(x − dx, y − dy)` at 33% brightness before rendering the entity itself.
This encodes velocity into a single RGB frame, removing the need for frame
stacking to infer motion direction and speed.

## Features

- **Branch-free game logic** — every game is a collection of JAX functions
  operating on a flat `chex.dataclass` pytree. All conditionals use
  `jnp.where`; no Python branching on traced values.
- **JIT + vmap ready** — the entire stack is written in JAX primitives
  (`jnp.where`, `jax.lax.fori_loop`, `jax.lax.scan`).
- **Pytree state** — `AtariState` is a `chex.dataclass` so it works out of
  the box with `jax.tree_util`, `optax`, and `flax`.
- **gymnax-style env API** — `AtaraxGame` exposes
  `reset(rng, params) → (obs, state)` and
  `step(rng, state, action, params) → (obs, state, reward, done, info)` with
  external state, fully compatible with `jit`, `vmap`, and `lax.scan`.
- **`envrax.make()` / `envrax.make_vec()`** — factory functions returning
  `(env, params)` and `(VmapEnv, params)` tuples; optional wrapper lists and
  a persistent XLA compilation cache built in.
- **`envrax.make_multi()` / `envrax.make_multi_vec()`** — convenience variants
  for creating one environment per game from a list of IDs; useful for
  multi-game training loops.
- **Composable wrappers** — two Atari-native wrappers (`AtariPreprocessing`,
  `EpisodicLife`) plus `JitWrapper` and additional generic wrappers from
  `envrax`, all re-exported through `atarax.wrappers`.
  See the [Wrappers](#wrappers) section below.
- **Rendering + interactive play** — single-frame rendering and a
  keyboard-driven `env.play()` loop backed by pygame.
- **All 57 Mnih et al. (2015) games** — every game passes the full smoke
  test suite (`reset`, `step`, `render`, `vmap`, `jit`, `pytree`).

## Requirements

- Python 3.13+
- JAX 0.9+ (CPU, CUDA, or TPU backend)

## Installation

```bash
pip install atarax
```

To also enable gif/mp4 recording (`record_episode`):

```bash
pip install "atarax[viz]"
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Achronus/atari-jax
cd atari-jax
uv sync
```

## Quick Start

### `envrax.make()`

`import atarax` automatically registers all Atari games into the envrax
registry. Use `envrax.make()` directly:

```python
import jax
import jax.numpy as jnp
import envrax
import atarax  # registers "atari/*-v0" environments

key = jax.random.PRNGKey(0)

# Raw environment — JIT-compiled by default
env, params = envrax.make("atari/breakout-v0")
obs, state = env.reset(key, params)           # obs: uint8[210, 160, 3]
obs, state, reward, done, info = env.step(key, state, jnp.int32(0), params)

# DQN preprocessing stack — classic grayscale 4-frame stack
from atarax.wrappers import AtariPreprocessing
env, params = envrax.make("atari/breakout-v0", wrappers=[AtariPreprocessing])
obs, state = env.reset(key, params)           # obs: uint8[84, 84, 4]

# Custom wrapper list (applied innermost → outermost)
from envrax.wrappers import GrayscaleObservation, ResizeObservation
env, params = envrax.make("atari/breakout-v0", wrappers=[GrayscaleObservation, ResizeObservation])

# Disable the XLA cache for ephemeral runs
env, params = envrax.make("atari/breakout-v0", cache_dir=None)

# Multi-step rollout via lax.scan
def rollout_step(carry, _):
    key, state = carry
    key, subkey = jax.random.split(key)
    obs, state, reward, done, info = env.step(subkey, state, jnp.int32(0), params)
    return (key, state), (obs, reward, done)

obs, state = env.reset(key, params)
(_, final_state), (obs_seq, rewards, dones) = jax.lax.scan(
    rollout_step, (key, state), None, length=128
)
# obs_seq: uint8[128, 210, 160, 3]
```

### `envrax.make_vec()`

```python
import jax
import jax.numpy as jnp
import envrax
import atarax

key = jax.random.PRNGKey(0)

# reset() splits the key 32 ways — each env gets a distinct random start
vec_env, params = envrax.make_vec("atari/breakout-v0", n_envs=32)
obs, states = vec_env.reset(key, params)      # obs: uint8[32, 210, 160, 3]

# step() operates across all 32 envs simultaneously
actions = jnp.zeros(32, dtype=jnp.int32)
obs, states, reward, done, info = vec_env.step(key, states, actions, params)

# Multi-step rollout via lax.scan
def vec_step(carry, _):
    key, states = carry
    key, subkey = jax.random.split(key)
    obs, states, reward, done, info = vec_env.step(subkey, states, actions, params)
    return (key, states), (obs, reward, done)

obs, states = vec_env.reset(key, params)
(_, final_states), (obs_seq, rewards, dones) = jax.lax.scan(
    vec_step, (key, states), None, length=128
)
# obs_seq: uint8[128, 32, 210, 160, 3]
```

### `envrax.make_multi()` / `envrax.make_multi_vec()`

Create one `(env, params)` or `(VmapEnv, params)` tuple per game from a list
of IDs. Intended for multi-game training loops where each game needs its own
independent environment.

```python
import jax
import envrax
import atarax

key = jax.random.PRNGKey(0)

# Returns List[Tuple[JaxEnv, EnvParams]] — one per game
results = envrax.make_multi(["atari/breakout-v0", "atari/asteroids-v0"])
env, params = results[0]
obs, state = env.reset(key, params)          # obs: uint8[210, 160, 3]

# Vectorized variant — Returns List[Tuple[VmapEnv, EnvParams]]
vec_results = envrax.make_multi_vec(["atari/breakout-v0"], n_envs=16)
vec_env, params = vec_results[0]
obs, states = vec_env.reset(key, params)     # obs: uint8[16, 210, 160, 3]

# List all registered games
print(envrax.registered_names())
```

### Rendering and Interactive Play

```python
import jax
import numpy as np
from atarax.env.games.breakout import Breakout
from atarax.game import AtaraxParams
from atarax.render import play, render_grid, record_episode

game   = Breakout()
params = AtaraxParams()

# Render a single frame — uint8[210, 160, 3], score + lives HUD included
obs, state = game.reset(jax.random.PRNGKey(0), params)
frame = game.render(state)

# Interactive pygame window (requires pygame) — Esc or close to quit
play("atari/breakout-v0", scale=3, fps=15)
```

#### Multi-environment grid

Tile N vmap'd frames into a single image — useful for monitoring parallel
training runs at a glance:

```python
N = 16
rngs = jax.random.split(jax.random.PRNGKey(0), N)
_, states = jax.vmap(game.reset, in_axes=(0, None))(rngs, params)
frames = np.asarray(jax.vmap(game.render)(states))  # (16, 210, 160, 3)

grid = render_grid(frames, nrow=4)   # (840, 640, 3)
```

#### Recording episodes

Save a gif or mp4 of a full episode (requires `pip install "atarax[viz]"`):

```python
_rng = jax.random.PRNGKey(42)

def random_policy(obs):
    global _rng
    _rng, key = jax.random.split(_rng)
    return jax.random.randint(key, shape=(), minval=0, maxval=4)

record_episode(game, params, random_policy, path="episode.gif", fps=15)
```

Per-game keyboard controls for interactive play are defined in each game's
`_key_map()` method.

## Wrappers

Composable wrappers, each accepting any `AtaraxGame` (or wrapped env) and
exposing the same `reset(rng, params)` / `step(rng, state, action, params)`
interface.

| Wrapper | Input | Output | Description | Extra state |
| --- | --- | --- | --- | --- |
| `AtariPreprocessing(n_stack=4)` | `uint8[210, 160, 3]` | `uint8[84, 84, 4]` | DQN stack: grayscale → resize → 4-frame stack + clip reward + episodic life + record stats | `EpisodeStatisticsState` |
| `EpisodicLife` | any env | same obs | Terminal on every life loss | `EpisodicLifeState` |
| `JitWrapper` | any env | same obs | JIT-compiles `reset` + `step` with a warmup pass; from `envrax`, applied automatically by `make()` when `jit_compile=True` | — |

Additional generic wrappers (grayscale, resize, normalize, frame-stack, clip
reward, episode statistics, and more) are available via
[`envrax`](https://github.com/Achronus/envrax) and re-exported through
`atarax.wrappers`.

`JitWrapper` can also be used standalone:

```python
import envrax
import atarax
from envrax.wrappers import JitWrapper

env, params = envrax.make("atari/breakout-v0", jit_compile=False)
env = JitWrapper(env)
```

### Observation format

The raw observation is a full-colour RGB frame — motion trails encode velocity
directly so no frame stacking is needed for basic experiments:

```python
import jax
import jax.numpy as jnp
import envrax
import atarax

key = jax.random.PRNGKey(0)
env, params = envrax.make("atari/breakout-v0")

obs, state = env.reset(key, params)          # obs: uint8[210, 160, 3]
obs, state, reward, done, info = env.step(key, state, jnp.int32(0), params)
# done                    → True on life loss or game over
# info["lives"]           → remaining lives
# info["score"]           → current score
```

For the classic Mnih et al. (2015) 4-frame grayscale stack:

```python
from atarax.wrappers import AtariPreprocessing

env, params = envrax.make("atari/breakout-v0", wrappers=[AtariPreprocessing])
obs, state = env.reset(key, params)          # obs: uint8[84, 84, 4]
obs, state, reward, done, info = env.step(key, state, jnp.int32(0), params)
# reward                  → clipped to {-1, 0, +1}
# info["episode"]["r"]    → episode return (non-zero at episode end)
# info["episode"]["l"]    → episode length (non-zero at episode end)
```

## Games

All 57 Mnih et al. (2015) games are implemented. Use `"atari/<name>-v0"` as the
`envrax.make()` ID — see [Quick Start](#envraxmake) for examples.

Each game is validated against the reference ALE C++ engine using a statistical
random-policy calibration. We target a ≤5% deviation from the ALE baseline
(ratio within [0.95×, 1.05×], 1× optimal). For the full calibration methodology,
fidelity bands, and known deviations, see [docs/fidelity_testing.md](docs/fidelity_testing.md).

## Architecture Notes

Game implementations use a three-level `chex.dataclass` pytree hierarchy
(defined in `atarax.state`):

| Class | Fields |
| --- | --- |
| `GameState` | `reward`, `done`, `step`, `episode_step` |
| `AtariState(GameState)` | + `lives`, `score`, `level`, `key` |
| `BreakoutState(AtariState)` | + game-specific fields (`ball_x`, `bricks`, …) |

A **unified** `AtaraxState` (defined in `atarax.state`) provides the full
fixed-capacity pytree schema — entity tables, projectile tables, grid fields,
polar physics — enabling `jax.vmap` across all games simultaneously via a
single shared pytree shape.

All game logic is branch-free — every conditional uses a JAX primitive so the
computation graph is fixed at trace time:

| Situation | Pattern used |
| --- | --- |
| Conditional state update | `jnp.where` |
| Fixed-count loop (e.g. 4× frame skip) | `jax.lax.fori_loop` |
| Multi-step rollout | `jax.lax.scan` |

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
