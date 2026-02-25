# Atari Jax (Atarax)

A pure-JAX library of Atari 2600 game environments using Brax-style game logic.
All mutable game state lives in an `AtariState` pytree, making each
`step(state, action) → state` call a stateless JAX computation that compiles
with `jax.jit` and batches with `jax.vmap` — no Python loops, no host-side
control flow, no ROM loading.

## Features

- **Brax-style game logic** — each game is a collection of branch-free JAX
  functions operating on a flat `chex.dataclass` pytree. No hardware emulation,
  no ROM files, no `ale-py` dependency.
- **JIT + vmap ready** — the entire stack is written in JAX primitives
  (`jnp.where`, `jax.lax.fori_loop`, `jax.lax.scan`). No Python-level
  branching on traced values.
- **Pytree state** — `AtariState` is a `chex.dataclass` so it works out of the
  box with `jax.tree_util`, `optax`, and `flax`.
- **gymnax-style env API** — `AtariEnv` exposes `reset(key)` / `step(state, action)`
  with external state, fully compatible with `jit`, `vmap`, and `lax.scan`.
- **`make()` / `make_vec()`** — Gymnasium-familiar factory functions returning `Env`
  and `VecEnv`; optional wrapper presets (including the standard DQN stack) and a
  persistent XLA compilation cache built in.
- **`make_multi()` / `make_multi_vec()`** — convenience variants for creating one
  `Env` or `VecEnv` per game from a list of IDs; useful for multi-game training loops.
- **Composable wrappers** — ten wrappers covering RL preprocessing and JIT
  compilation, fully compatible with `jax.jit`, batching, and `lax.scan`. See
  the [Wrappers](#wrappers) table below.
- **Rendering + interactive play** — single-frame rendering and a keyboard-driven
  `play()` loop backed by pygame.

## Requirements

- Python 3.13+
- JAX 0.9+ (CPU, CUDA, or TPU backend)

## Installation

```bash
pip install atarax
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Achronus/atari-jax
cd atari-jax
uv sync
```

## Quick Start

### `make()`

```python
import jax
from atarax.env import make

key = jax.random.PRNGKey(0)

# Raw environment — use the "atari/<name>-v0" string
env = make("atari/breakout-v0")
obs, state = env.reset(key)                   # obs: uint8[210, 160, 3]
obs, state, reward, done, info = env.step(state, env.sample(key))

# Full DQN preprocessing stack (JIT-compiled with XLA cache by default)
env = make("atari/breakout-v0", preset=True)
obs, state = env.reset(key)                   # obs: uint8[84, 84, 4]

# AtariPreprocessing can also be used standalone
from atarax.env import AtariPreprocessing
env = AtariPreprocessing(make("atari/breakout-v0", jit_compile=False))
env = AtariPreprocessing(make("atari/breakout-v0", jit_compile=False), h=42, w=42, n_stack=2)

# Custom wrapper list (applied innermost → outermost)
from atarax.env import GrayscaleObservation, ResizeObservation
env = make("atari/breakout-v0", wrappers=[GrayscaleObservation, ResizeObservation])

# Disable the XLA cache for ephemeral runs
env = make("atari/breakout-v0", preset=True, cache_dir=None)

# Multi-step rollout via lax.scan
import jax.numpy as jnp
actions = jnp.zeros(128, dtype=jnp.int32)
final_state, (obs, reward, done, info) = env.rollout(state, actions)
# obs: uint8[128, 84, 84, 4]
```

### `make_vec()`

```python
import jax
import jax.numpy as jnp
from atarax.env import make_vec

key = jax.random.PRNGKey(0)

# reset() splits the key 32 ways — each env gets a distinct random start
vec_env = make_vec("atari/breakout-v0", n_envs=32, preset=True)
obs, states = vec_env.reset(key)              # obs: uint8[32, 84, 84, 4]

# step() and sample() operate across all 32 envs simultaneously
actions = vec_env.sample(key)                 # int32[32]
obs, states, reward, done, info = vec_env.step(states, actions)

# Multi-step rollout via lax.scan + vmap
actions = jnp.zeros((32, 128), dtype=jnp.int32)
final_states, (obs, reward, done, info) = vec_env.rollout(states, actions)
# obs: uint8[32, 128, 84, 84, 4]
```

### `make_multi()` / `make_multi_vec()`

Create one `Env` or `VecEnv` per game from a list of IDs. Intended for
multi-game training loops where each game needs its own independent environment.

```python
import jax
from atarax.env import make_multi, make_multi_vec

key = jax.random.PRNGKey(0)

# Returns List[Env] — one per game
envs = make_multi(["atari/breakout-v0"])
obs, state = envs[0].reset(key)              # obs: uint8[210, 160, 3]

# Vectorized variant — Returns List[VecEnv]
vec_envs = make_multi_vec(["atari/breakout-v0"], n_envs=16)
obs, states = vec_envs[0].reset(key)         # obs: uint8[16, 210, 160, 3]

# Use a predefined group (games not yet implemented raise ValueError at make time)
from atarax.games.registry import GAME_GROUPS
envs = make_multi(GAME_GROUPS["atari5"], preset=True)
```

Predefined groups:

| Group | Size | Games |
| --- | --- | --- |
| `"atari5"` | 5 | `breakout`, `ms_pacman`, `pong`, `qbert`, `space_invaders` |
| `"atari10"` | 10 | `alien`, `beam_rider`, `breakout`, `enduro`, `montezuma_revenge`, … |
| `"atari26"` | 26 | Standard Mnih 26-game subset |
| `"atari57"` | 57 | Full Mnih et al. (2015) benchmark suite |

### Rendering and Interactive Play

```python
import jax
from atarax.utils.render import play, render
from atarax.env import make

key = jax.random.PRNGKey(0)

# Render a single frame in a pygame window
env = make("atari/breakout-v0")
obs, state = env.reset(key)
render(obs)                                  # 320×420 window (scale=2 default)
render(obs, scale=4, caption="Breakout")

# Play a game interactively (keyboard control, native 210×160 RGB)
play("atari/breakout-v0")                    # scale=3 default → 480×630 window
play("atari/breakout-v0", scale=2, fps=30)
```

Keyboard controls for `play()`:

| Key | Action |
| --- | --- |
| Arrow keys / `W A S D` | Movement |
| `Space` | Fire |
| `Esc` / close window | Quit |

## Wrappers

Ten composable wrappers, each accepting any `Env` and exposing the same
`reset(key)` / `step(state, action)` interface.

| Wrapper | Input | Output | Description | Extra state |
| --- | --- | --- | --- | --- |
| `AtariPreprocessing` | `uint8[210, 160, 3]` | `uint8[84, 84, 4]` | Full DQN stack (six wrappers applied) | `EpisodeStatisticsState` |
| `GrayscaleObservation` | `uint8[210, 160, 3]` | `uint8[210, 160]` | NTSC luminance conversion | — |
| `ResizeObservation(h, w)` | `uint8[H, W]` | `uint8[h, w]` | Bilinear resize (default 84×84) | — |
| `NormalizeObservation` | `uint8[...]` | `float32[...]` in `[0, 1]` | Divide by 255 | — |
| `FrameStackObservation(n_stack)` | `uint8[H, W]` | `uint8[H, W, n_stack]` | Rolling frame buffer (default 4) | `FrameStackState` |
| `ClipReward` | any reward | `float32 ∈ {−1, 0, +1}` | Sign clipping | — |
| `ExpandDims` | any env | same obs | Adds a trailing `1` dim to `reward` and `done` | — |
| `EpisodeDiscount` | any env | same obs | Converts `done` bool to float32 discount (`1.0` continues, `0.0` terminated) | — |
| `EpisodicLife` | any env | same obs | Terminal on every life loss | `EpisodicLifeState` |
| `RecordEpisodeStatistics` | any env | same obs | Tracks episode return + length in `info["episode"]` | `EpisodeStatisticsState` |
| `JitWrapper` | any env | same obs | JIT-compiles `reset` + `step` with a warmup pass; applied automatically by `make()` when `jit_compile=True` | — |

Stateless wrappers pass the inner state through unchanged. Stateful wrappers
return a `chex.dataclass` pytree that carries extra data alongside the inner
state — both are fully compatible with `jit`, `vmap`, and `lax.scan`.

`JitWrapper` can also be used standalone to eagerly compile any env:

```python
from atarax.env import JitWrapper, make

env = JitWrapper(make("atari/breakout-v0", jit_compile=False))
```

### Standard DQN preprocessing stack

The standard Mnih et al. (2015) observation pipeline:

```python
import jax
from atarax.env import make

env = make("atari/breakout-v0", preset=True)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)                  # obs: uint8[84, 84, 4]
obs, state, reward, done, info = env.step(state, env.sample(key))
# done                    → True on life loss or game over
# reward                  → clipped to {-1, 0, +1}
# info["episode"]["r"]    → episode return (non-zero at episode end)
# info["episode"]["l"]    → episode length (non-zero at episode end)
```

## Supported Games

Atarax targets all 57 Mnih et al. (2015) Atari environments with more added in
each release. Currently implemented:

| Game | `make()` ID |
| --- | --- |
| Breakout | `"atari/breakout-v0"` |

## Architecture Notes

`AtariState` is a flat `chex.dataclass` pytree — every field is a fixed-shape
JAX array, making it compatible with `jax.vmap`, `jax.lax.scan`, and any
JAX-native optimiser. Game implementations extend the three-level state
hierarchy:

| Class | Fields |
| --- | --- |
| `GameState` | `reward`, `done`, `step`, `episode_step` |
| `AtariState(GameState)` | + `lives`, `score` |
| `BreakoutState(AtariState)` | + game-specific fields (`ball_x`, `bricks`, …) |

All game logic is branch-free — every conditional uses a JAX primitive so the
computation graph is fixed at trace time:

| Situation | Pattern used |
| --- | --- |
| Conditional state update | `jnp.where` |
| Fixed-count loop (e.g. frame skip) | `jax.lax.fori_loop` |
| Multi-step rollout | `jax.lax.scan` |

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
