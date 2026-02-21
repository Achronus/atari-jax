# Atari Jax (Atarax)

A pure-JAX implementation of all 57 standard Atari 2600 ALE environments.
All mutable emulator state lives in an `AtariState` pytree, making each
`step(state, action) → state` call a stateless JAX computation that compiles
with `jax.jit` and batches with `jax.vmap` — no Python loops, no host-side
control flow.

## Features

- **Hardware-accurate emulation** — MOS 6507 CPU (all 56 legal opcodes), M6532
  RIOT chip, full NTSC TIA rasteriser (background, playfield, players, missiles,
  ball, collisions, HMOVE).
- **JIT + vmap ready** — the entire emulation stack is written in JAX primitives
  (`jax.lax.switch`, `jax.lax.fori_loop`, `jnp.where`). No Python-level
  branching on traced values.
- **All 57 Mnih et al. (2015) games** — reward extraction, terminal detection,
  and lives counters sourced directly from the ALE reference implementation.
- **Pytree state** — `AtariState` is a `chex.dataclass` so it works out of the
  box with `jax.tree_util`, `optax`, and `flax`.
- **gymnax-style env API** — `AtariEnv` exposes `reset(key)` / `step(state, action)`
  with external state, fully compatible with `jit`, `vmap`, and `lax.scan`.
- **`make()` / `make_vec()`** — Gymnasium-familiar factory functions with optional
  wrapper presets (including the standard DQN stack) and `jit_compile` support.
- **Composable wrappers** — five preprocessing wrappers, all `jit` and `vmap`
  compatible. See the [Wrappers](#wrappers) table below.

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

Game ROMs are loaded via [ale-py](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
(installed automatically).  ROMs must be accepted under the ALE ROM licence:

```python
import ale_py
ale_py.ALEInterface().loadROM(ale_py.roms.Breakout)  # accept licence on first run
```

## Quick Start

### `make()`

```python
import jax
from atarax.env import EnvSpec, make

key = jax.random.PRNGKey(0)

# Raw environment — use EnvSpec or the "atari/<name>-v0" string
env = make(EnvSpec("atari", "breakout"))
env = make("atari/breakout-v0")               # equivalent string form
obs, state = env.reset(key)                   # obs: uint8[210, 160, 3]
obs, state, reward, done, info = env.step(state, env.sample(key))

# Full DQN preprocessing stack (JIT-compiled + local XLA cache by default)
env = make("atari/breakout-v0", preset=True)
obs, state = env.reset(key)                   # obs: uint8[84, 84, 4]

# Custom wrapper list (applied innermost → outermost)
from atarax.env import GrayscaleWrapper, ResizeWrapper
env = make("atari/breakout-v0", wrappers=[GrayscaleWrapper, ResizeWrapper])

# Spinner shown on first (compilation) call of each method
env = make("atari/breakout-v0", preset=True, show_compile_progress=True)
obs, state = env.reset(key)                   # ⠹ Compiling reset... → ✓ Compiling reset...
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

# JIT-compiled with spinner feedback
vec_env = make_vec("atari/breakout-v0", n_envs=32, preset=True, show_compile_progress=True)
```

### Rendering and Interactive Play

```python
import jax
from atarax.utils.render import play, render
from atarax.env import make

key = jax.random.PRNGKey(0)

# Render a single frame in a pygame window
env = make("atari/breakout-v0")
obs, state = env.reset(key)
render(state)                          # 320×420 window (scale=2 default)
render(state, scale=4, caption="Breakout")

# Play a game interactively (keyboard control, native 210×160 RGB)
play("atari/breakout-v0")             # scale=3 default → 480×630 window
play("atari/breakout-v0", scale=2, fps=30)
```

Keyboard controls for `play()`:

| Key | Action |
| --- | --- |
| Arrow keys / `W A S D` | Movement |
| `Space` | Fire |
| `Esc` / close window | Quit |

## Wrappers

Five composable RL preprocessing wrappers, each accepting an `AtariEnv` or
another wrapper and exposing the same `reset(key)` / `step(state, action)`
interface.

| Wrapper | Input | Output | Description | Extra state |
| --- | --- | --- | --- | --- |
| `GrayscaleWrapper` | `uint8[210, 160, 3]` | `uint8[210, 160]` | NTSC luminance conversion | — |
| `ResizeWrapper(out_h, out_w)` | `uint8[H, W]` | `uint8[out_h, out_w]` | Bilinear resize (default 84×84) | — |
| `FrameStackWrapper(n_stack)` | `uint8[H, W]` | `uint8[H, W, n_stack]` | Rolling frame buffer (default 4) | `FrameStackState` |
| `ClipRewardWrapper` | any reward | `float32 ∈ {−1, 0, +1}` | Sign clipping | — |
| `EpisodicLifeWrapper` | any env | same obs | Terminal on every life loss | `EpisodicLifeState` |

Stateless wrappers pass the inner state through unchanged. Stateful wrappers
return a `chex.dataclass` pytree that carries extra data alongside the inner
state — both are fully compatible with `jit`, `vmap`, and `lax.scan`.

### Standard DQN preprocessing stack

The standard Mnih et al. (2015) observation pipeline:

```python
import jax
from atarax.env import make

env = make("atari/breakout-v0", preset=True, jit_compile=True)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)           # obs: uint8[84, 84, 4]
obs, state, reward, done, info = env.step(state, env.sample(key))
# done              — True on life loss or game over
# reward            — clipped to {-1, 0, +1}
# info["real_done"] — True only on true game over
```

## Supported Games

All 57 environments from the standard RL benchmark (Mnih et al. 2015, *Human-level
control through deep reinforcement learning*, Nature):

| ID | `ale_name` | ID | `ale_name` | ID | `ale_name` |
|----|------------|----|------------|----|------------|
| 0  | alien | 20 | fishing_derby | 40 | riverraid |
| 1  | amidar | 21 | freeway | 41 | road_runner |
| 2  | assault | 22 | frostbite | 42 | robotank |
| 3  | asterix | 23 | gopher | 43 | seaquest |
| 4  | asteroids | 24 | gravitar | 44 | skiing |
| 5  | atlantis | 25 | hero | 45 | solaris |
| 6  | bank_heist | 26 | ice_hockey | 46 | space_invaders |
| 7  | battle_zone | 27 | jamesbond | 47 | star_gunner |
| 8  | beam_rider | 28 | kangaroo | 48 | tennis |
| 9  | berzerk | 29 | krull | 49 | time_pilot |
| 10 | bowling | 30 | kung_fu_master | 50 | tutankham |
| 11 | boxing | 31 | montezuma_revenge | 51 | up_n_down |
| 12 | breakout | 32 | ms_pacman | 52 | venture |
| 13 | centipede | 33 | name_this_game | 53 | video_pinball |
| 14 | chopper_command | 34 | phoenix | 54 | wizard_of_wor |
| 15 | crazy_climber | 35 | pitfall | 55 | yars_revenge |
| 16 | defender | 36 | pong | 56 | zaxxon |
| 17 | demon_attack | 37 | pooyan | | |
| 18 | double_dunk | 38 | private_eye | | |
| 19 | enduro | 39 | qbert | | |

Look up a game ID at runtime:

```python
from atarax.games.registry import GAME_IDS
game_id = GAME_IDS["seaquest"]  # → 43
```

## Architecture Notes

`AtariState` is a flat `chex.dataclass` pytree — every field is a fixed-shape
JAX array, making it compatible with `jax.vmap`, `jax.lax.scan`, and any
JAX-native optimiser.

The emulation loop follows the 6502 timing model: each CPU cycle advances the
TIA by 3 colour clocks.  One ALE frame = 262 scanlines × 76 CPU cycles = 19,912
colour clocks.

All branching inside the emulator uses JAX primitives so the computation graph
is fixed at trace time:

| Situation | Pattern used |
|-----------|-------------|
| Dispatch on opcode / game ID | `jax.lax.switch` |
| Conditional state update | `jax.lax.cond` / `jnp.where` |
| Fixed-count loop | `jax.lax.fori_loop` |

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
