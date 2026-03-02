# Atarax

A pure-JAX library of all 57 Atari 2600 game environments. Each game's
mutable state lives in an `AtariState` pytree, making every
`step(rng, state, action, params) → (obs, state, reward, done, info)` call a
stateless JAX computation that compiles with `jax.jit` and batches with
`jax.vmap` — no Python loops, no host-side control flow, no ROM loading.

Part of the [Envrax suite](https://github.com/Achronus/envrax) — installs
`envrax` automatically.

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
- **`make()` / `make_vec()`** — factory functions returning `(env, params)`
  and `(VmapEnv, params)` tuples; optional wrapper presets (including the
  standard DQN stack) and a persistent XLA compilation cache built in.
- **`make_multi()` / `make_multi_vec()`** — convenience variants for creating
  one environment per game from a list of IDs; useful for multi-game training
  loops.
- **Composable wrappers** — three Atari-native wrappers (`AtariPreprocessing`,
  `EpisodicLife`, `JitWrapper`) plus additional generic wrappers from `envrax`.
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
import jax.numpy as jnp
from atarax import make, AtaraxParams

key = jax.random.PRNGKey(0)
params = AtaraxParams()

# Raw environment — use the "atari/<name>-v0" string
env, params = make("atari/breakout-v0")
obs, state = env.reset(key, params)           # obs: uint8[210, 160, 3]
obs, state, reward, done, info = env.step(key, state, jnp.int32(0), params)

# Full DQN preprocessing stack (JIT-compiled with XLA cache by default)
env, params = make("atari/breakout-v0", preset=True)
obs, state = env.reset(key, params)           # obs: uint8[84, 84, 4]

# AtariPreprocessing can also be used standalone
from atarax.wrappers import AtariPreprocessing
env, _ = make("atari/breakout-v0", jit_compile=False)
env = AtariPreprocessing(env)                 # 84×84, 4-frame stack
env = AtariPreprocessing(env, h=42, w=42, n_stack=2)

# Custom wrapper list (applied innermost → outermost)
from atarax.wrappers import GrayscaleObservation, ResizeObservation
env, params = make("atari/breakout-v0", wrappers=[GrayscaleObservation, ResizeObservation])

# Disable the XLA cache for ephemeral runs
env, params = make("atari/breakout-v0", preset=True, cache_dir=None)

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
# obs_seq: uint8[128, 84, 84, 4]
```

### `make_vec()`

```python
import jax
import jax.numpy as jnp
from atarax import make_vec

key = jax.random.PRNGKey(0)

# reset() splits the key 32 ways — each env gets a distinct random start
vec_env, params = make_vec("atari/breakout-v0", n_envs=32, preset=True)
obs, states = vec_env.reset(key, params)      # obs: uint8[32, 84, 84, 4]

# step() operates across all 32 envs simultaneously
keys = jax.random.split(key, 32)
actions = jnp.zeros(32, dtype=jnp.int32)
obs, states, reward, done, info = vec_env.step(key, states, actions, params)

# Multi-step rollout via lax.scan + vmap
def vec_step(carry, _):
    key, states = carry
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, 32)
    obs, states, reward, done, info = vec_env.step(subkey, states, actions, params)
    return (key, states), (obs, reward, done)

obs, states = vec_env.reset(key, params)
(_, final_states), (obs_seq, rewards, dones) = jax.lax.scan(
    vec_step, (key, states), None, length=128
)
# obs_seq: uint8[128, 32, 84, 84, 4]
```

### `make_multi()` / `make_multi_vec()`

Create one `(env, params)` or `(VmapEnv, params)` tuple per game from a list
of IDs. Intended for multi-game training loops where each game needs its own
independent environment.

```python
import jax
from atarax import make_multi, make_multi_vec

key = jax.random.PRNGKey(0)

# Returns List[Tuple[AtaraxGame, AtaraxParams]] — one per game
results = make_multi(["atari/breakout-v0", "atari/pong-v0"])
env, params = results[0]
obs, state = env.reset(key, params)          # obs: uint8[210, 160, 3]

# Vectorized variant — Returns List[Tuple[VmapEnv, AtaraxParams]]
vec_results = make_multi_vec(["atari/breakout-v0"], n_envs=16)
vec_env, params = vec_results[0]
obs, states = vec_env.reset(key, params)     # obs: uint8[16, 210, 160, 3]

# List all registered games
from atarax.games import GAMES
print(sorted(GAMES))
```

### Rendering and Interactive Play

```python
import jax
import jax.numpy as jnp
from atarax import make

key = jax.random.PRNGKey(0)
env, params = make("atari/breakout-v0", jit_compile=False)

# Render a single frame (returns uint8[210, 160, 3])
obs, state = env.reset(key, params)
frame = env.render(state)

# Play a game interactively in a pygame window
env.play()                                   # Esc or close window to quit
env.play(scale=4, fps=30)
```

`Esc` or closing the window always quits. Per-game keyboard controls are
defined in each game's `_key_map()` method.

## Wrappers

Composable wrappers, each accepting any `AtaraxGame` (or wrapped env) and
exposing the same `reset(rng, params)` / `step(rng, state, action, params)`
interface.

| Wrapper | Input | Output | Description | Extra state |
| --- | --- | --- | --- | --- |
| `AtariPreprocessing` | `uint8[210, 160, 3]` | `uint8[84, 84, 4]` | Full DQN stack (grayscale → resize → frame-stack → clip reward → record stats + episodic life) | `EpisodeStatisticsState` |
| `EpisodicLife` | any env | same obs | Terminal on every life loss | `EpisodicLifeState` |
| `JitWrapper` | any env | same obs | JIT-compiles `reset` + `step` with a warmup pass; applied automatically by `make()` when `jit_compile=True` | — |

Additional generic wrappers (grayscale, resize, normalize, frame-stack, clip
reward, episode statistics, and more) are available via the
[`envrax`](https://github.com/Achronus/envrax) package and re-exported through
`atarax.wrappers`.

`JitWrapper` can also be used standalone to eagerly compile any env:

```python
from atarax.wrappers import JitWrapper
from atarax import make

env, params = make("atari/breakout-v0", jit_compile=False)
env = JitWrapper(env)
```

### Standard DQN preprocessing stack

The standard Mnih et al. (2015) observation pipeline:

```python
import jax
import jax.numpy as jnp
from atarax import make

key = jax.random.PRNGKey(0)
env, params = make("atari/breakout-v0", preset=True)

obs, state = env.reset(key, params)          # obs: uint8[84, 84, 4]
obs, state, reward, done, info = env.step(key, state, jnp.int32(0), params)
# done                    → True on life loss or game over
# reward                  → clipped to {-1, 0, +1}
# info["episode"]["r"]    → episode return (non-zero at episode end)
# info["episode"]["l"]    → episode length (non-zero at episode end)
```

## ALE Fidelity

Each game is calibrated against the ALE random-policy baseline using 1,000
parallel environments (JAX vmap), SEED=42, 3,000 agent steps (12,000 emulated
frames). Band = mean ± 3·SE where SE = std / √1,000.

| Game | ALE Baseline | JAX Mean | JAX Std | Fidelity Band | Notes |
| --- | --- | --- | --- | --- | --- |
| Pong | −20.7 | −19.66 | 1.19 | [−22.0, −17.0] | Close match (< 5% gap) |
| Breakout | 1.7 | 8.52 | 7.30 | [3.0, 15.0] | JAX ~5× higher; branch-free collision + JAX PRNG differ |
| Space Invaders | 148.0 | 198.22 | 44.87 | [150.0, 250.0] | JAX ~34% higher; approximated simultaneous collision timing |
| Freeway | 0.0 | 0.00 | 0.00 | [−0.1, 0.5] | Exact match; random agent never crosses the road |
| Boxing | 0.1 | −1.99 | 3.39 | [−6.0, 2.0] | JAX CPU AI more aggressive than ALE's; net reward is negative |
| Tennis | −23.8 | −24.00 | 0.00 | [−24.5, −23.5] | Close match; random player never returns, CPU wins 6×4 points |

## Supported Games

Use `"atari/<name>-v0"` as the `make()` ID (underscores become hyphens):
`"atari/space_invaders-v0"`, `"atari/ms_pacman-v0"`, etc.

| Game | `make()` ID |
| --- | --- |
| Alien | `"atari/alien-v0"` |
| Amidar | `"atari/amidar-v0"` |
| Assault | `"atari/assault-v0"` |
| Asterix | `"atari/asterix-v0"` |
| Asteroids | `"atari/asteroids-v0"` |
| Atlantis | `"atari/atlantis-v0"` |
| Bank Heist | `"atari/bank_heist-v0"` |
| Battle Zone | `"atari/battle_zone-v0"` |
| Beam Rider | `"atari/beam_rider-v0"` |
| Berzerk | `"atari/berzerk-v0"` |
| Bowling | `"atari/bowling-v0"` |
| Boxing | `"atari/boxing-v0"` |
| Breakout | `"atari/breakout-v0"` |
| Centipede | `"atari/centipede-v0"` |
| Chopper Command | `"atari/chopper_command-v0"` |
| Crazy Climber | `"atari/crazy_climber-v0"` |
| Defender | `"atari/defender-v0"` |
| Demon Attack | `"atari/demon_attack-v0"` |
| Double Dunk | `"atari/double_dunk-v0"` |
| Enduro | `"atari/enduro-v0"` |
| Fishing Derby | `"atari/fishing_derby-v0"` |
| Freeway | `"atari/freeway-v0"` |
| Frostbite | `"atari/frostbite-v0"` |
| Gopher | `"atari/gopher-v0"` |
| Gravitar | `"atari/gravitar-v0"` |
| Hero | `"atari/hero-v0"` |
| Ice Hockey | `"atari/ice_hockey-v0"` |
| James Bond | `"atari/jamesbond-v0"` |
| Kangaroo | `"atari/kangaroo-v0"` |
| Krull | `"atari/krull-v0"` |
| Kung Fu Master | `"atari/kung_fu_master-v0"` |
| Montezuma's Revenge | `"atari/montezuma_revenge-v0"` |
| Ms. Pac-Man | `"atari/ms_pacman-v0"` |
| Name This Game | `"atari/name_this_game-v0"` |
| Phoenix | `"atari/phoenix-v0"` |
| Pitfall | `"atari/pitfall-v0"` |
| Pong | `"atari/pong-v0"` |
| Private Eye | `"atari/private_eye-v0"` |
| Q*bert | `"atari/qbert-v0"` |
| River Raid | `"atari/riverraid-v0"` |
| Road Runner | `"atari/road_runner-v0"` |
| Robotank | `"atari/robotank-v0"` |
| Seaquest | `"atari/seaquest-v0"` |
| Skiing | `"atari/skiing-v0"` |
| Solaris | `"atari/solaris-v0"` |
| Space Invaders | `"atari/space_invaders-v0"` |
| Star Gunner | `"atari/star_gunner-v0"` |
| Surround | `"atari/surround-v0"` |
| Tennis | `"atari/tennis-v0"` |
| Time Pilot | `"atari/time_pilot-v0"` |
| Tutankham | `"atari/tutankham-v0"` |
| Up 'n Down | `"atari/up_n_down-v0"` |
| Venture | `"atari/venture-v0"` |
| Video Pinball | `"atari/video_pinball-v0"` |
| Wizard of Wor | `"atari/wizard_of_wor-v0"` |
| Yars' Revenge | `"atari/yars_revenge-v0"` |
| Zaxxon | `"atari/zaxxon-v0"` |

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
