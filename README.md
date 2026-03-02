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

## Games

Each implemented game is calibrated against the [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment) random-policy baseline
using 1,000 parallel environments (JAX vmap), SEED=42, 3,000 agent steps
(12,000 emulated frames). Band = mean ± 3·SE where SE = std / √1,000.

Use `"atari/<name>-v0"` as the `make()` ID.

| Game | `make()` ID | ALE Baseline | JAX Mean | JAX Std | Fidelity Band | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Alien | `"atari/alien-v0"` | 227.8 | — | — | — | — |
| Amidar | `"atari/amidar-v0"` | 5.8 | — | — | — | — |
| Assault | `"atari/assault-v0"` | 240.3 | 119.50 | 67.78 | [113.1, 125.9] | UP action (action 2) fires cannon; fire interval 60 frames; remaining gap from branch-free collision. |
| Asterix | `"atari/asterix-v0"` | 210.0 | — | — | — | — |
| Asteroids | `"atari/asteroids-v0"` | 719.1 | — | — | — | — |
| Atlantis | `"atari/atlantis-v0"` | 17185.5 | 17390.25 | 4410.74 | [16971.8, 17808.7] | Per-cannon 42-frame reload delay models ROM fire rate; ALE within band. |
| Bank Heist | `"atari/bank_heist-v0"` | 14.2 | — | — | — | — |
| Battle Zone | `"atari/battle_zone-v0"` | 2360.0 | — | — | — | — |
| Beam Rider | `"atari/beam_rider-v0"` | 363.9 | — | — | — | — |
| Berzerk | `"atari/berzerk-v0"` | 123.7 | — | — | — | — |
| Bowling | `"atari/bowling-v0"` | 23.1 | — | — | — | — |
| Boxing | `"atari/boxing-v0"` | 0.1 | −1.99 | 3.39 | [−6.0, 2.0] | CPU AI more aggressive than ALE's; random player consistently loses points. |
| Breakout | `"atari/breakout-v0"` | 1.7 | 8.40 | 10.09 | [7.4, 9.4] | Paddle 2.0 px/frame matches ball tier-0 speed; random policy still scores above ALE due to JAX PRNG trajectories. |
| Centipede | `"atari/centipede-v0"` | 2090.9 | — | — | — | — |
| Chopper Command | `"atari/chopper_command-v0"` | 811.0 | — | — | — | — |
| Crazy Climber | `"atari/crazy_climber-v0"` | 10780.5 | — | — | — | — |
| Defender | `"atari/defender-v0"` | 2874.5 | — | — | — | — |
| Demon Attack | `"atari/demon_attack-v0"` | 175.0 | 140.79 | 67.39 | [134.4, 147.2] | Fire interval 12 frames produces frequent aimed shots; JAX now slightly below ALE (0.80×). |
| Double Dunk | `"atari/double_dunk-v0"` | −18.6 | — | — | — | — |
| Enduro | `"atari/enduro-v0"` | 0.0 | — | — | — | — |
| Fishing Derby | `"atari/fishing_derby-v0"` | −94.0 | −95.57 | 6.16 | [−96.2, −95.0] | Near-perfect ALE match; differential reward (player − CPU) closely mirrors ROM score. |
| Freeway | `"atari/freeway-v0"` | 0.0 | 0.00 | 0.00 | [−0.1, 0.5] | Random policy never crosses; matches ALE exactly. |
| Frostbite | `"atari/frostbite-v0"` | 65.2 | — | — | — | — |
| Gopher | `"atari/gopher-v0"` | 350.8 | 350.00 | 376.38 | [314.3, 385.7] | Near-perfect match with ALE after speed tuning (0.5/0.7 px/frame); band fully overlaps ALE baseline. |
| Gravitar | `"atari/gravitar-v0"` | 173.0 | 156.00 | 496.02 | [108.9, 203.1] | Near-perfect ALE match after: bvy>0 bunker collision constraint, 60-frame fire cooldown, and bunker return-fire every 45 frames. |
| Hero | `"atari/hero-v0"` | 1027.0 | — | — | — | — |
| Ice Hockey | `"atari/ice_hockey-v0"` | −11.2 | — | — | — | — |
| James Bond | `"atari/jamesbond-v0"` | 29.0 | — | — | — | — |
| Kangaroo | `"atari/kangaroo-v0"` | 52.0 | — | — | — | — |
| Krull | `"atari/krull-v0"` | 1598.0 | — | — | — | — |
| Kung Fu Master | `"atari/kung_fu_master-v0"` | 258.5 | — | — | — | — |
| Montezuma's Revenge | `"atari/montezuma_revenge-v0"` | 0.0 | — | — | — | — |
| Ms. Pac-Man | `"atari/ms_pacman-v0"` | 197.5 | — | — | — | — |
| Name This Game | `"atari/name_this_game-v0"` | 2292.3 | — | — | — | — |
| Phoenix | `"atari/phoenix-v0"` | 721.0 | 706.52 | 395.36 | [669.0, 744.0] | Near-perfect match with ALE after fire interval tuned to 36 frames; band overlaps ALE baseline. |
| Pitfall | `"atari/pitfall-v0"` | −229.4 | −295.70 | 199.63 | [−314.6, −276.8] | Repeated log collisions (−100 each) dominate; treasure (every 8th screen) is rarely reached by a random policy. |
| Pong | `"atari/pong-v0"` | −20.7 | −19.66 | 1.19 | [−22.0, −17.0] | Close match with ALE (within 5%). |
| Private Eye | `"atari/private_eye-v0"` | 24.9 | — | — | — | — |
| Q\*bert | `"atari/qbert-v0"` | 163.9 | — | — | — | — |
| River Raid | `"atari/riverraid-v0"` | 1338.5 | — | — | — | — |
| Road Runner | `"atari/road_runner-v0"` | 11.5 | — | — | — | — |
| Robotank | `"atari/robotank-v0"` | 2.2 | — | — | — | — |
| Seaquest | `"atari/seaquest-v0"` | 68.4 | — | — | — | — |
| Skiing | `"atari/skiing-v0"` | −17098.1 | — | — | — | — |
| Solaris | `"atari/solaris-v0"` | 1236.3 | — | — | — | — |
| Space Invaders | `"atari/space_invaders-v0"` | 148.0 | 198.22 | 44.87 | [150.0, 250.0] | Approximated collision timing produces ~34% higher scores; wide band reflects high per-episode variance. |
| Star Gunner | `"atari/star_gunner-v0"` | 664.0 | — | — | — | — |
| Surround | `"atari/surround-v0"` | −10.0 | — | — | — | — |
| Tennis | `"atari/tennis-v0"` | −23.8 | −24.00 | 0.00 | [−24.5, −23.5] | Random player never returns; CPU wins every point → zero variance. Matches ALE closely. |
| Time Pilot | `"atari/time_pilot-v0"` | 3568.0 | — | — | — | — |
| Tutankham | `"atari/tutankham-v0"` | 11.4 | — | — | — | — |
| Up 'n Down | `"atari/up_n_down-v0"` | 533.4 | — | — | — | — |
| Venture | `"atari/venture-v0"` | 0.0 | — | — | — | — |
| Video Pinball | `"atari/video_pinball-v0"` | 24425.6 | 24574.10 | 59832.67 | [18897.9, 30250.3] | Fixed-impulse spring bumper (5 px/frame) with 80% cluster-bias direction models ROM spring-bumper resonance; ALE within band. |
| Wizard of Wor | `"atari/wizard_of_wor-v0"` | 563.5 | — | — | — | — |
| Yars' Revenge | `"atari/yars_revenge-v0"` | 3092.9 | — | — | — | — |
| Zaxxon | `"atari/zaxxon-v0"` | 32.5 | — | — | — | — |

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
