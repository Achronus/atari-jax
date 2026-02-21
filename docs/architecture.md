# atari-jax — Architecture Overview

A pure-JAX reimplementation of all 57 Atari 2600 ALE environments as
vmappable pytrees.  Every environment step is a stateless function
`(state, action) → state` that compiles with `jax.jit` and batches with
`jax.vmap` — no Python loops, no host-side control flow.

## Repository Layout

```text
atari_jax/
  core/                 Hardware emulation layer
    state.py            AtariState pytree definition
    bus.py              13-bit address decoder + chip dispatch
    cpu/                MOS 6507 CPU (split by opcode category)
    riot.py             M6532 RIOT — RAM, timer, I/O ports
    tia.py              TIA rasteriser (NTSC, full scanline)
    cart.py             ROM cartridge + bankswitching schemes
    frame.py            emulate_scanline + emulate_frame
  games/
    base.py             AtariGame ABC (reset/step concrete; rewards abstract)
    registry.py         57-game registry + jax.lax.switch dispatch tables
    __init__.py         get_reward / is_terminal public API
    roms/               One module per game (57 files)
  env/
    atari_env.py        AtariEnv + EnvParams
    make.py             make() / make_vec() factory functions
    spaces.py           Box + Discrete observation/action spaces
    vec_env.py          make_rollout_fn — lax.scan-based rollout helper
    wrappers/           Composable RL preprocessing wrappers
  utils/
    rom_loader.py       load_rom(game_id) — lazy ale-py import
    preprocess.py       preprocess(frame) — NTSC grayscale + 84×84
    render.py           render(state) — lazy pygame import

tests/
  conftest.py           Session fixtures (FakeEnv, XLA cache config)
  test_core_smoke.py    CPU / bus / RIOT / cart unit tests
  test_tia.py           TIA unit tests
  test_parity.py        JIT / vmap / ROM smoke tests
  game/                 Per-game reward + terminal tests (57 games)
  env/                  AtariEnv, make, vec_env tests (ROM-backed)
  wrappers/             Per-wrapper unit tests + combined DQN stack

docs/
  architecture.md       This file — design overview
  core/                 Hardware layer reference (cpu, tia, riot, bus, cart, state)
  environments/         Per-game user reference (57 files + index)
```

## Data Flow

```text
action (int32)
    │
    ▼
bus_write (TIA WSYNC / joystick port)
    │
    ▼
cpu_step ──► bus_read / bus_write  ──► cart / riot / tia
    │              │
    │         riot timer ticks (per CPU cycle)
    │
    ▼
tia_rasterise ──► scanline pixels (uint8[160, 3])
    │
    ▼
emulate_frame ──► AtariState (screen, ram, reward, terminal, lives, …)
    │
    ▼
AtariGame.get_reward / is_terminal ──► float32 reward, bool terminal
```

## Environment API

### `AtariEnv`

The primary user-facing environment.  Wraps the emulation stack with a
gymnax-style interface: `reset(key) → (obs, state)` and
`step(state, action) → (obs, state, reward, done, info)`.

```python
from atari_jax.env import AtariEnv, EnvParams

env = AtariEnv("breakout")
env = AtariEnv("breakout", params=EnvParams(frame_skip=4, noop_max=30))
```

`AtariEnv` is always a Python-constant object — never JAX-traced.  The
`state` it returns is an `AtariState` pytree that is the only value that
passes through `jit` and `vmap`.

### `make()` / `make_vec()`

Convenience factory functions with an optional wrapper preset:

```python
from atari_jax.env import make, make_vec

env = make("breakout")                          # raw AtariEnv
env = make("breakout", preset="dqn")            # full DQN stack
env, rollout = make_vec("breakout", 32, preset="dqn")
```

`make(wrappers=[...])` accepts a list of wrapper classes applied innermost-first.
`preset="dqn"` applies: Grayscale → Resize(84×84) → FrameStack(4) → ClipReward
→ EpisodicLife.

### Wrappers

All five wrappers implement the same `reset` / `step` interface as `AtariEnv`.

| Wrapper | Transformation | State type |
| --- | --- | --- |
| `GrayscaleWrapper` | `uint8[210,160,3]` → `uint8[210,160]` | pass-through |
| `ResizeWrapper` | `uint8[H,W]` → `uint8[84,84]` | pass-through |
| `FrameStackWrapper` | `uint8[H,W]` → `uint8[H,W,N]` | `FrameStackState` |
| `ClipRewardWrapper` | reward → `sign(reward)` | pass-through |
| `EpisodicLifeWrapper` | terminal on life loss | `EpisodicLifeState` |

Stateless wrappers pass the inner state through unchanged.  Stateful wrappers
carry extra fields in a `@chex.dataclass` pytree nested alongside the inner
state, so arbitrary stacking works naturally with `jit` and `vmap`.

### Compiled rollout

`make_rollout_fn(env)` returns a function backed by `jax.lax.scan`:

```python
rollout = jax.jit(make_rollout_fn(env))
final_state, (obs, reward, done, info) = rollout(state, actions)

# Parallel agents via vmap
batched = jax.jit(jax.vmap(make_rollout_fn(env)))
```

## JAX Dispatch Patterns

| Situation | Pattern |
| --- | --- |
| Dispatch on opcode / game ID | `jax.lax.switch(idx, branches, *operands)` |
| Conditional state update | `jax.lax.cond(pred, true_fn, false_fn, operand)` |
| Element-wise conditional | `jnp.where(mask, a, b)` |
| Fixed-count loop | `jax.lax.fori_loop(0, N, body_fn, init)` |
| Scan over a sequence | `jax.lax.scan(fn, init, xs)` |

The critical invariant: **no Python branching on traced values**.  Any JAX
array produced inside a `jit` or `vmap` context is a traced value and cannot
be used as a Python `if` condition.  All branching must go through the
primitives above.

## Key Invariants for Contributors

- **All `AtariState` fields are fixed-shape JAX arrays** — no Python scalars.
  Use `jnp.uint8(0xFF)`, not `0xFF`.
- **State is immutable** — use `state.__replace__(field=value)` (chex
  dataclass method), never mutate fields in place.
- **Dtypes must be explicit** — all branches of `jax.lax.switch` / `cond`
  must return identical shapes and dtypes.
- **Absolute imports only** — `from atari_jax.core.xxx import ...`
  (no relative dot imports).

## Testing

```bash
# Fast unit tests (no ROM required)
uv run pytest tests/ -v --ignore=tests/env/test_atari_env.py --ignore=tests/env/test_make.py --ignore=tests/test_parity.py

# Full suite (requires ale-py ROM acceptance)
uv run pytest tests/ -v

# Specific subsystems
uv run pytest tests/test_core_smoke.py tests/test_tia.py -v   # emulator
uv run pytest tests/game/ -v                                   # all 57 games
uv run pytest tests/wrappers/ -v                               # wrappers
uv run pytest tests/env/ -v                                    # env API
```
