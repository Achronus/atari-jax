# Atari Jax

A pure-JAX reimplementation of all 57 standard Atari 2600 ALE environments as
fully vmappable pytrees.  Inspired by [Brax](https://github.com/google/brax): every
environment step is a stateless function `(state, action) → state` that compiles
with `jax.jit` and batches with `jax.vmap` — no Python loops, no host-side
control flow.

---

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

---

## Requirements

- Python 3.13+
- JAX 0.9+ (CPU, CUDA, or TPU backend)

---

## Installation

```bash
pip install atari-jax
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

---

## Quick Start

```python
import jax.numpy as jnp
from atari_jax.games.registry import GAME_IDS
from atari_jax.games.roms.breakout import Breakout
from atari_jax.utils.rom_loader import load_rom

# Load ROM bytes via ale-py
rom = load_rom(GAME_IDS["breakout"])  # uint8[4096]

# Initialise the game
game = Breakout()
state = game.reset(rom)

# Step with action 1 (FIRE)
state = game.step(state, rom, jnp.int32(1))

print(f"reward={float(state.reward):.1f}  lives={int(state.lives)}  terminal={bool(state.terminal)}")
```

### JIT compilation

```python
import jax
import jax.numpy as jnp
from atari_jax.games.roms.breakout import Breakout
from atari_jax.utils.rom_loader import load_rom
from atari_jax.games.registry import GAME_IDS

rom = load_rom(GAME_IDS["breakout"])
game = Breakout()

reset_jit = jax.jit(game.reset)
step_jit  = jax.jit(game.step)

state = reset_jit(rom)
state = step_jit(state, rom, jnp.int32(0))
```

### Batched rollout with vmap

```python
import jax
import jax.numpy as jnp
from atari_jax.games.roms.breakout import Breakout
from atari_jax.utils.rom_loader import load_rom
from atari_jax.games.registry import GAME_IDS

rom = load_rom(GAME_IDS["breakout"])
game = Breakout()

N = 64
roms = jnp.broadcast_to(rom, (N,) + rom.shape)

# Reset N environments in parallel
reset_batch = jax.jit(jax.vmap(game.reset))
states = reset_batch(roms)

# Step all N environments simultaneously
actions = jnp.zeros(N, dtype=jnp.int32)
step_batch = jax.jit(jax.vmap(game.step))
states = step_batch(states, roms, actions)
```

---

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
from atari_jax.games.registry import GAME_IDS
game_id = GAME_IDS["seaquest"]  # → 43
```

---

## Project Structure

```text
atari_jax/
  core/
    state.py      AtariState pytree (CPU, RIOT, TIA, cartridge + episode fields)
    bus.py        13-bit address decoder — routes reads/writes to TIA/RIOT/ROM
    cpu/          MOS 6507 CPU — 256-entry opcode table, all 56 legal opcodes
    riot.py       M6532 — 128-byte RAM, interval timer, joystick I/O
    tia.py        Full NTSC rasteriser — sprites, playfield, collisions, HMOVE
    cart.py       ROM cartridge + F8 bankswitching
    frame.py      emulate_frame — runs one ALE frame (262 scanlines)
  games/
    base.py       AtariGame ABC
    registry.py   GameSpec, GAME_IDS, REWARD_FNS, TERMINAL_FNS
    roms/         57 game modules — reward/terminal/lives + reset/step
  utils/
    rom_loader.py load_rom(game_id) — lazy ale-py import
tests/
  test_core_smoke.py   CPU, bus, RIOT, cartridge unit tests
  test_tia.py          TIA rasteriser unit tests
  test_parity.py       JIT/vmap smoke tests
  game/
    test_breakout.py   Breakout reward/terminal logic tests
    test_registry.py   Registry dispatch tests
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests run in parallel by default via `pytest-xdist`.

---

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

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
