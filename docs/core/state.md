# state.py — AtariState Pytree

**File:** `atari_jax/core/state.py`
**ALE reference:** `System.cxx`, `M6502.cxx`, `M6532.cxx`, `TIA.cxx`

## Purpose

Defines the complete Atari 2600 machine state as a JAX pytree.  A pytree is
a nested Python container (dict / dataclass / namedtuple) whose leaves are
JAX arrays.  Registering the state as a pytree means:

- `jax.vmap(f)(batched_state)` runs `f` over a batch of states in parallel
- `jax.jit(f)(state)` compiles `f` once for the given state shape/dtype
- `jax.lax.scan(f, state, xs)` unrolls `f` over a sequence without retracing

**Rule:** every field must be a fixed-shape `jnp` array — no Python scalars,
no lists, no dynamic shapes.

## Classes

### `CPUState`

MOS 6502 / 6507 registers.

| Field  | dtype   | Shape | Description |
|--------|---------|-------|-------------|
| `a`    | uint8   | ()    | Accumulator |
| `x`    | uint8   | ()    | X index register |
| `y`    | uint8   | ()    | Y index register |
| `sp`   | uint8   | ()    | Stack pointer — page-1 offset, grows downward |
| `pc`   | uint16  | ()    | Program counter |
| `flags`| uint8   | ()    | Processor status: N V 1 B D I Z C (bits 7..0) |

**Flags bit positions:**

```text
Bit  7  6  5  4  3  2  1  0
     N  V  1  B  D  I  Z  C
```

- `N` Negative — set when result bit-7 = 1
- `V` oVerflow — set on signed arithmetic overflow
- `1` always set (bit 5 is hardwired to 1 on real 6502)
- `B` Break — set when BRK pushes flags to stack
- `D` Decimal — BCD mode (Atari ignores it in practice but we track it)
- `I` Interrupt disable
- `Z` Zero — set when result = 0
- `C` Carry

Initial value: `0x24` (`I=1`, bit-5=1, all others 0) — matches ALE reset.

### `RIOTState`

MOS 6532 RAM I/O Timer chip.

| Field            | dtype  | Shape  | Description |
|------------------|--------|--------|-------------|
| `ram`            | uint8  | (128,) | General-purpose working RAM |
| `timer`          | uint8  | ()     | Value written at last `TIM*T` write |
| `interval_shift` | uint8  | ()     | 0 / 3 / 6 / 10 → divide by 1 / 8 / 64 / 1024 |
| `cycles_when_set`| int32  | ()     | `state.cycles` at the moment of the last timer write |
| `port_a`         | uint8  | ()     | Joystick byte (active-low: 0xFF = no buttons) |
| `port_b`         | uint8  | ()     | Console switches (0xFF = defaults) |

**Timer design note:** Rather than decrementing `timer` on every CPU cycle
(which would force us to store an intermediate value between steps), the
live timer value is *computed on demand* from elapsed cycles:

```python
delta     = cycles - cycles_when_set
timer_val = timer - (delta >> interval_shift) - 1   # while >= 0
# after underflow: counts down 1/cycle from the overflow point
```

This matches `M6532::peek` in ALE and is JAX-friendly because it avoids
storing rapidly-changing intermediate state.

### `TIAState`

Television Interface Adaptor.

| Field        | dtype  | Shape    | Description |
|--------------|--------|----------|-------------|
| `regs`       | uint8  | (64,)    | Shadow copy of the 64 TIA write-only registers |
| `collisions` | uint16 | ()       | 15 collision-latch bits (Phase 2) |
| `scanline`   | uint8  | (160,)   | Pixel colour indices for the current scanline |

`regs[i]` mirrors the last value written to TIA address `i & 0x3F`.
Most TIA registers are write-only; reads return 0 (or a collision byte from
specific addresses in Phase 2).

### `AtariState`

Top-level pytree.  Every emulation function takes and returns this.

| Field          | dtype   | Shape         | Description |
|----------------|---------|---------------|-------------|
| `cpu`          | —       | CPUState      | CPU registers |
| `riot`         | —       | RIOTState     | RIOT chip |
| `tia`          | —       | TIAState      | TIA chip |
| `bank`         | uint8   | ()            | Active ROM bank index |
| `screen`       | uint8   | (210, 160, 3) | Rendered RGB frame |
| `frame`        | int32   | ()            | Total emulated frames since power-on |
| `episode_frame`| int32   | ()            | Frames in the current episode |
| `lives`        | int32   | ()            | Lives remaining (game-specific) |
| `terminal`     | bool    | ()            | True when episode ended this step |
| `reward`       | float32 | ()            | Reward earned in last step |
| `cycles`       | int32   | ()            | Cumulative CPU cycles (used by RIOT timer) |

## Factory Functions

```python
new_cpu_state()   -> CPUState     # zeroed registers; SP=0xFF, flags=0x24
new_riot_state()  -> RIOTState    # zeroed RAM; interval_shift=6 (ALE default)
new_tia_state()   -> TIAState     # zeroed regs / scanline
new_atari_state() -> AtariState   # structural blank for vmap
```

`new_atari_state()` is a structural template only — it gives every field a
concrete dtype and shape so JAX can trace through the functions without
needing a real ROM.  Actual gameplay state is produced by loading ROM bytes
and calling `cpu_reset()` followed by `emulate_frame()`.

## Updating State

`@chex.dataclass` generates a `.replace(**kwargs)` method that returns a new
instance with the specified fields changed, leaving all others identical.

```python
# Update the accumulator and PC:
new_state = state.replace(cpu=state.cpu.replace(a=jnp.uint8(42),
                                                 pc=jnp.uint16(0x1234)))
```

This is the only way to "mutate" state — the original is never modified.
