# riot.py — M6532 RIOT Chip

**File:** `atarax/core/riot.py`
**ALE reference:** `M6532.cxx`

## Overview

The MOS 6532 "RIOT" (RAM I/O Timer) chip provides:

1. **128 bytes of working RAM** — the only general-purpose RAM in the Atari 2600
2. **Programmable interval timer** — used for timing game loops and audio
3. **Two I/O ports** — Port A (joystick) and Port B (console switches)

## Address Decoding

Decoded by the system bus (`bus.py`) before reaching this module.

- **RAM:** `(addr & 0x1080) == 0x0080` AND `(addr & 0x0200) == 0`
  - RAM index: `addr & 0x7F` (7-bit, 0–127)
- **I/O:** `(addr & 0x1080) == 0x0080` AND `(addr & 0x0200) != 0`
  - Register: `addr & 0x07` (8 registers)

## RAM (`riot_ram_read` / `riot_ram_write`)

Straightforward array read/write.  The 7-bit RAM index is `addr13 & 0x7F`.

```python
ram_idx = addr13 & 0x7F
value   = state.riot.ram[ram_idx]          # read
new_ram = state.riot.ram.at[ram_idx].set(v) # write
```

The stack lives in the upper half of RAM (indices 0x40–0x7F = addresses
0x180–0x1FF) when the 6502 SP is in the normal range.

## Timer

### Register Map (writes)

| addr & 0x17 | Hex  | Interval  | `interval_shift` |
|-------------|------|-----------|-----------------|
| 0x14        | TIM1T | Divide by 1 | 0 |
| 0x15        | TIM8T | Divide by 8 | 3 |
| 0x16        | TIM64T | Divide by 64 | 6 (ALE default at reset) |
| 0x17        | T1024T | Divide by 1024 | 10 |

Write condition: `(addr & 0x14) == 0x14` (bits 4 and 2 both set).
Mode selected by `addr & 0x03`.

On a timer write, three fields are saved to state:

- `timer` ← value written (uint8)
- `interval_shift` ← mode-dependent shift (0/3/6/10)
- `cycles_when_set` ← current `state.cycles` (int32 snapshot)

### Timer read (computed on demand)

Rather than maintaining a decrementing counter, the live value is derived
from elapsed cycles whenever it is read.  This avoids adding a rapidly-
changing field to every emulation tick.

```python
delta     = state.cycles - state.cycles_when_set
normal    = timer - (delta >> interval_shift) - 1

if normal >= 0:  return normal        # still counting down
else:            return overflow_point - delta - 1
                 # where overflow_point = timer << interval_shift
```

The `- 1` offset matches ALE's `M6532::peek` case 0x04.

### IRQ flag

`0x80` if the timer has underflowed (normal < 0), otherwise `0x00`.
Returned when reading registers 0x05 or 0x07.

### I/O Register Map (reads)

| addr & 0x07 | Name | Returns |
|-------------|------|---------|
| 0x00 | SWCHA | `port_a` — joystick (active-low, 0xFF = idle) |
| 0x01 | SWACNT | `0x00` — DDRA stub |
| 0x02 | SWCHB | `port_b` — console switches (0xFF = defaults) |
| 0x03 | SWBCNT | `0x00` — DDRB stub |
| 0x04 | INTIM | Live timer value |
| 0x05 | INSTAT | IRQ flag (0x80 or 0x00) |
| 0x06 | — | Timer mirror |
| 0x07 | — | IRQ flag mirror |

### Port A (joystick) bit layout

```text
Bit  7  6  5  4  3  2  1  0
     P2 P2 P2 P2 P1 P1 P1 P1
         ↑right ↑left ↑down ↑up (active-low)
```

### Port B (console switches) bit layout

```text
Bit  7  6  5  4  3  2  1  0
     —  —  —  —  —  —  SELECT RESET
     (active-low)
```

## Current Stubs

Port A and Port B **writes** (DDR configuration) are no-ops.  Games almost
never change the data-direction registers from their default (input = 0xFF).
