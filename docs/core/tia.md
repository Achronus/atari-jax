# tia.py — TIA (Television Interface Adaptor)

**File:** `atari_jax/core/tia.py`
**ALE reference:** `TIA.cxx`

## Overview

The TIA is the graphics and audio chip.  It generates the video signal
directly in sync with the CPU — there is no frame buffer in the original
hardware.  The CPU "races the beam", writing to TIA registers to control
what appears on each scanline as the electron gun sweeps across the screen.

## Phase 1 (current) — Register Shadow Only

All 64 TIA registers are write-only in hardware.  In Phase 1:

- **Reads** return `0x00` (correct behaviour for most registers).
- **Writes** shadow the value into `state.tia.regs[addr & 0x3F]`.

This lets the CPU execute game code that writes to TIA without crashing, and
preserves the last-written value for Phase 2 to act on.

```python
def tia_write(state, addr13, value):
    reg_idx  = addr13 & 0x3F        # 6-bit register index
    new_regs = state.tia.regs.at[reg_idx].set(value)
    return state.replace(tia=state.tia.replace(regs=new_regs))
```

## Phase 2 — Scanline Rasteriser (TODO)

The rasteriser will be implemented in Phase 2 to get Breakout working
end-to-end.  Key registers involved:

### Background & Playfield

| Register | Addr | Description |
|----------|------|-------------|
| COLUBK | 0x09 | Background colour |
| COLUPF | 0x08 | Playfield / ball colour |
| PF0 | 0x0D | Playfield byte 0 (4 bits, reflected) |
| PF1 | 0x0E | Playfield byte 1 (8 bits) |
| PF2 | 0x0F | Playfield byte 2 (8 bits) |
| CTRLPF | 0x0A | Playfield control (reflect, score, priority) |

### Players

| Register | Addr | Description |
|----------|------|-------------|
| COLUP0 | 0x06 | Player 0 colour |
| COLUP1 | 0x07 | Player 1 colour |
| GRP0 | 0x1B | Player 0 graphics byte |
| GRP1 | 0x1C | Player 1 graphics byte |
| RESP0 | 0x10 | Reset player 0 horizontal position |
| RESP1 | 0x11 | Reset player 1 horizontal position |
| HMP0 | 0x20 | Player 0 horizontal motion |
| HMP1 | 0x21 | Player 1 horizontal motion |
| NUSIZ0 | 0x04 | Player 0 / missile 0 size & copies |
| NUSIZ1 | 0x05 | Player 1 / missile 1 size & copies |

### Missiles & Ball

| Register | Addr | Description |
|----------|------|-------------|
| RESM0 | 0x12 | Reset missile 0 position |
| RESM1 | 0x13 | Reset missile 1 position |
| RESBL | 0x14 | Reset ball position |
| ENAM0 | 0x1D | Enable missile 0 |
| ENAM1 | 0x1E | Enable missile 1 |
| ENABL | 0x1F | Enable ball |

### Timing

| Register | Addr | Description |
|----------|------|-------------|
| WSYNC | 0x02 | Wait for sync — CPU halts until end of scanline |
| VSYNC | 0x00 | Vertical sync control |
| VBLANK | 0x01 | Vertical blank / input latch control |
| HMOVE | 0x2A | Apply horizontal motion registers |

### Collision Latches (reads)

| Register | Addr | Bits 7–6 |
|----------|------|----------|
| CXM0P | 0x00 | M0-P1, M0-P0 |
| CXM1P | 0x01 | M1-P0, M1-P1 |
| CXP0FB | 0x02 | P0-PF, P0-BL |
| CXP1FB | 0x03 | P1-PF, P1-BL |
| CXM0FB | 0x04 | M0-PF, M0-BL |
| CXM1FB | 0x05 | M1-PF, M1-BL |
| CXBLPF | 0x06 | BL-PF, — |
| CXPPMM | 0x07 | P0-P1, M0-M1 |
| CXCLR  | 0x2C | Clear all collision latches |

## Scanline Timing

Each NTSC scanline is 228 colour clocks wide; the CPU runs at 1/3 the
colour clock rate so one CPU cycle = 3 colour clocks.

```text
Scanline = 68 colour clocks blank + 160 colour clocks visible
         = 22.67 CPU cycles blank + 53.33 CPU cycles visible
         = 76 CPU cycles per scanline (rounded, as ALE counts it)
```

Full NTSC frame:

- 3 lines VSYNC
- 37 lines VBLANK
- 192 lines visible (actual game content)
- 30 lines overscan
- Total: 262 lines × 76 CPU cycles = 19,912 cycles per frame
