# cart.py — ROM Cartridge & Bankswitching

**File:** `atari_jax/core/cart.py`
**ALE reference:** `Cart*.cxx` (CartF8.cxx, CartF6.cxx, etc.)

## Overview

The Atari 2600 ROM address space is 4 KB (0x1000–0x1FFF).  Early games fit
in 4 KB; later games use **bankswitching** — swapping pages of a larger ROM
into that 4 KB window by accessing special "hotspot" addresses.

`cart.py` handles:

1. `cart_read` — index into the flat ROM array using the active bank
2. `cart_write` — detect hotspot accesses and update `state.bank`

## ROM Storage

The full ROM is stored as a flat `uint8[ROM_SIZE]` array, never modified
at runtime.  The active bank selects which 4 KB slice is currently visible:

```python
physical_offset = bank * 4096 + (addr13 & 0x0FFF)
byte = rom[physical_offset]
```

`rom` is passed as a separate argument (not part of `AtariState`) so XLA
can treat it as a compile-time constant.

## Bankswitching Schemes

### Scheme IDs (stored in ROMMetadata.scheme_id)

| ID | Name | ROM size | Banks | Hotspot addresses |
|----|------|----------|-------|-------------------|
| 0 | None | 2 KB / 4 KB | 1 | — |
| 1 | F8 | 8 KB | 2 | 0xFF8 (bank 0), 0xFF9 (bank 1) |
| 2 | F6 | 16 KB | 4 | 0xFF6–0xFF9 |
| 3 | F4 | 32 KB | 8 | 0xFF4–0xFFB |
| 4 | FE | 8 KB | 2 | JSR/RTS to 0x01FE/0x01FF |
| 5 | E0 | 8 KB | 8 × 1 KB | 0xFE0–0xFF9 |
| 6 | 3F | 8 KB | 4 × 2 KB | 0x003F writes select bank |
| 7 | 3E | 32 KB + 1 KB RAM | — | — |
| 8 | FA | 12 KB | 3 | 0xFF8–0xFFA |
| 9 | CV | 2 KB + 1 KB RAM | — | — |
| 10 | UA | 8 KB | 2 | 0x220/0x240 |
| 11 | AR | Starpath Supercharger | — | — |

### F8 (implemented)

Used by: Breakout, Space Invaders, many common games.

- 2 banks of 4 KB each (total 8 KB ROM)
- Hotspots at `addr & 0xFFF == 0xFF8` (select bank 0) or `0xFF9` (bank 1)
- Any **access** (read or write) to a hotspot triggers the switch

```python
f8_trigger = (addr13 & 0xFFF) >= 0xFF8
f8_bank    = (addr13 & 0x001).astype(jnp.uint8)   # 0xFF8→0, 0xFF9→1
```

Condition: `(scheme_id == 1) & f8_trigger` — uses `jax.lax.cond` so it
compiles to a conditional select rather than branching.

### All Others (stub)

Schemes 2–11 currently return `state` unchanged on write.  They will be
implemented in Phase 3.

## ROMMetadata

`cart_write` receives a `rom_meta` object with at minimum:

```python
rom_meta.scheme_id  # jax.Array int32
```

This is populated at ROM load time by inspecting the ROM file size and
header bytes.  In the current test suite, `_NoRomMeta` (from `cpu.py`) is
used as a sentinel (scheme_id=0) for writes that can never bankswitch.

## Identifying the Scheme

At ROM load time (not yet implemented), the scheme is identified by:

1. ROM file size: 2 KB → no switch; 4 KB → no switch; 8 KB → F8 or FE or E0; etc.
2. ROM header patterns (some games embed their scheme in the filename)
3. ALE's `CartDetector.cxx` logic (final reference)

For the initial Breakout implementation, scheme_id is hardcoded to 1 (F8).
