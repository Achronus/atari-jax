# bus.py — System Bus & Address Decoder

**File:** `atarax/core/bus.py`
**ALE reference:** `System.cxx` (install loop), `M6532.cxx`

## Purpose

The Atari 2600 uses a MOS 6507 CPU which has only 13 address lines, giving a
0x0000–0x1FFF address space.  Multiple chips share that space via hardware
address decoding.  `bus.py` replicates that decoding in pure JAX and
dispatches reads/writes to the correct chip module.

## Address Decoding

The 6507's 13 bits are decoded by examining specific bits — not by checking
ranges.  This is how the real hardware works and means some addresses are
mirrored many times across the space.

```text
Region      Condition (13-bit address)          Chip       Region index
────────────────────────────────────────────────────────────────────────
TIA         default (all other)                 tia.py     0
RIOT RAM    (addr & 0x1080) == 0x0080           riot.py    1
              AND (addr & 0x0200) == 0
RIOT I/O    (addr & 0x1080) == 0x0080           riot.py    2
              AND (addr & 0x0200) != 0
ROM         (addr & 0x1000) != 0                cart.py    3
```

### Key bit positions

| Bit | Hex    | Meaning |
|-----|--------|---------|
| 12  | 0x1000 | ROM select (any address with bit-12 set hits cartridge) |
| 7   | 0x0080 | RIOT select (must be set together with bit-6) |
| 6   | 0x0040 | — |
| 9   | 0x0200 | RIOT I/O vs RAM (bit-9 set = I/O registers) |

### Stack page split

The 6502 stack page (0x100–0x1FF) is split:

- 0x100–0x17F: bit-7 = 0 → routes to **TIA** (reads 0, writes are silently
  dropped by the TIA register shadow for indices > 0x3F)
- 0x180–0x1FF: bit-7 = 1 → routes to **RIOT RAM** (usable stack space)

In practice the 6502 SP is initialised to 0xFF, so the first push writes to
0x1FF (RIOT RAM index 0x7F) and the stack grows down through RIOT RAM only.

## Implementation: `_region(addr13)`

```python
def _region(addr13):
    a = addr13.astype(jnp.int32)
    is_rom      = (a & 0x1000) != 0
    is_riot_any = (a & 0x1080) == 0x0080
    is_riot_io  = is_riot_any & ((a & 0x0200) != 0)
    is_riot_ram = is_riot_any & ((a & 0x0200) == 0)
    return jnp.where(is_rom,      3,
           jnp.where(is_riot_io,  2,
           jnp.where(is_riot_ram, 1, 0)))
```

Returns an int32 in {0, 1, 2, 3}.  Priority: ROM > RIOT I/O > RIOT RAM > TIA.

## Public API

### `bus_read(state, rom, addr) -> uint8`

Reads one byte.  `addr` is masked to 13 bits internally.

```python
byte = bus_read(state, rom, jnp.int32(0x1000))  # first byte of ROM bank 0
```

Dispatch via `jax.lax.switch(region, [tia_read, riot_ram_read, riot_io_read, cart_read], addr13)`.

### `bus_write(state, rom_meta, addr, value) -> AtariState`

Writes one byte, returns updated state.  `rom_meta` carries the ROM's
bankswitching scheme ID — needed for cartridge writes that trigger bank
switches.

```python
state = bus_write(state, rom_meta, jnp.int32(0x80), jnp.uint8(0x42))
```

## Notes

- `rom` is never mutated — it is a static `uint8[ROM_SIZE]` passed as a
  non-state argument so XLA can treat it as a compile-time constant.
- `rom_meta` only needs a `.scheme_id` attribute (an int32 JAX scalar).
  `cpu.py` supplies a `_NoRomMeta` sentinel with `scheme_id = 0` for writes
  that can never touch the cartridge (stack ops, zero-page, etc.).
