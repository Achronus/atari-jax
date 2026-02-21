# cpu.py — MOS 6507 CPU

**File:** `atarax/core/cpu.py`
**ALE reference:** `M6502.cxx`, `M6502Hi.ins`

## Overview

Implements the MOS 6507 (a 6502 variant with 13 address lines and no
interrupt pins) as a JAX-traceable state machine.

The central JAX constraint: **no Python branching on traced values**.
The opcode byte fetched at runtime is a JAX array — you cannot do
`if opcode == 0xA9: ...` inside a jit-compiled function.  Instead, every
opcode is a Python function in a 256-entry list, and dispatch happens via:

```python
new_state, cycles = jax.lax.switch(opcode, OPCODE_TABLE, state, rom)
```

`jax.lax.switch` compiles all 256 branches and selects the correct one at
runtime without Python control flow.

## Opcode Function Signature

Every entry in `OPCODE_TABLE` must have exactly this signature:

```python
def _op_xxx(state: AtariState, rom: jax.Array) -> tuple[AtariState, jax.Array]:
    ...
    return new_state, jnp.int32(cycles)
```

The cycle count is returned (not added to state) so the caller can accumulate
it: `cpu_step` adds it to `state.cycles` after the switch.

Unimplemented opcodes use `_op_undef` which advances PC by 1 and returns the
cycle count from `_CYCLES[opcode]`.

## Implemented Opcodes (Week 1)

### Load / Store

| Opcode | Mnemonic | Addr Mode | Cycles |
|--------|----------|-----------|--------|
| A9 | LDA | Immediate | 2 |
| A5 | LDA | Zero page | 3 |
| B5 | LDA | Zero page,X | 4 |
| AD | LDA | Absolute | 4 |
| BD | LDA | Absolute,X | 4 |
| B9 | LDA | Absolute,Y | 4 |
| A1 | LDA | (Indirect,X) | 6 |
| B1 | LDA | (Indirect),Y | 5 |
| A2 | LDX | Immediate | 2 |
| A6 | LDX | Zero page | 3 |
| B6 | LDX | Zero page,Y | 4 |
| AE | LDX | Absolute | 4 |
| BE | LDX | Absolute,Y | 4 |
| A0 | LDY | Immediate | 2 |
| A4 | LDY | Zero page | 3 |
| B4 | LDY | Zero page,X | 4 |
| AC | LDY | Absolute | 4 |
| BC | LDY | Absolute,X | 4 |
| 85 | STA | Zero page | 3 |
| 95 | STA | Zero page,X | 4 |
| 8D | STA | Absolute | 4 |
| 9D | STA | Absolute,X | 5 |
| 99 | STA | Absolute,Y | 5 |
| 81 | STA | (Indirect,X) | 6 |
| 91 | STA | (Indirect),Y | 6 |
| 86 | STX | Zero page | 3 |
| 96 | STX | Zero page,Y | 4 |
| 8E | STX | Absolute | 4 |
| 84 | STY | Zero page | 3 |
| 94 | STY | Zero page,X | 4 |
| 8C | STY | Absolute | 4 |

### Register Transfers (all 2 cycles)

| Opcode | Mnemonic | Sets flags |
|--------|----------|------------|
| AA | TAX | N, Z |
| A8 | TAY | N, Z |
| 8A | TXA | N, Z |
| 98 | TYA | N, Z |
| BA | TSX | N, Z |
| 9A | TXS | — |

### Branches (2 cycles not-taken; taken adds 1; page cross adds 1 more)

| Opcode | Mnemonic | Condition |
|--------|----------|-----------|
| 90 | BCC | C = 0 |
| B0 | BCS | C = 1 |
| F0 | BEQ | Z = 1 |
| 30 | BMI | N = 1 |
| D0 | BNE | Z = 0 |
| 10 | BPL | N = 0 |
| 50 | BVC | V = 0 |
| 70 | BVS | V = 1 |

> Note: branch page-crossing penalty (+1 cycle) is not yet implemented.
> The base 2-cycle cost is returned regardless.

### Jump / Subroutine

| Opcode | Mnemonic | Cycles | Notes |
|--------|----------|--------|-------|
| 4C | JMP abs | 3 | Direct jump |
| 6C | JMP ind | 5 | Page-wrap bug preserved: hi byte read from `(ptr & 0xFF00) \| ((ptr+1) & 0x00FF)` |
| 20 | JSR | 6 | Pushes return address - 1 |
| 60 | RTS | 6 | Pops address + 1 |

### Stack

| Opcode | Mnemonic | Cycles | Notes |
|--------|----------|--------|-------|
| 48 | PHA | 3 | Push A |
| 08 | PHP | 3 | Push flags with B set |
| 68 | PLA | 4 | Pull A, sets N/Z |
| 28 | PLP | 4 | Pull flags; bit-5 forced 1, B forced 1 |

### Flag Operations (all 2 cycles)

| Opcode | Mnemonic | Effect |
|--------|----------|--------|
| 18 | CLC | Clear carry |
| 38 | SEC | Set carry |
| D8 | CLD | Clear decimal |
| F8 | SED | Set decimal |
| 58 | CLI | Clear interrupt disable |
| 78 | SEI | Set interrupt disable |
| B8 | CLV | Clear overflow |

### Other

| Opcode | Mnemonic | Cycles | Notes |
|--------|----------|--------|-------|
| EA | NOP | 2 | No operation |
| 00 | BRK | 7 | Push PC+2, push flags\|B, load IRQ vector 0xFFFE/0xFFFF |
| 40 | RTI | 6 | Pull flags, pull PC (no +1 unlike RTS) |

## Week 2 (TODO)

ADC, SBC, INC, DEC, AND, ORA, EOR, CMP, CPX, CPY, ASL, LSR, ROL, ROR, BIT
and their addressing mode variants.

## Addressing Modes

| Helper | Mode | Bytes | Description |
|--------|------|-------|-------------|
| `_addr_zp` | Zero page | 2 | `uint8` operand = direct zero-page address |
| `_addr_zp_x` | Zero page,X | 2 | `(zp + X) & 0xFF` — wraps within page 0 |
| `_addr_zp_y` | Zero page,Y | 2 | `(zp + Y) & 0xFF` |
| `_addr_abs` | Absolute | 3 | 16-bit address from bytes 1+2 |
| `_addr_abs_x` | Absolute,X | 3 | `abs + X` |
| `_addr_abs_y` | Absolute,Y | 3 | `abs + Y` |
| `_addr_ind_x` | (Indirect,X) | 2 | `zp = (byte1 + X) & 0xFF`; read 16-bit ptr from `zp` |
| `_addr_ind_y` | (Indirect),Y | 2 | Read 16-bit ptr from `byte1`; add Y to the result |

All helpers take `(state, rom)` and return an `int32` effective address.

## Stack Mechanics

SP is a zero-page offset into page 1 (0x100–0x1FF).

- **Push:** write to `0x100 | SP`, then `SP = (SP - 1) & 0xFF`
- **Pull:** `SP = (SP + 1) & 0xFF`, then read from `0x100 | SP`

Addresses 0x180–0x1FF map to RIOT RAM (the usable stack region).
Addresses 0x100–0x17F map to TIA (harmless on push; reads return 0 on pull).

`_NoRomMeta` sentinel (scheme_id=0) is passed to `bus_write` for all stack
and zero-page operations — these never reach the cartridge so the bankswitch
logic is a no-op.

## Cycle Count Table

`_CYCLES` is the 256-entry `int32` array transcribed verbatim from
`M6502.cxx: ourInstructionProcessorCycleTable`.  It gives the base cycle
cost for each opcode.  `cpu_step` indexes it for `_op_undef` fallbacks;
implemented opcodes return their own hardcoded constant.

## Public API

### `cpu_step(state, rom) -> (AtariState, int32)`

Fetch opcode at PC, dispatch via `jax.lax.switch`, return updated state and
cycles used.  Also accumulates cycles into `state.cycles`.

### `cpu_reset(state, rom) -> AtariState`

Reads the reset vector from addresses 0xFFFC (lo) and 0xFFFD (hi), sets PC
to that address.  Called once after ROM is loaded before any emulation.

### `OPCODE_TABLE`

The 256-entry list of opcode functions.  Exposed so tests can inspect or
temporarily override entries.
