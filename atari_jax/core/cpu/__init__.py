"""MOS 6507 CPU — opcode table, fetch/decode/execute via `jax.lax.switch`.

JAX constraint: no Python branching on traced values.  All opcode dispatch
uses `jax.lax.switch(opcode, OPCODE_TABLE, state, rom)`.

Opcode handler contract
-----------------------
Every entry in `OPCODE_TABLE` shares the same signature::

    def _op_xxx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:
        ...

Parameters
----------
state : AtariState
    Current machine state.  Must not be mutated; return a new state via
    `.__replace__()`.
rom : jax.Array
    uint8[ROM_SIZE] — Static ROM bytes (never mutated).

Returns
-------
state : AtariState
    Updated machine state (PC already advanced, registers/flags set).
cycles : jax.Array
    int32 — CPU cycles consumed by this instruction.

Unimplemented slots fall through to `_op_undef` (advance PC by 1).

Implemented: load/store, transfer, branch, jump/subroutine, stack, flag ops,
             NOP, BRK, RTI.
Pending: ADC, SBC, INC, DEC, AND, ORA, EOR, CMP, CPX, CPY, ASL, LSR, ROL,
         ROR, BIT.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from atari_jax.core.bus import bus_read
from atari_jax.core.cpu._helpers import _op_undef
from atari_jax.core.cpu._ops_branch import (
    _op_bcc,
    _op_bcs,
    _op_beq,
    _op_bmi,
    _op_bne,
    _op_bpl,
    _op_bvc,
    _op_bvs,
    _op_jmp_abs,
    _op_jmp_ind,
    _op_jsr,
    _op_rts,
)
from atari_jax.core.cpu._ops_control import (
    _op_brk,
    _op_clc,
    _op_cld,
    _op_cli,
    _op_clv,
    _op_nop,
    _op_pha,
    _op_php,
    _op_pla,
    _op_plp,
    _op_rti,
    _op_sec,
    _op_sed,
    _op_sei,
)
from atari_jax.core.cpu._ops_load_store import (
    _op_lda_abs,
    _op_lda_absx,
    _op_lda_absy,
    _op_lda_imm,
    _op_lda_indx,
    _op_lda_indy,
    _op_lda_zp,
    _op_lda_zpx,
    _op_ldx_abs,
    _op_ldx_absy,
    _op_ldx_imm,
    _op_ldx_zp,
    _op_ldx_zpy,
    _op_ldy_abs,
    _op_ldy_absx,
    _op_ldy_imm,
    _op_ldy_zp,
    _op_ldy_zpx,
    _op_sta_abs,
    _op_sta_absx,
    _op_sta_absy,
    _op_sta_indx,
    _op_sta_indy,
    _op_sta_zp,
    _op_sta_zpx,
    _op_stx_abs,
    _op_stx_zp,
    _op_stx_zpy,
    _op_sty_abs,
    _op_sty_zp,
    _op_sty_zpx,
    _op_tax,
    _op_tay,
    _op_tsx,
    _op_txa,
    _op_txs,
    _op_tya,
)
from atari_jax.core.state import AtariState

_T = [_op_undef] * 256

# Load / Store
_T[0xA9] = _op_lda_imm
_T[0xA5] = _op_lda_zp
_T[0xB5] = _op_lda_zpx
_T[0xAD] = _op_lda_abs
_T[0xBD] = _op_lda_absx
_T[0xB9] = _op_lda_absy
_T[0xA1] = _op_lda_indx
_T[0xB1] = _op_lda_indy

_T[0xA2] = _op_ldx_imm
_T[0xA6] = _op_ldx_zp
_T[0xB6] = _op_ldx_zpy
_T[0xAE] = _op_ldx_abs
_T[0xBE] = _op_ldx_absy

_T[0xA0] = _op_ldy_imm
_T[0xA4] = _op_ldy_zp
_T[0xB4] = _op_ldy_zpx
_T[0xAC] = _op_ldy_abs
_T[0xBC] = _op_ldy_absx

_T[0x85] = _op_sta_zp
_T[0x95] = _op_sta_zpx
_T[0x8D] = _op_sta_abs
_T[0x9D] = _op_sta_absx
_T[0x99] = _op_sta_absy
_T[0x81] = _op_sta_indx
_T[0x91] = _op_sta_indy

_T[0x86] = _op_stx_zp
_T[0x96] = _op_stx_zpy
_T[0x8E] = _op_stx_abs
_T[0x84] = _op_sty_zp
_T[0x94] = _op_sty_zpx
_T[0x8C] = _op_sty_abs

# Transfers
_T[0xAA] = _op_tax
_T[0xA8] = _op_tay
_T[0x8A] = _op_txa
_T[0x98] = _op_tya
_T[0xBA] = _op_tsx
_T[0x9A] = _op_txs

# Branch
_T[0x90] = _op_bcc
_T[0xB0] = _op_bcs
_T[0xF0] = _op_beq
_T[0x30] = _op_bmi
_T[0xD0] = _op_bne
_T[0x10] = _op_bpl
_T[0x50] = _op_bvc
_T[0x70] = _op_bvs

# Jump / subroutine
_T[0x4C] = _op_jmp_abs
_T[0x6C] = _op_jmp_ind
_T[0x20] = _op_jsr
_T[0x60] = _op_rts

# Stack
_T[0x48] = _op_pha
_T[0x08] = _op_php
_T[0x68] = _op_pla
_T[0x28] = _op_plp

# Flag operations
_T[0x18] = _op_clc
_T[0x38] = _op_sec
_T[0xD8] = _op_cld
_T[0xF8] = _op_sed
_T[0x58] = _op_cli
_T[0x78] = _op_sei
_T[0xB8] = _op_clv

# Interrupt / NOP
_T[0x00] = _op_brk
_T[0x40] = _op_rti
_T[0xEA] = _op_nop

OPCODE_TABLE = _T


def cpu_step(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:
    """
    Fetch, decode, and execute one 6507 instruction.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    state : AtariState
        Updated machine state after the instruction, with `cycles` incremented.
    cycles : jax.Array
        int32 — CPU cycles consumed by this instruction.
    """
    opcode = bus_read(state, rom, state.cpu.pc.astype(jnp.int32)).astype(jnp.int32)
    new_state, cycles = jax.lax.switch(opcode, OPCODE_TABLE, state, rom)
    new_state = new_state.__replace__(
        cycles=(new_state.cycles + cycles).astype(jnp.int32)
    )
    return new_state, cycles


def cpu_reset(state: AtariState, rom: jax.Array) -> AtariState:
    """
    Load the reset vector from 0xFFFC/0xFFFD and set PC.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    state : AtariState
        Updated state with `cpu.pc` pointing at the reset vector address.
    """
    lo = bus_read(state, rom, jnp.int32(0xFFFC)).astype(jnp.int32)
    hi = bus_read(state, rom, jnp.int32(0xFFFD)).astype(jnp.int32)
    return state.__replace__(cpu=state.cpu.__replace__(pc=jnp.uint16(lo | (hi << 8))))
