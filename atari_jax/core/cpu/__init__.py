# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""MOS 6507 CPU — opcode table, fetch/decode/execute via `jax.lax.switch`.

JAX constraint: no Python branching on traced values.  All opcode dispatch
uses `jax.lax.switch(opcode, OPCODE_TABLE, state, rom)`.

Opcode handler contract
-----------------------
Every entry in `OPCODE_TABLE` shares the same signature::

    def _op_xxx(state: AtariState, rom: chex.Array) -> Tuple[AtariState, chex.Array]:
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
             NOP, BRK, RTI, ADC, SBC, AND, ORA, EOR, CMP, CPX, CPY, INC, DEC,
             INX, INY, DEX, DEY, ASL, LSR, ROL, ROR, BIT.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from atari_jax.core.bus import bus_read
from atari_jax.core.cpu._helpers import _op_undef
from atari_jax.core.cpu._ops_alu import (
    _op_adc_abs,
    _op_adc_absx,
    _op_adc_absy,
    _op_adc_imm,
    _op_adc_indx,
    _op_adc_indy,
    _op_adc_zp,
    _op_adc_zpx,
    _op_and_abs,
    _op_and_absx,
    _op_and_absy,
    _op_and_imm,
    _op_and_indx,
    _op_and_indy,
    _op_and_zp,
    _op_and_zpx,
    _op_asl_abs,
    _op_asl_absx,
    _op_asl_acc,
    _op_asl_zp,
    _op_asl_zpx,
    _op_bit_abs,
    _op_bit_zp,
    _op_cmp_abs,
    _op_cmp_absx,
    _op_cmp_absy,
    _op_cmp_imm,
    _op_cmp_indx,
    _op_cmp_indy,
    _op_cmp_zp,
    _op_cmp_zpx,
    _op_cpx_abs,
    _op_cpx_imm,
    _op_cpx_zp,
    _op_cpy_abs,
    _op_cpy_imm,
    _op_cpy_zp,
    _op_dec_abs,
    _op_dec_absx,
    _op_dec_zp,
    _op_dec_zpx,
    _op_dex,
    _op_dey,
    _op_eor_abs,
    _op_eor_absx,
    _op_eor_absy,
    _op_eor_imm,
    _op_eor_indx,
    _op_eor_indy,
    _op_eor_zp,
    _op_eor_zpx,
    _op_inc_abs,
    _op_inc_absx,
    _op_inc_zp,
    _op_inc_zpx,
    _op_inx,
    _op_iny,
    _op_lsr_abs,
    _op_lsr_absx,
    _op_lsr_acc,
    _op_lsr_zp,
    _op_lsr_zpx,
    _op_ora_abs,
    _op_ora_absx,
    _op_ora_absy,
    _op_ora_imm,
    _op_ora_indx,
    _op_ora_indy,
    _op_ora_zp,
    _op_ora_zpx,
    _op_rol_abs,
    _op_rol_absx,
    _op_rol_acc,
    _op_rol_zp,
    _op_rol_zpx,
    _op_ror_abs,
    _op_ror_absx,
    _op_ror_acc,
    _op_ror_zp,
    _op_ror_zpx,
    _op_sbc_abs,
    _op_sbc_absx,
    _op_sbc_absy,
    _op_sbc_imm,
    _op_sbc_indx,
    _op_sbc_indy,
    _op_sbc_zp,
    _op_sbc_zpx,
)
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

# ADC
_T[0x69] = _op_adc_imm
_T[0x65] = _op_adc_zp
_T[0x75] = _op_adc_zpx
_T[0x6D] = _op_adc_abs
_T[0x7D] = _op_adc_absx
_T[0x79] = _op_adc_absy
_T[0x61] = _op_adc_indx
_T[0x71] = _op_adc_indy

# SBC
_T[0xE9] = _op_sbc_imm
_T[0xE5] = _op_sbc_zp
_T[0xF5] = _op_sbc_zpx
_T[0xED] = _op_sbc_abs
_T[0xFD] = _op_sbc_absx
_T[0xF9] = _op_sbc_absy
_T[0xE1] = _op_sbc_indx
_T[0xF1] = _op_sbc_indy

# AND
_T[0x29] = _op_and_imm
_T[0x25] = _op_and_zp
_T[0x35] = _op_and_zpx
_T[0x2D] = _op_and_abs
_T[0x3D] = _op_and_absx
_T[0x39] = _op_and_absy
_T[0x21] = _op_and_indx
_T[0x31] = _op_and_indy

# ORA
_T[0x09] = _op_ora_imm
_T[0x05] = _op_ora_zp
_T[0x15] = _op_ora_zpx
_T[0x0D] = _op_ora_abs
_T[0x1D] = _op_ora_absx
_T[0x19] = _op_ora_absy
_T[0x01] = _op_ora_indx
_T[0x11] = _op_ora_indy

# EOR
_T[0x49] = _op_eor_imm
_T[0x45] = _op_eor_zp
_T[0x55] = _op_eor_zpx
_T[0x4D] = _op_eor_abs
_T[0x5D] = _op_eor_absx
_T[0x59] = _op_eor_absy
_T[0x41] = _op_eor_indx
_T[0x51] = _op_eor_indy

# CMP
_T[0xC9] = _op_cmp_imm
_T[0xC5] = _op_cmp_zp
_T[0xD5] = _op_cmp_zpx
_T[0xCD] = _op_cmp_abs
_T[0xDD] = _op_cmp_absx
_T[0xD9] = _op_cmp_absy
_T[0xC1] = _op_cmp_indx
_T[0xD1] = _op_cmp_indy

# CPX / CPY
_T[0xE0] = _op_cpx_imm
_T[0xE4] = _op_cpx_zp
_T[0xEC] = _op_cpx_abs
_T[0xC0] = _op_cpy_imm
_T[0xC4] = _op_cpy_zp
_T[0xCC] = _op_cpy_abs

# INC / DEC
_T[0xE6] = _op_inc_zp
_T[0xF6] = _op_inc_zpx
_T[0xEE] = _op_inc_abs
_T[0xFE] = _op_inc_absx
_T[0xE8] = _op_inx
_T[0xC8] = _op_iny
_T[0xC6] = _op_dec_zp
_T[0xD6] = _op_dec_zpx
_T[0xCE] = _op_dec_abs
_T[0xDE] = _op_dec_absx
_T[0xCA] = _op_dex
_T[0x88] = _op_dey

# ASL / LSR / ROL / ROR
_T[0x0A] = _op_asl_acc
_T[0x06] = _op_asl_zp
_T[0x16] = _op_asl_zpx
_T[0x0E] = _op_asl_abs
_T[0x1E] = _op_asl_absx
_T[0x4A] = _op_lsr_acc
_T[0x46] = _op_lsr_zp
_T[0x56] = _op_lsr_zpx
_T[0x4E] = _op_lsr_abs
_T[0x5E] = _op_lsr_absx
_T[0x2A] = _op_rol_acc
_T[0x26] = _op_rol_zp
_T[0x36] = _op_rol_zpx
_T[0x2E] = _op_rol_abs
_T[0x3E] = _op_rol_absx
_T[0x6A] = _op_ror_acc
_T[0x66] = _op_ror_zp
_T[0x76] = _op_ror_zpx
_T[0x6E] = _op_ror_abs
_T[0x7E] = _op_ror_absx

# BIT
_T[0x24] = _op_bit_zp
_T[0x2C] = _op_bit_abs

OPCODE_TABLE = _T


def cpu_step(state: AtariState, rom: chex.Array) -> Tuple[AtariState, chex.Array]:
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


def cpu_reset(state: AtariState, rom: chex.Array) -> AtariState:
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
