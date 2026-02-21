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

"""Arithmetic, logic, shift, compare, and increment/decrement opcode handlers."""

from typing import Tuple

import jax
import jax.numpy as jnp

from atari_jax.core.bus import bus_read, bus_write
from atari_jax.core.cpu._helpers import (
    _C,
    _N,
    _NO_ROM_META,
    _V,
    _Z,
    _addr_abs,
    _addr_abs_x,
    _addr_abs_y,
    _addr_ind_x,
    _addr_ind_y,
    _addr_zp,
    _addr_zp_x,
    _advance,
    _pc_read,
    _set_flag,
    _set_nz,
)
from atari_jax.core.state import AtariState


def _adc_core(
    state: AtariState, a: jax.Array, operand: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute A + operand + C and derive updated flags.

    Parameters
    ----------
    state : AtariState
        Current machine state (provides the existing flags and carry bit).
    a : jax.Array
        uint8 — Accumulator value.
    operand : jax.Array
        uint8 — Second operand (use `0xFF ^ M` for SBC).

    Returns
    -------
    result : jax.Array
        uint8 — 8-bit addition result.
    flags : jax.Array
        uint8 — Updated processor status with N, V, Z, C set.
    """
    a8 = a.astype(jnp.uint8)
    op8 = operand.astype(jnp.uint8)
    c = (state.cpu.flags & _C).astype(jnp.int32)
    r32 = a8.astype(jnp.int32) + op8.astype(jnp.int32) + c
    result8 = (r32 & jnp.int32(0xFF)).astype(jnp.uint8)
    new_c = r32 > jnp.int32(0xFF)
    v_bits = (~(a8 ^ op8)) & (a8 ^ result8) & jnp.uint8(0x80)
    new_v = v_bits != jnp.uint8(0)
    flags = _set_nz(state.cpu.flags, result8)
    flags = _set_flag(flags, _C, new_c)
    flags = _set_flag(flags, _V, new_v)
    return result8, flags


def _cmp_flags(flags: jax.Array, reg: jax.Array, operand: jax.Array) -> jax.Array:
    """
    Compute reg - operand and update N, Z, C flags (result discarded).

    Parameters
    ----------
    flags : jax.Array
        uint8 — Current processor status byte.
    reg : jax.Array
        uint8 — Register value (A, X, or Y).
    operand : jax.Array
        uint8 — Value to compare against.

    Returns
    -------
    flags : jax.Array
        uint8 — Updated flags: C=1 if reg >= operand, N/Z from difference.
    """
    r8 = (reg.astype(jnp.int32) - operand.astype(jnp.int32)).astype(jnp.uint8)
    flags = _set_nz(flags, r8)
    flags = _set_flag(flags, _C, reg.astype(jnp.uint8) >= operand.astype(jnp.uint8))
    return flags


def _op_adc_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 69
    """ADC #imm — add immediate to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(state, state.cpu.a, _pc_read(state, rom, 1))
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_adc_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 65
    """ADC zp — add zero-page byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_zp(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_adc_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 75
    """ADC zp,X — add zero-page,X byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_zp_x(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_adc_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 6D
    """ADC abs — add absolute byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_abs(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_adc_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 7D
    """ADC abs,X — add absolute,X byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_abs_x(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_adc_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 79
    """ADC abs,Y — add absolute,Y byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_abs_y(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_adc_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 61
    """ADC (ind,X) — add (indirect,X) byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_ind_x(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_adc_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 71
    """ADC (ind),Y — add (indirect),Y byte to accumulator with carry; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, bus_read(state, rom, _addr_ind_y(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_sbc_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # E9
    """SBC #imm — subtract immediate from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, jnp.uint8(0xFF) ^ _pc_read(state, rom, 1)
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_sbc_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # E5
    """SBC zp — subtract zero-page byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state, state.cpu.a, jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_zp(state, rom))
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_sbc_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # F5
    """SBC zp,X — subtract zero-page,X byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_zp_x(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_sbc_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # ED
    """SBC abs — subtract absolute byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_abs(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_sbc_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # FD
    """SBC abs,X — subtract absolute,X byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_abs_x(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_sbc_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # F9
    """SBC abs,Y — subtract absolute,Y byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_abs_y(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_sbc_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # E1
    """SBC (ind,X) — subtract (indirect,X) byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_ind_x(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_sbc_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # F1
    """SBC (ind),Y — subtract (indirect),Y byte from accumulator with borrow; set N, V, Z, C."""
    result, flags = _adc_core(
        state,
        state.cpu.a,
        jnp.uint8(0xFF) ^ bus_read(state, rom, _addr_ind_y(state, rom)),
    )
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_and_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 29
    """AND #imm — AND immediate with accumulator; set N, Z."""
    val = state.cpu.a & _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_and_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 25
    """AND zp — AND zero-page byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_and_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 35
    """AND zp,X — AND zero-page,X byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_zp_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_and_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 2D
    """AND abs — AND absolute byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_and_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 3D
    """AND abs,X — AND absolute,X byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_abs_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_and_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 39
    """AND abs,Y — AND absolute,Y byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_abs_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_and_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 21
    """AND (ind,X) — AND (indirect,X) byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_ind_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_and_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 31
    """AND (ind),Y — AND (indirect),Y byte with accumulator; set N, Z."""
    val = state.cpu.a & bus_read(state, rom, _addr_ind_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_ora_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 09
    """ORA #imm — OR immediate with accumulator; set N, Z."""
    val = state.cpu.a | _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_ora_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 05
    """ORA zp — OR zero-page byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_ora_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 15
    """ORA zp,X — OR zero-page,X byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_zp_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ora_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 0D
    """ORA abs — OR absolute byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ora_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 1D
    """ORA abs,X — OR absolute,X byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_abs_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ora_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 19
    """ORA abs,Y — OR absolute,Y byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_abs_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ora_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 01
    """ORA (ind,X) — OR (indirect,X) byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_ind_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_ora_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 11
    """ORA (ind),Y — OR (indirect),Y byte with accumulator; set N, Z."""
    val = state.cpu.a | bus_read(state, rom, _addr_ind_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_eor_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 49
    """EOR #imm — XOR immediate with accumulator; set N, Z."""
    val = state.cpu.a ^ _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_eor_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 45
    """EOR zp — XOR zero-page byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_eor_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 55
    """EOR zp,X — XOR zero-page,X byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_zp_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_eor_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 4D
    """EOR abs — XOR absolute byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_eor_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 5D
    """EOR abs,X — XOR absolute,X byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_abs_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_eor_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 59
    """EOR abs,Y — XOR absolute,Y byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_abs_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_eor_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 41
    """EOR (ind,X) — XOR (indirect,X) byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_ind_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_eor_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 51
    """EOR (ind),Y — XOR (indirect),Y byte with accumulator; set N, Z."""
    val = state.cpu.a ^ bus_read(state, rom, _addr_ind_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_cmp_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # C9
    """CMP #imm — compare accumulator with immediate; set N, Z, C."""
    flags = _cmp_flags(state.cpu.flags, state.cpu.a, _pc_read(state, rom, 1))
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(2)


def _op_cmp_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # C5
    """CMP zp — compare accumulator with zero-page byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_zp(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(3)


def _op_cmp_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # D5
    """CMP zp,X — compare accumulator with zero-page,X byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_zp_x(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(4)


def _op_cmp_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # CD
    """CMP abs — compare accumulator with absolute byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_abs(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)


def _op_cmp_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # DD
    """CMP abs,X — compare accumulator with absolute,X byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_abs_x(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)


def _op_cmp_absy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # D9
    """CMP abs,Y — compare accumulator with absolute,Y byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_abs_y(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)


def _op_cmp_indx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # C1
    """CMP (ind,X) — compare accumulator with (indirect,X) byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_ind_x(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(6)


def _op_cmp_indy(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # D1
    """CMP (ind),Y — compare accumulator with (indirect),Y byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.a, bus_read(state, rom, _addr_ind_y(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(5)


def _op_cpx_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # E0
    """CPX #imm — compare X with immediate; set N, Z, C."""
    flags = _cmp_flags(state.cpu.flags, state.cpu.x, _pc_read(state, rom, 1))
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(2)


def _op_cpx_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # E4
    """CPX zp — compare X with zero-page byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.x, bus_read(state, rom, _addr_zp(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(3)


def _op_cpx_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # EC
    """CPX abs — compare X with absolute byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.x, bus_read(state, rom, _addr_abs(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)


def _op_cpy_imm(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # C0
    """CPY #imm — compare Y with immediate; set N, Z, C."""
    flags = _cmp_flags(state.cpu.flags, state.cpu.y, _pc_read(state, rom, 1))
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(2)


def _op_cpy_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # C4
    """CPY zp — compare Y with zero-page byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.y, bus_read(state, rom, _addr_zp(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(3)


def _op_cpy_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # CC
    """CPY abs — compare Y with absolute byte; set N, Z, C."""
    flags = _cmp_flags(
        state.cpu.flags, state.cpu.y, bus_read(state, rom, _addr_abs(state, rom))
    )
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)


def _op_inc_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # E6
    """INC zp — increment zero-page byte; set N, Z."""
    addr = _addr_zp(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) + jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_inc_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # F6
    """INC zp,X — increment zero-page,X byte; set N, Z."""
    addr = _addr_zp_x(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) + jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_inc_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # EE
    """INC abs — increment absolute byte; set N, Z."""
    addr = _addr_abs(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) + jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_inc_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # FE
    """INC abs,X — increment absolute,X byte; set N, Z."""
    addr = _addr_abs_x(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) + jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_inx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # E8
    """INX — increment X register; set N, Z."""
    result = (state.cpu.x.astype(jnp.int32) + jnp.int32(1)).astype(jnp.uint8)
    cpu = state.cpu.__replace__(
        x=result,
        flags=_set_nz(state.cpu.flags, result),
        pc=(state.cpu.pc + jnp.uint16(1)),
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_iny(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # C8
    """INY — increment Y register; set N, Z."""
    result = (state.cpu.y.astype(jnp.int32) + jnp.int32(1)).astype(jnp.uint8)
    cpu = state.cpu.__replace__(
        y=result,
        flags=_set_nz(state.cpu.flags, result),
        pc=(state.cpu.pc + jnp.uint16(1)),
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_dec_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # C6
    """DEC zp — decrement zero-page byte; set N, Z."""
    addr = _addr_zp(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) - jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_dec_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # D6
    """DEC zp,X — decrement zero-page,X byte; set N, Z."""
    addr = _addr_zp_x(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) - jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_dec_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # CE
    """DEC abs — decrement absolute byte; set N, Z."""
    addr = _addr_abs(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) - jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_dec_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # DE
    """DEC abs,X — decrement absolute,X byte; set N, Z."""
    addr = _addr_abs_x(state, rom)
    result = (bus_read(state, rom, addr).astype(jnp.int32) - jnp.int32(1)).astype(
        jnp.uint8
    )
    new_flags = _set_nz(state.cpu.flags, result)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_dex(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # CA
    """DEX — decrement X register; set N, Z."""
    result = (state.cpu.x.astype(jnp.int32) - jnp.int32(1)).astype(jnp.uint8)
    cpu = state.cpu.__replace__(
        x=result,
        flags=_set_nz(state.cpu.flags, result),
        pc=(state.cpu.pc + jnp.uint16(1)),
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_dey(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 88
    """DEY — decrement Y register; set N, Z."""
    result = (state.cpu.y.astype(jnp.int32) - jnp.int32(1)).astype(jnp.uint8)
    cpu = state.cpu.__replace__(
        y=result,
        flags=_set_nz(state.cpu.flags, result),
        pc=(state.cpu.pc + jnp.uint16(1)),
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_asl_acc(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 0A
    """ASL A — shift accumulator left; old bit 7 → C; set N, Z, C."""
    val = state.cpu.a
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) << jnp.int32(1)).astype(jnp.uint8)
    flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_asl_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 06
    """ASL zp — shift zero-page byte left; old bit 7 → C; set N, Z, C."""
    addr = _addr_zp(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) << jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_asl_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 16
    """ASL zp,X — shift zero-page,X byte left; old bit 7 → C; set N, Z, C."""
    addr = _addr_zp_x(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) << jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_asl_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 0E
    """ASL abs — shift absolute byte left; old bit 7 → C; set N, Z, C."""
    addr = _addr_abs(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) << jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_asl_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 1E
    """ASL abs,X — shift absolute,X byte left; old bit 7 → C; set N, Z, C."""
    addr = _addr_abs_x(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) << jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_lsr_acc(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 4A
    """LSR A — shift accumulator right; old bit 0 → C; set N, Z, C."""
    val = state.cpu.a
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) >> jnp.int32(1)).astype(jnp.uint8)
    flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_lsr_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 46
    """LSR zp — shift zero-page byte right; old bit 0 → C; set N, Z, C."""
    addr = _addr_zp(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) >> jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_lsr_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 56
    """LSR zp,X — shift zero-page,X byte right; old bit 0 → C; set N, Z, C."""
    addr = _addr_zp_x(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) >> jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_lsr_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 4E
    """LSR abs — shift absolute byte right; old bit 0 → C; set N, Z, C."""
    addr = _addr_abs(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) >> jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_lsr_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 5E
    """LSR abs,X — shift absolute,X byte right; old bit 0 → C; set N, Z, C."""
    addr = _addr_abs_x(state, rom)
    val = bus_read(state, rom, addr)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = (val.astype(jnp.int32) >> jnp.int32(1)).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_rol_acc(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 2A
    """ROL A — rotate accumulator left through carry; old bit 7 → C; set N, Z, C."""
    val = state.cpu.a
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = ((val.astype(jnp.int32) << jnp.int32(1)) | old_c).astype(jnp.uint8)
    flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_rol_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 26
    """ROL zp — rotate zero-page byte left through carry; old bit 7 → C; set N, Z, C."""
    addr = _addr_zp(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = ((val.astype(jnp.int32) << jnp.int32(1)) | old_c).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_rol_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 36
    """ROL zp,X — rotate zero-page,X byte left through carry; old bit 7 → C; set N, Z, C."""
    addr = _addr_zp_x(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = ((val.astype(jnp.int32) << jnp.int32(1)) | old_c).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_rol_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 2E
    """ROL abs — rotate absolute byte left through carry; old bit 7 → C; set N, Z, C."""
    addr = _addr_abs(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = ((val.astype(jnp.int32) << jnp.int32(1)) | old_c).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_rol_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 3E
    """ROL abs,X — rotate absolute,X byte left through carry; old bit 7 → C; set N, Z, C."""
    addr = _addr_abs_x(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x80)) != jnp.uint8(0)
    result = ((val.astype(jnp.int32) << jnp.int32(1)) | old_c).astype(jnp.uint8)
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_ror_acc(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 6A
    """ROR A — rotate accumulator right through carry; old bit 0 → C; set N, Z, C."""
    val = state.cpu.a
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = ((old_c << jnp.int32(7)) | (val.astype(jnp.int32) >> jnp.int32(1))).astype(
        jnp.uint8
    )
    flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    cpu = state.cpu.__replace__(
        a=result, flags=flags, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_ror_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 66
    """ROR zp — rotate zero-page byte right through carry; old bit 0 → C; set N, Z, C."""
    addr = _addr_zp(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = ((old_c << jnp.int32(7)) | (val.astype(jnp.int32) >> jnp.int32(1))).astype(
        jnp.uint8
    )
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(5)


def _op_ror_zpx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 76
    """ROR zp,X — rotate zero-page,X byte right through carry; old bit 0 → C; set N, Z, C."""
    addr = _addr_zp_x(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = ((old_c << jnp.int32(7)) | (val.astype(jnp.int32) >> jnp.int32(1))).astype(
        jnp.uint8
    )
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 2
    ), jnp.int32(6)


def _op_ror_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 6E
    """ROR abs — rotate absolute byte right through carry; old bit 0 → C; set N, Z, C."""
    addr = _addr_abs(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = ((old_c << jnp.int32(7)) | (val.astype(jnp.int32) >> jnp.int32(1))).astype(
        jnp.uint8
    )
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(6)


def _op_ror_absx(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 7E
    """ROR abs,X — rotate absolute,X byte right through carry; old bit 0 → C; set N, Z, C."""
    addr = _addr_abs_x(state, rom)
    val = bus_read(state, rom, addr)
    old_c = (state.cpu.flags & _C).astype(jnp.int32)
    new_c = (val & jnp.uint8(0x01)) != jnp.uint8(0)
    result = ((old_c << jnp.int32(7)) | (val.astype(jnp.int32) >> jnp.int32(1))).astype(
        jnp.uint8
    )
    new_flags = _set_flag(_set_nz(state.cpu.flags, result), _C, new_c)
    state = bus_write(state, _NO_ROM_META, addr, result)
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=new_flags)), 3
    ), jnp.int32(7)


def _op_bit_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 24
    """BIT zp — test bits: Z = (A & M)==0, N = M7, V = M6."""
    m = bus_read(state, rom, _addr_zp(state, rom))
    flags = _set_flag(state.cpu.flags, _Z, (state.cpu.a & m) == jnp.uint8(0))
    flags = _set_flag(flags, _N, (m & jnp.uint8(0x80)) != jnp.uint8(0))
    flags = _set_flag(flags, _V, (m & jnp.uint8(0x40)) != jnp.uint8(0))
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 2
    ), jnp.int32(3)


def _op_bit_abs(
    state: AtariState, rom: jax.Array
) -> Tuple[AtariState, jax.Array]:  # 2C
    """BIT abs — test bits: Z = (A & M)==0, N = M7, V = M6."""
    m = bus_read(state, rom, _addr_abs(state, rom))
    flags = _set_flag(state.cpu.flags, _Z, (state.cpu.a & m) == jnp.uint8(0))
    flags = _set_flag(flags, _N, (m & jnp.uint8(0x80)) != jnp.uint8(0))
    flags = _set_flag(flags, _V, (m & jnp.uint8(0x40)) != jnp.uint8(0))
    return _advance(
        state.__replace__(cpu=state.cpu.__replace__(flags=flags)), 3
    ), jnp.int32(4)
