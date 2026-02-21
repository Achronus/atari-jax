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

"""Load, store, and register-transfer opcode handlers."""

from typing import Tuple

import jax
import jax.numpy as jnp

from atari_jax.core.bus import bus_read, bus_write
from atari_jax.core.cpu._helpers import (
    _NO_ROM_META,
    _advance,
    _addr_abs,
    _addr_abs_x,
    _addr_abs_y,
    _addr_ind_x,
    _addr_ind_y,
    _addr_zp,
    _addr_zp_x,
    _addr_zp_y,
    _pc_read,
    _set_nz,
)
from atari_jax.core.state import AtariState


def _op_lda_imm(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A9
    """LDA #imm — load accumulator with immediate byte; set N, Z."""
    val = _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_lda_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A5
    """LDA zp — load accumulator from zero-page address; set N, Z."""
    val = bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_lda_zpx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B5
    """LDA zp,X — load accumulator from zero-page,X address; set N, Z."""
    val = bus_read(state, rom, _addr_zp_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_lda_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # AD
    """LDA abs — load accumulator from absolute address; set N, Z."""
    val = bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_lda_absx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # BD
    """LDA abs,X — load accumulator from absolute,X address; set N, Z."""
    val = bus_read(state, rom, _addr_abs_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_lda_absy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B9
    """LDA abs,Y — load accumulator from absolute,Y address; set N, Z."""
    val = bus_read(state, rom, _addr_abs_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_lda_indx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A1
    """LDA (ind,X) — load accumulator from (indirect,X) address; set N, Z."""
    val = bus_read(state, rom, _addr_ind_x(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(6)


def _op_lda_indy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B1
    """LDA (ind),Y — load accumulator from (indirect),Y address; set N, Z."""
    val = bus_read(state, rom, _addr_ind_y(state, rom))
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_ldx_imm(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A2
    """LDX #imm — load X register with immediate byte; set N, Z."""
    val = _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_ldx_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A6
    """LDX zp — load X register from zero-page address; set N, Z."""
    val = bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_ldx_zpy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B6
    """LDX zp,Y — load X register from zero-page,Y address; set N, Z."""
    val = bus_read(state, rom, _addr_zp_y(state, rom))
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ldx_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # AE
    """LDX abs — load X register from absolute address; set N, Z."""
    val = bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ldx_absy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # BE
    """LDX abs,Y — load X register from absolute,Y address; set N, Z."""
    val = bus_read(state, rom, _addr_abs_y(state, rom))
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ldy_imm(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A0
    """LDY #imm — load Y register with immediate byte; set N, Z."""
    val = _pc_read(state, rom, 1)
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_ldy_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A4
    """LDY zp — load Y register from zero-page address; set N, Z."""
    val = bus_read(state, rom, _addr_zp(state, rom))
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_ldy_zpx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B4
    """LDY zp,X — load Y register from zero-page,X address; set N, Z."""
    val = bus_read(state, rom, _addr_zp_x(state, rom))
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(2))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ldy_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # AC
    """LDY abs — load Y register from absolute address; set N, Z."""
    val = bus_read(state, rom, _addr_abs(state, rom))
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_ldy_absx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # BC
    """LDY abs,X — load Y register from absolute,X address; set N, Z."""
    val = bus_read(state, rom, _addr_abs_x(state, rom))
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(3))
    )
    return state.__replace__(cpu=cpu), jnp.int32(4)


def _op_sta_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 85
    """STA zp — store accumulator to zero-page address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp(state, rom), state.cpu.a)
    return _advance(state, 2), jnp.int32(3)


def _op_sta_zpx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 95
    """STA zp,X — store accumulator to zero-page,X address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp_x(state, rom), state.cpu.a)
    return _advance(state, 2), jnp.int32(4)


def _op_sta_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 8D
    """STA abs — store accumulator to absolute address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_abs(state, rom), state.cpu.a)
    return _advance(state, 3), jnp.int32(4)


def _op_sta_absx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 9D
    """STA abs,X — store accumulator to absolute,X address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_abs_x(state, rom), state.cpu.a)
    return _advance(state, 3), jnp.int32(5)


def _op_sta_absy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 99
    """STA abs,Y — store accumulator to absolute,Y address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_abs_y(state, rom), state.cpu.a)
    return _advance(state, 3), jnp.int32(5)


def _op_sta_indx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 81
    """STA (ind,X) — store accumulator to (indirect,X) address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_ind_x(state, rom), state.cpu.a)
    return _advance(state, 2), jnp.int32(6)


def _op_sta_indy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 91
    """STA (ind),Y — store accumulator to (indirect),Y address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_ind_y(state, rom), state.cpu.a)
    return _advance(state, 2), jnp.int32(6)


def _op_stx_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 86
    """STX zp — store X register to zero-page address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp(state, rom), state.cpu.x)
    return _advance(state, 2), jnp.int32(3)


def _op_stx_zpy(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 96
    """STX zp,Y — store X register to zero-page,Y address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp_y(state, rom), state.cpu.x)
    return _advance(state, 2), jnp.int32(4)


def _op_stx_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 8E
    """STX abs — store X register to absolute address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_abs(state, rom), state.cpu.x)
    return _advance(state, 3), jnp.int32(4)


def _op_sty_zp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 84
    """STY zp — store Y register to zero-page address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp(state, rom), state.cpu.y)
    return _advance(state, 2), jnp.int32(3)


def _op_sty_zpx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 94
    """STY zp,X — store Y register to zero-page,X address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_zp_x(state, rom), state.cpu.y)
    return _advance(state, 2), jnp.int32(4)


def _op_sty_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 8C
    """STY abs — store Y register to absolute address; no flags changed."""
    state = bus_write(state, _NO_ROM_META, _addr_abs(state, rom), state.cpu.y)
    return _advance(state, 3), jnp.int32(4)


def _op_tax(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # AA
    """TAX — transfer accumulator to X; set N, Z."""
    val = state.cpu.a
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_tay(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # A8
    """TAY — transfer accumulator to Y; set N, Z."""
    val = state.cpu.a
    cpu = state.cpu.__replace__(
        y=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_txa(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 8A
    """TXA — transfer X to accumulator; set N, Z."""
    val = state.cpu.x
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_tya(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 98
    """TYA — transfer Y to accumulator; set N, Z."""
    val = state.cpu.y
    cpu = state.cpu.__replace__(
        a=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_tsx(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # BA
    """TSX — transfer stack pointer to X; set N, Z."""
    val = state.cpu.sp
    cpu = state.cpu.__replace__(
        x=val, flags=_set_nz(state.cpu.flags, val), pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_txs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 9A
    """TXS — transfer X to stack pointer; no flags changed."""
    cpu = state.cpu.__replace__(sp=state.cpu.x, pc=(state.cpu.pc + jnp.uint16(1)))
    return state.__replace__(cpu=cpu), jnp.int32(2)
