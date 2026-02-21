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

"""Flag constants, addressing-mode functions, and shared CPU helpers."""

from typing import Tuple

import jax
import jax.numpy as jnp

from atarax.core.bus import bus_read, bus_write
from atarax.core.cart import ROMMetadata
from atarax.core.state import AtariState

_N = jnp.uint8(0x80)  # Negative
_V = jnp.uint8(0x40)  # Overflow
_B = jnp.uint8(0x10)  # Break
_D = jnp.uint8(0x08)  # Decimal
_I = jnp.uint8(0x04)  # Interrupt disable
_Z = jnp.uint8(0x02)  # Zero
_C = jnp.uint8(0x01)  # Carry

_NO_ROM_META = ROMMetadata(scheme_id=jnp.int32(0))


def _set_flag(flags: jax.Array, mask: jax.Array, val: jax.Array) -> jax.Array:
    """
    Set or clear the flag bit(s) in `mask` based on the truth value of `val`.

    Parameters
    ----------
    flags : jax.Array
        uint8 — Current processor status byte.
    mask : jax.Array
        uint8 — Bit mask identifying the flag(s) to modify.
    val : jax.Array
        bool-like — If truthy the flag is set; otherwise cleared.

    Returns
    -------
    flags : jax.Array
        uint8 — Updated processor status byte.
    """
    return jnp.where(val, flags | mask, flags & ~mask)


def _set_nz(flags: jax.Array, val: jax.Array) -> jax.Array:
    """
    Update N and Z flags to reflect a uint8 result value.

    Parameters
    ----------
    flags : jax.Array
        uint8 — Current processor status byte.
    val : jax.Array
        uint8 — Result of the operation that drives N/Z.

    Returns
    -------
    flags : jax.Array
        uint8 — Updated processor status byte with N and Z set or cleared.
    """
    v = val.astype(jnp.uint8)
    f = _set_flag(flags, _N, (v & jnp.uint8(0x80)) != jnp.uint8(0))
    f = _set_flag(f, _Z, v == jnp.uint8(0))
    return f


def _pc_read(state: AtariState, rom: jax.Array, offset: int) -> jax.Array:
    """
    Read one byte from the bus at address PC + `offset`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.
    offset : int
        Byte offset from the current PC (e.g. 1 for the first operand byte).

    Returns
    -------
    byte : jax.Array
        uint8 — Byte at address (PC + offset) & 0xFFFF.
    """
    addr = (state.cpu.pc.astype(jnp.int32) + jnp.int32(offset)) & jnp.int32(0xFFFF)
    return bus_read(state, rom, addr)


def _addr_zp(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Zero-page addressing: return the one-byte operand as an address.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — Effective address in the range 0x0000–0x00FF.
    """
    return _pc_read(state, rom, 1).astype(jnp.int32)


def _addr_zp_x(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Zero-page,X addressing: (operand + X) & 0xFF.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — Effective address, wrapping within page zero.
    """
    base = _pc_read(state, rom, 1).astype(jnp.int32)
    return (base + state.cpu.x.astype(jnp.int32)) & jnp.int32(0xFF)


def _addr_zp_y(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Zero-page,Y addressing: (operand + Y) & 0xFF.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — Effective address, wrapping within page zero.
    """
    base = _pc_read(state, rom, 1).astype(jnp.int32)
    return (base + state.cpu.y.astype(jnp.int32)) & jnp.int32(0xFF)


def _addr_abs(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Absolute addressing: 16-bit address from two operand bytes (lo, hi).

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — Full 16-bit effective address.
    """
    lo = _pc_read(state, rom, 1).astype(jnp.int32)
    hi = _pc_read(state, rom, 2).astype(jnp.int32)
    return lo | (hi << 8)


def _addr_abs_x(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Absolute,X addressing: absolute address + X.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — 16-bit effective address.
    """
    return (_addr_abs(state, rom) + state.cpu.x.astype(jnp.int32)) & jnp.int32(0xFFFF)


def _addr_abs_y(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    Absolute,Y addressing: absolute address + Y.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — 16-bit effective address.
    """
    return (_addr_abs(state, rom) + state.cpu.y.astype(jnp.int32)) & jnp.int32(0xFFFF)


def _addr_ind_x(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    (Indirect,X) addressing: zero-page pointer pre-indexed by X.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — 16-bit effective address read from the zero-page pointer.
    """
    zp = (
        _pc_read(state, rom, 1).astype(jnp.int32) + state.cpu.x.astype(jnp.int32)
    ) & 0xFF
    lo = bus_read(state, rom, zp).astype(jnp.int32)
    hi = bus_read(state, rom, (zp + 1) & jnp.int32(0xFF)).astype(jnp.int32)
    return lo | (hi << 8)


def _addr_ind_y(state: AtariState, rom: jax.Array) -> jax.Array:
    """
    (Indirect),Y addressing: zero-page pointer post-indexed by Y.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    addr : jax.Array
        int32 — 16-bit effective address read from the zero-page pointer + Y.
    """
    zp = _pc_read(state, rom, 1).astype(jnp.int32)
    lo = bus_read(state, rom, zp).astype(jnp.int32)
    hi = bus_read(state, rom, (zp + jnp.int32(1)) & jnp.int32(0xFF)).astype(jnp.int32)
    base = lo | (hi << 8)
    return (base + state.cpu.y.astype(jnp.int32)) & jnp.int32(0xFFFF)


def _stack_push(state: AtariState, value: jax.Array) -> AtariState:
    """
    Push one byte onto the hardware stack and decrement SP.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    value : jax.Array
        uint8 — Byte to push.

    Returns
    -------
    state : AtariState
        Updated state with the byte written to 0x0100 + SP and SP decremented.
    """
    addr = jnp.int32(0x0100) | state.cpu.sp.astype(jnp.int32)
    state = bus_write(state, _NO_ROM_META, addr, value)
    new_sp = (state.cpu.sp.astype(jnp.int32) - 1) & jnp.int32(0xFF)
    return state.__replace__(cpu=state.cpu.__replace__(sp=new_sp.astype(jnp.uint8)))


def _stack_pull(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:
    """
    Increment SP and pull one byte from the hardware stack.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.

    Returns
    -------
    state : AtariState
        Updated state with SP incremented.
    value : jax.Array
        uint8 — Byte pulled from 0x0100 + new SP.
    """
    new_sp = (state.cpu.sp.astype(jnp.int32) + 1) & jnp.int32(0xFF)
    state = state.__replace__(cpu=state.cpu.__replace__(sp=new_sp.astype(jnp.uint8)))
    addr = jnp.int32(0x0100) | new_sp
    val = bus_read(state, rom, addr)
    return state, val


_CYCLES = jnp.array(
    [
        7,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        3,
        2,
        2,
        2,
        4,
        4,
        6,
        6,  # 0x
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # 1x
        6,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        4,
        2,
        2,
        2,
        4,
        4,
        6,
        6,  # 2x
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # 3x
        6,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        3,
        2,
        2,
        2,
        3,
        4,
        6,
        6,  # 4x
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # 5x
        6,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        4,
        2,
        2,
        2,
        5,
        4,
        6,
        6,  # 6x
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # 7x
        2,
        6,
        2,
        6,
        3,
        3,
        3,
        3,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,  # 8x
        2,
        6,
        2,
        6,
        4,
        4,
        4,
        4,
        2,
        5,
        2,
        5,
        5,
        5,
        5,
        5,  # 9x
        2,
        6,
        2,
        6,
        3,
        3,
        3,
        4,
        2,
        2,
        2,
        2,
        4,
        4,
        4,
        4,  # ax
        2,
        5,
        2,
        5,
        4,
        4,
        4,
        4,
        2,
        4,
        2,
        4,
        4,
        4,
        4,
        4,  # bx
        2,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        2,
        2,
        2,
        2,
        4,
        4,
        6,
        6,  # cx
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # dx
        2,
        6,
        2,
        8,
        3,
        3,
        5,
        5,
        2,
        2,
        2,
        2,
        4,
        4,
        6,
        6,  # ex
        2,
        5,
        2,
        8,
        4,
        4,
        6,
        6,
        2,
        4,
        2,
        7,
        4,
        4,
        7,
        7,  # fx
    ],
    dtype=jnp.int32,
)


def _advance(state: AtariState, by: int) -> AtariState:
    """
    Advance the program counter by `by` bytes.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    by : int
        Number of bytes to add to PC (typically the instruction length).

    Returns
    -------
    state : AtariState
        Updated state with PC incremented.
    """
    new_pc = (state.cpu.pc.astype(jnp.int32) + jnp.int32(by)) & jnp.int32(0xFFFF)
    return state.__replace__(cpu=state.cpu.__replace__(pc=new_pc.astype(jnp.uint16)))


def _op_undef(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:
    """Advance PC by 1 for any undefined or unimplemented opcode; cycles from `_CYCLES`."""
    opcode = bus_read(state, rom, state.cpu.pc.astype(jnp.int32)).astype(jnp.int32)
    return _advance(state, 1), _CYCLES[opcode]
