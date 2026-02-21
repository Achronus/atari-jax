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

"""Branch, jump, and subroutine opcode handlers."""

from typing import Tuple

import jax
import jax.numpy as jnp

from atari_jax.core.bus import bus_read
from atari_jax.core.cpu._helpers import (
    _C,
    _N,
    _V,
    _Z,
    _addr_abs,
    _pc_read,
    _stack_pull,
    _stack_push,
)
from atari_jax.core.state import AtariState


def _branch(state: AtariState, rom: jax.Array, taken: jax.Array) -> AtariState:
    """
    Apply a signed relative branch to PC if `taken`, otherwise advance by 2.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.
    taken : jax.Array
        bool — Whether the branch condition is met.

    Returns
    -------
    state : AtariState
        Updated state with PC set to the branch target or PC+2.
    """
    offset_raw = _pc_read(state, rom, 1).astype(jnp.int32)
    offset = jnp.where(offset_raw >= 128, offset_raw - 256, offset_raw)
    pc_next = (state.cpu.pc.astype(jnp.int32) + jnp.int32(2)) & jnp.int32(0xFFFF)
    pc_branch = (pc_next + offset) & jnp.int32(0xFFFF)
    new_pc = jnp.where(taken, pc_branch, pc_next).astype(jnp.uint16)
    return state.__replace__(cpu=state.cpu.__replace__(pc=new_pc))


def _op_bcc(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 90
    """BCC — branch if carry clear (C=0)."""
    return _branch(state, rom, (state.cpu.flags & _C) == jnp.uint8(0)), jnp.int32(2)


def _op_bcs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B0
    """BCS — branch if carry set (C=1)."""
    return _branch(state, rom, (state.cpu.flags & _C) != jnp.uint8(0)), jnp.int32(2)


def _op_beq(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # F0
    """BEQ — branch if equal / zero set (Z=1)."""
    return _branch(state, rom, (state.cpu.flags & _Z) != jnp.uint8(0)), jnp.int32(2)


def _op_bmi(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 30
    """BMI — branch if minus / negative set (N=1)."""
    return _branch(state, rom, (state.cpu.flags & _N) != jnp.uint8(0)), jnp.int32(2)


def _op_bne(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # D0
    """BNE — branch if not equal / zero clear (Z=0)."""
    return _branch(state, rom, (state.cpu.flags & _Z) == jnp.uint8(0)), jnp.int32(2)


def _op_bpl(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 10
    """BPL — branch if positive / negative clear (N=0)."""
    return _branch(state, rom, (state.cpu.flags & _N) == jnp.uint8(0)), jnp.int32(2)


def _op_bvc(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 50
    """BVC — branch if overflow clear (V=0)."""
    return _branch(state, rom, (state.cpu.flags & _V) == jnp.uint8(0)), jnp.int32(2)


def _op_bvs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 70
    """BVS — branch if overflow set (V=1)."""
    return _branch(state, rom, (state.cpu.flags & _V) != jnp.uint8(0)), jnp.int32(2)


def _op_jmp_abs(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 4C
    """JMP abs — jump to absolute address."""
    ea = _addr_abs(state, rom)
    cpu = state.cpu.__replace__(pc=jnp.uint16(ea))
    return state.__replace__(cpu=cpu), jnp.int32(3)


def _op_jmp_ind(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 6C
    """JMP (ind) — jump to indirect address; 6502 page-wrap bug preserved."""
    ptr = _addr_abs(state, rom)
    lo = bus_read(state, rom, ptr).astype(jnp.int32)
    hi_addr = (ptr & jnp.int32(0xFF00)) | ((ptr + jnp.int32(1)) & jnp.int32(0x00FF))
    hi = bus_read(state, rom, hi_addr).astype(jnp.int32)
    cpu = state.cpu.__replace__(pc=jnp.uint16(lo | (hi << 8)))
    return state.__replace__(cpu=cpu), jnp.int32(5)


def _op_jsr(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 20
    """JSR abs — jump to subroutine; push return address (PC+2) onto stack."""
    ret = (state.cpu.pc.astype(jnp.int32) + jnp.int32(2)) & jnp.int32(0xFFFF)
    hi = ((ret >> 8) & 0xFF).astype(jnp.uint8)
    lo = (ret & 0xFF).astype(jnp.uint8)
    state = _stack_push(state, hi)
    state = _stack_push(state, lo)
    ea = _addr_abs(state, rom)
    return state.__replace__(cpu=state.cpu.__replace__(pc=jnp.uint16(ea))), jnp.int32(6)


def _op_rts(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 60
    """RTS — return from subroutine; pull return address from stack and jump to PC+1."""
    state, lo = _stack_pull(state, rom)
    state, hi = _stack_pull(state, rom)
    addr = lo.astype(jnp.int32) | (hi.astype(jnp.int32) << 8)
    new_pc = (addr + jnp.int32(1)) & jnp.int32(0xFFFF)
    return state.__replace__(
        cpu=state.cpu.__replace__(pc=jnp.uint16(new_pc))
    ), jnp.int32(6)
