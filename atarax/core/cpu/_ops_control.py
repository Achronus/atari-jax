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

"""Stack, flag, interrupt, and system opcode handlers."""

from typing import Tuple

import jax
import jax.numpy as jnp

from atarax.core.bus import bus_read
from atarax.core.cpu._helpers import (
    _B,
    _C,
    _D,
    _I,
    _V,
    _advance,
    _set_nz,
    _stack_pull,
    _stack_push,
)
from atarax.core.state import AtariState


def _op_nop(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # EA
    """NOP — no operation; advance PC by 1."""
    return _advance(state, 1), jnp.int32(2)


def _op_pha(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 48
    """PHA — push accumulator onto stack."""
    state = _stack_push(state, state.cpu.a)
    return _advance(state, 1), jnp.int32(3)


def _op_php(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 08
    """PHP — push processor status (flags) onto stack with B set."""
    state = _stack_push(state, state.cpu.flags | _B)
    return _advance(state, 1), jnp.int32(3)


def _op_pla(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 68
    """PLA — pull accumulator from stack; set N, Z."""
    state, val = _stack_pull(state, rom)
    cpu = state.cpu.__replace__(a=val, flags=_set_nz(state.cpu.flags, val))
    return _advance(state.__replace__(cpu=cpu), 1), jnp.int32(4)


def _op_plp(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 28
    """PLP — pull processor status from stack; bit-5 and B forced set."""
    state, val = _stack_pull(state, rom)
    new_flags = (val | jnp.uint8(0x20)) | _B
    cpu = state.cpu.__replace__(flags=new_flags)
    return _advance(state.__replace__(cpu=cpu), 1), jnp.int32(4)


def _op_clc(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 18
    """CLC — clear carry flag (C=0)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags & ~_C, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_sec(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 38
    """SEC — set carry flag (C=1)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags | _C, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_cld(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # D8
    """CLD — clear decimal flag (D=0)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags & ~_D, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_sed(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # F8
    """SED — set decimal flag (D=1)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags | _D, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_cli(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 58
    """CLI — clear interrupt disable flag (I=0)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags & ~_I, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_sei(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 78
    """SEI — set interrupt disable flag (I=1)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags | _I, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_clv(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # B8
    """CLV — clear overflow flag (V=0)."""
    cpu = state.cpu.__replace__(
        flags=state.cpu.flags & ~_V, pc=(state.cpu.pc + jnp.uint16(1))
    )
    return state.__replace__(cpu=cpu), jnp.int32(2)


def _op_brk(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 00
    """BRK — software interrupt; push PC+2 and flags (B set), jump via IRQ vector."""
    ret = (state.cpu.pc.astype(jnp.int32) + jnp.int32(2)) & jnp.int32(0xFFFF)
    hi = ((ret >> 8) & 0xFF).astype(jnp.uint8)
    lo = (ret & 0xFF).astype(jnp.uint8)
    state = _stack_push(state, hi)
    state = _stack_push(state, lo)
    state = _stack_push(state, state.cpu.flags | _B)
    vec_lo = bus_read(state, rom, jnp.int32(0xFFFE)).astype(jnp.int32)
    vec_hi = bus_read(state, rom, jnp.int32(0xFFFF)).astype(jnp.int32)
    new_pc = jnp.uint16(vec_lo | (vec_hi << 8))
    cpu = state.cpu.__replace__(pc=new_pc, flags=state.cpu.flags | _I)
    return state.__replace__(cpu=cpu), jnp.int32(7)


def _op_rti(state: AtariState, rom: jax.Array) -> Tuple[AtariState, jax.Array]:  # 40
    """RTI — return from interrupt; pull flags then PC from stack."""
    state, f = _stack_pull(state, rom)
    state, lo = _stack_pull(state, rom)
    state, hi = _stack_pull(state, rom)
    new_flags = (f | jnp.uint8(0x20)) | _B
    new_pc = jnp.uint16(lo.astype(jnp.int32) | (hi.astype(jnp.int32) << 8))
    cpu = state.cpu.__replace__(pc=new_pc, flags=new_flags)
    return state.__replace__(cpu=cpu), jnp.int32(6)
