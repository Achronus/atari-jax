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

"""System bus — Atari 2600 memory map and chip dispatch.

Atari 2600 address decoding (6507 has 13 address lines, 0x0000–0x1FFF):

  Region      Condition                          Chip
  ─────────────────────────────────────────────────────
  ROM         (addr & 0x1000) != 0               Cartridge
  RIOT RAM    (addr & 0x1080) == 0x0080          M6532 RAM
                AND (addr & 0x0200) == 0
  RIOT I/O    (addr & 0x1080) == 0x0080          M6532 I/O
                AND (addr & 0x0200) != 0
  TIA         all other (bit-7=0, bit-12=0)      TIA

The stack (page 1: 0x100–0x1FF) splits between TIA (0x100–0x17F, bit-7=0)
and RIOT RAM (0x180–0x1FF, bit-7=1).  In practice the 6502 SP starts at
0xFF so valid stack addresses are 0x1FF–0x180, all in RIOT RAM.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.core import cart, riot, tia
from atarax.core.cart import ROMMetadata
from atarax.core.state import AtariState


def _region(addr13: jax.Array) -> jax.Array:
    """
    Map a 13-bit address to a chip region index.

    Parameters
    ----------
    addr13 : jax.Array
        int32 — 13-bit bus address (0x0000–0x1FFF).

    Returns
    -------
    jax.Array
        int32 — Region index: 0=TIA, 1=RIOT RAM, 2=RIOT I/O, 3=ROM.
    """
    a = addr13.astype(jnp.int32)
    is_rom = (a & 0x1000) != 0
    is_riot_any = (a & 0x1080) == 0x0080
    is_riot_io = is_riot_any & ((a & 0x0200) != 0)
    is_riot_ram = is_riot_any & ((a & 0x0200) == 0)

    return jnp.where(
        is_rom,
        jnp.int32(3),
        jnp.where(
            is_riot_io, jnp.int32(2), jnp.where(is_riot_ram, jnp.int32(1), jnp.int32(0))
        ),
    )


def bus_read(state: AtariState, rom: chex.Array, addr: chex.Array) -> chex.Array:
    """
    Read one byte from the Atari bus.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes (never mutated).
    addr : jax.Array
        Bus address (any width); masked to 13 bits internally.

    Returns
    -------
    byte : jax.Array
        uint8 — Byte read from the addressed chip.
    """
    addr13 = addr.astype(jnp.int32) & jnp.int32(0x1FFF)
    region = _region(addr13)

    return jax.lax.switch(
        region,
        [
            lambda a: tia.tia_read(state, a),
            lambda a: riot.riot_ram_read(state, a),
            lambda a: riot.riot_io_read(state, a),
            lambda a: cart.cart_read(rom, state.bank, a),
        ],
        addr13,
    )


def bus_write(
    state: AtariState,
    rom_meta: ROMMetadata,
    addr: chex.Array,
    value: chex.Array,
) -> AtariState:
    """
    Write one byte to the Atari bus.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom_meta : ROMMetadata
        Cartridge metadata; `scheme_id` selects the active bankswitching scheme.
    addr : jax.Array
        Bus address (any width); masked to 13 bits internally.
    value : jax.Array
        Byte to write (any integer dtype; truncated to uint8 internally).

    Returns
    -------
    state : AtariState
        Updated machine state.
    """
    addr13 = addr.astype(jnp.int32) & jnp.int32(0x1FFF)
    value8 = value.astype(jnp.uint8)
    region = _region(addr13)

    return jax.lax.switch(
        region,
        [
            lambda a, v: tia.tia_write(state, a, v),
            lambda a, v: riot.riot_ram_write(state, a, v),
            lambda a, v: riot.riot_io_write(state, a, v),
            lambda a, v: cart.cart_write(state, rom_meta.scheme_id, a, v),
        ],
        addr13,
        value8,
    )
