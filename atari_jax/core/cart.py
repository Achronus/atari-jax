"""ROM cartridge + bankswitching schemes."""

import chex
import jax
import jax.numpy as jnp

from atari_jax.core.state import AtariState


@chex.dataclass
class ROMMetadata:
    """
    Static cartridge metadata passed alongside the ROM array.

    Parameters
    ----------
    scheme_id : jax.Array
        int32 — Bankswitching scheme: 0=none, 1=F8, 2=F6, 3=F4, …
    """

    scheme_id: jax.Array


def cart_read(rom: jax.Array, bank: jax.Array, addr13: jax.Array) -> jax.Array:
    """
    Read one byte from the cartridge ROM.

    Parameters
    ----------
    rom : jax.Array
        uint8[ROM_SIZE] — Full ROM bytes (static, never mutated).
    bank : jax.Array
        uint8 — Currently active 4 KB bank index.
    addr13 : jax.Array
        int32 — 13-bit bus address (0x1000–0x1FFF).

    Returns
    -------
    byte : jax.Array
        uint8 — Byte at the selected bank + page offset.
    """
    page_offset = addr13 & jnp.int32(0x0FFF)
    physical = bank.astype(jnp.int32) * jnp.int32(4096) + page_offset
    return rom[physical]


def cart_write(
    state: AtariState,
    scheme_id: jax.Array,
    addr13: jax.Array,
    value: jax.Array,
) -> AtariState:
    """
    Handle bankswitching writes.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    scheme_id : jax.Array
        int32 — Bankswitching scheme index (0=none, 1=F8, 2=F6, 3=F4, …).
    addr13 : jax.Array
        int32 — 13-bit bus address of the triggering write.
    value : jax.Array
        uint8 — Value being written (unused by F8; may matter for other
        schemes such as 3F which encode the bank in the written byte).

    Returns
    -------
    state : AtariState
        Updated state with `bank` reflecting any triggered bank switch.

    Notes
    -----
    Only F8 (scheme_id=1) is implemented.  All others are no-ops.
    F8 switches on any access to 0xFF8 (bank 0) or 0xFF9 (bank 1).
    """
    # F8: two 4 KB banks, switched by accessing 0xFF8 or 0xFF9.
    f8_trigger = (addr13 & jnp.int32(0xFFF)) >= jnp.int32(0xFF8)
    f8_bank = (addr13 & jnp.int32(0x001)).astype(jnp.uint8)

    new_bank = jax.lax.cond(
        (scheme_id == jnp.int32(1)) & f8_trigger,
        lambda: f8_bank,
        lambda: state.bank,
    )
    return state.__replace__(bank=new_bank)
