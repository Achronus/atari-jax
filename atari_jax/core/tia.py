"""TIA — Television Interface Adaptor."""

import jax
import jax.numpy as jnp

from atari_jax.core.state import AtariState


def tia_read(state: AtariState, addr13: jax.Array) -> jax.Array:
    """
    Read one byte from the TIA.

    Most TIA registers are write-only; this stub returns 0 for all addresses.
    Phase 2 will return collision-latch bytes for addresses 0x00–0x07.

    Parameters
    ----------
    state : AtariState
        Current machine state (unused in Phase 1).
    addr13 : jax.Array
        int32 — 13-bit bus address.

    Returns
    -------
    byte : jax.Array
        uint8 — Always 0x00 in Phase 1.
    """
    return jnp.uint8(0)


def tia_write(state: AtariState, addr13: jax.Array, value: jax.Array) -> AtariState:
    """
    Shadow a TIA register write into `state.tia.regs`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address; register index = `addr13 & 0x3F`.
    value : jax.Array
        uint8 — Value to shadow.

    Returns
    -------
    state : AtariState
        Updated state with `tia.regs[addr & 0x3F]` set to `value`.
    """
    reg_idx = addr13 & jnp.int32(0x3F)
    new_regs = state.tia.regs.at[reg_idx].set(value.astype(jnp.uint8))
    return state.__replace__(tia=state.tia.__replace__(regs=new_regs))
