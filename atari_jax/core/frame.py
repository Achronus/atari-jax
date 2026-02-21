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

"""Frame-level emulation loop — one ALE frame = 262 scanlines × 76 CPU cycles."""

import chex
import jax
import jax.numpy as jnp

from atari_jax.core.cpu import cpu_step
from atari_jax.core.state import AtariState
from atari_jax.core.tia import render_scanline


# ALE action index → SWCHA byte (active-low; bit low = direction pressed).
# Bit pattern: bit 3 = P0 right, bit 2 = P0 left (Breakout joystick convention).
_SWCHA: jax.Array = jnp.array(
    [
        0xFF, 0xFF, 0xFF, 0xF7, 0xFB, 0xFF, 0xF3, 0xFF,
        0xF7, 0xFB, 0xFF, 0xF7, 0xFB, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF,
    ],
    dtype=jnp.uint8,
)


def emulate_scanline(state: AtariState, rom: chex.Array, sl_idx: chex.Array) -> AtariState:
    """
    Emulate one scanline: run the CPU for 76 cycles then render if visible.

    The CPU is run in a `jax.lax.while_loop` until 76 cycles have elapsed.
    When WSYNC is active the CPU is stalled (1 cycle per loop iteration);
    WSYNC is cleared at the end of the scanline regardless.  Visible scanlines
    (indices 37–246) are rendered via `render_scanline` and written to
    `state.screen`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.
    sl_idx : jax.Array
        int32 — Scanline index within the frame (0–261).

    Returns
    -------
    state : AtariState
        Updated state; `screen` updated for visible scanlines, `tia.wsync`
        cleared, `tia.hpos` reset to 0.
    """
    cycles_start = state.cycles
    # Reset hpos at the start of each scanline
    state = state.__replace__(tia=state.tia.__replace__(hpos=jnp.uint8(0)))

    def _cond(s: AtariState) -> jax.Array:
        return (s.cycles - cycles_start) < jnp.int32(76)

    def _body(s: AtariState) -> AtariState:
        new_s, cyc = cpu_step(s, rom)
        # Advance hpos by cycles × 3 colour clocks
        new_hpos = (
            (new_s.tia.hpos.astype(jnp.int32) + cyc * jnp.int32(3)) % jnp.int32(228)
        ).astype(jnp.uint8)
        new_s = new_s.__replace__(tia=new_s.tia.__replace__(hpos=new_hpos))

        # WSYNC stall: advance 1 cycle without executing the instruction
        stall_hpos = (
            (s.tia.hpos.astype(jnp.int32) + jnp.int32(3)) % jnp.int32(228)
        ).astype(jnp.uint8)
        stalled_s = s.__replace__(
            cycles=(s.cycles + jnp.int32(1)).astype(jnp.int32),
            tia=s.tia.__replace__(hpos=stall_hpos),
        )
        return jax.lax.cond(s.tia.wsync, lambda: stalled_s, lambda: new_s)

    state = jax.lax.while_loop(_cond, _body, state)
    state = state.__replace__(tia=state.tia.__replace__(wsync=jnp.bool_(False)))

    # Render visible scanlines (37–246 → screen rows 0–209)
    visible = (sl_idx >= jnp.int32(37)) & (sl_idx < jnp.int32(247))
    screen_row = jnp.clip(sl_idx - jnp.int32(37), 0, 209)
    new_state, pixels = render_scanline(state)
    new_screen = new_state.screen.at[screen_row].set(pixels)
    return jax.lax.cond(
        visible,
        lambda: new_state.__replace__(screen=new_screen),
        lambda: state,
    )


def emulate_frame(state: AtariState, rom: chex.Array, action: chex.Array) -> AtariState:
    """
    Emulate one full ALE frame (262 scanlines).

    Writes the ALE action to RIOT port A (SWCHA) and sets the TIA fire flag,
    then runs 262 scanlines via `jax.lax.fori_loop`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : jax.Array
        uint8[ROM_SIZE] — Static ROM bytes.
    action : jax.Array
        int32 — ALE action index (0–17).  Action 1 = FIRE.

    Returns
    -------
    state : AtariState
        Updated state after the frame; `frame` incremented by 1.
    """
    swcha = _SWCHA[action.astype(jnp.int32)]
    state = state.__replace__(riot=state.riot.__replace__(port_a=swcha))
    state = state.__replace__(
        tia=state.tia.__replace__(
            fire=(action == jnp.int32(1)).astype(jnp.bool_)
        )
    )

    state = jax.lax.fori_loop(
        0,
        262,
        lambda sl, s: emulate_scanline(s, rom, jnp.int32(sl)),
        state,
    )

    return state.__replace__(frame=(state.frame + jnp.int32(1)).astype(jnp.int32))
