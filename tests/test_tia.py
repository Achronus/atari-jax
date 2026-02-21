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

"""Unit tests for atari_jax.core.tia — palette, rasteriser, and register writes.

Run with:
    pytest tests/test_tia.py -v
"""

import jax.numpy as jnp

from atari_jax.core.state import new_atari_state
from atari_jax.core.tia import _NTSC_PALETTE, render_scanline, tia_write


def _state_with_regs(reg_writes):
    """Return a fresh AtariState with selected TIA registers pre-written."""
    state = new_atari_state()
    for addr, val in reg_writes.items():
        state = tia_write(state, jnp.int32(addr), jnp.uint8(val))
    return state


def test_ntsc_palette_shape():
    assert _NTSC_PALETTE.shape == (128, 3)
    assert _NTSC_PALETTE.dtype == jnp.uint8


def test_ntsc_palette_grey_ramp():
    # Hue 0 (indices 0, 16, 32, …) should be grey (R == G == B).
    for lum in range(8):
        idx = lum * 16
        r, g, b = (
            int(_NTSC_PALETTE[idx, 0]),
            int(_NTSC_PALETTE[idx, 1]),
            int(_NTSC_PALETTE[idx, 2]),
        )
        assert r == g == b, f"Expected grey at palette[{idx}], got ({r},{g},{b})"


def test_render_scanline_background():
    # All TIA regs zero → background colour 0x00 → palette index 0 → black.
    state = new_atari_state()
    _, pixels = render_scanline(state)
    assert pixels.shape == (160, 3)
    assert int(jnp.sum(pixels)) == 0, "Expected all-black scanline"


def test_render_scanline_background_colour():
    # COLUBK (reg 0x09) = 0x0E → palette index 7 → luma 7 grey.
    state = _state_with_regs({0x09: 0x0E})
    _, pixels = render_scanline(state)
    expected = _NTSC_PALETTE[7]
    assert jnp.all(pixels == expected), "Background colour mismatch"


def test_render_scanline_playfield_left():
    # PF0 = 0xF0 → bits 7:4 set → left groups 0-3 active → pixels 0-15 filled.
    # CTRLPF=0 (copy mode): right half duplicates left → pixels 80-95 also filled.
    # COLUPF (0x08) = 0x0E, COLUBK (0x09) = 0x00.
    state = _state_with_regs({0x0D: 0xF0, 0x08: 0x0E})
    _, pixels = render_scanline(state)
    pf_colour = _NTSC_PALETTE[7]
    assert jnp.all(pixels[:16] == pf_colour), "Left PF groups 0-3 should be filled"
    assert jnp.all(pixels[80:96] == pf_colour), (
        "Right PF groups 0-3 (copy) should be filled"
    )
    assert int(jnp.sum(pixels[16:80])) == 0, "Mid pixels 16-79 should be background"
    assert int(jnp.sum(pixels[96:])) == 0, "End pixels 96-159 should be background"


def test_player_sprite_position():
    # GRP0 (0x1B) = 0xFF (all 8 bits set), p0_pos set via RESP0 at hpos=68.
    state = new_atari_state()
    # Set hpos = 68 then write RESP0 → p0_pos = 0
    state = state.__replace__(tia=state.tia.__replace__(hpos=jnp.uint8(68)))
    state = tia_write(state, jnp.int32(0x10), jnp.uint8(0))  # RESP0
    state = tia_write(state, jnp.int32(0x1B), jnp.uint8(0xFF))  # GRP0
    state = tia_write(state, jnp.int32(0x06), jnp.uint8(0x0E))  # COLUP0
    _, pixels = render_scanline(state)
    # p0_pos == 0 → pixels 0-7 should be COLUP0 colour
    p0_colour = _NTSC_PALETTE[7]
    assert jnp.all(pixels[:8] == p0_colour), "Player 0 should occupy pixels 0-7"
    assert int(jnp.sum(pixels[8:16])) == 0, "Pixels 8-15 should be background"


def test_hmove_shifts_sprite():
    # Place p0 at pixel 10, then HMOVE with HMP0 nibble = 1 (shift left by 1).
    state = new_atari_state()
    state = state.__replace__(tia=state.tia.__replace__(p0_pos=jnp.uint8(10)))
    # HMP0 (reg 0x20): nibble 1 in high nibble = 0x10
    state = tia_write(state, jnp.int32(0x20), jnp.uint8(0x10))
    state = tia_write(state, jnp.int32(0x2A), jnp.uint8(0))  # HMOVE
    assert int(state.tia.p0_pos) == 9, f"Expected p0_pos=9, got {int(state.tia.p0_pos)}"


def test_cxclr_clears_collisions():
    # Manually set a collision bit then write CXCLR.
    state = new_atari_state()
    state = state.__replace__(tia=state.tia.__replace__(collisions=jnp.uint16(0xFFFF)))
    state = tia_write(state, jnp.int32(0x2C), jnp.uint8(0))  # CXCLR
    assert int(state.tia.collisions) == 0, "CXCLR should zero all collision bits"


def test_wsync_flag_set_by_write():
    # Writing to WSYNC (0x02) should set the wsync flag.
    state = new_atari_state()
    assert not bool(state.tia.wsync)
    state = tia_write(state, jnp.int32(0x02), jnp.uint8(0))
    assert bool(state.tia.wsync), "wsync should be True after write to 0x02"
