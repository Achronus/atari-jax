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

"""TIA — Television Interface Adaptor: register shadow, rasteriser, and I/O."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from atarax.core.state import AtariState


def _make_ntsc_palette() -> jax.Array:
    """
    Compute the 128-entry Atari NTSC colour look-up table.

    Returns
    -------
    palette : jax.Array
        uint8[128, 3] — RGB values for each 7-bit Atari colour index
        (bits 7:1 of a TIA colour register).
    """
    idx = jnp.arange(128, dtype=jnp.int32)
    hue = (idx & 0x0F).astype(jnp.int32)
    lum = ((idx >> 4) & 0x07).astype(jnp.int32)

    # Grey ramp: luma 0-7 → 0, 34, 68, …, 238
    grey_v = (lum * 34).astype(jnp.uint8)
    grey_rgb = jnp.stack([grey_v, grey_v, grey_v], axis=1)  # [128, 3]

    # Coloured entries: YIQ model, hue 1-15 at 26° increments
    angle_rad = (hue - 1).astype(jnp.float32) * (26.0 * jnp.pi / 180.0)
    saturation = jnp.float32(0.55)
    lightness = jnp.float32(0.07) + lum.astype(jnp.float32) * jnp.float32(0.12)

    y = lightness
    i = saturation * jnp.cos(angle_rad)
    q = saturation * jnp.sin(angle_rad)

    r = y + jnp.float32(0.9563) * i + jnp.float32(0.6210) * q
    g = y - jnp.float32(0.2721) * i - jnp.float32(0.6474) * q
    b = y - jnp.float32(1.1070) * i + jnp.float32(1.7046) * q

    color_rgb = jnp.stack(
        [
            jnp.clip(r * 255, 0, 255).astype(jnp.uint8),
            jnp.clip(g * 255, 0, 255).astype(jnp.uint8),
            jnp.clip(b * 255, 0, 255).astype(jnp.uint8),
        ],
        axis=1,
    )  # [128, 3]

    return jnp.where((hue == 0)[:, None], grey_rgb, color_rgb)


_NTSC_PALETTE: jax.Array = _make_ntsc_palette()


# Number of sprite copies for each nusiz[2:0] value
_NUSIZ_COPIES: jax.Array = jnp.array([1, 2, 2, 3, 2, 1, 3, 1], dtype=jnp.int32)

# Pixel offsets for each of three possible copies (unused slots = 0)
_NUSIZ_OFFSETS: jax.Array = jnp.array(
    [
        [0, 0, 0],  # 0: one copy
        [0, 16, 0],  # 1: two copies close
        [0, 32, 0],  # 2: two copies medium
        [0, 16, 32],  # 3: three copies close
        [0, 64, 0],  # 4: two copies wide
        [0, 0, 0],  # 5: one copy double-width
        [0, 32, 64],  # 6: three copies medium
        [0, 0, 0],  # 7: one copy quadruple-width
    ],
    dtype=jnp.int32,
)


def _hpos_to_pixel(hpos: jax.Array) -> jax.Array:
    """
    Convert a TIA horizontal colour-clock position to a screen pixel index.

    Parameters
    ----------
    hpos : jax.Array
        uint8 — Colour-clock counter (0–227).  Visible pixels start at 68.

    Returns
    -------
    pixel : jax.Array
        uint8 — Screen pixel index (0–159); clamped to 0 when hpos < 68.
    """
    raw = hpos.astype(jnp.int32) - jnp.int32(68)
    return jnp.clip(raw, 0, 159).astype(jnp.uint8)


def _build_pf_mask(
    pf0: chex.Array,
    pf1: chex.Array,
    pf2: chex.Array,
    ctrlpf: chex.Array,
) -> chex.Array:
    """
    Expand PF0/PF1/PF2 register bits into a 160-pixel playfield mask.

    Parameters
    ----------
    pf0 : jax.Array
        uint8 — TIA PF0 register (bits 7:4 used, MSB-first left-to-right).
    pf1 : jax.Array
        uint8 — TIA PF1 register (bits 7:0, MSB-first).
    pf2 : jax.Array
        uint8 — TIA PF2 register (bits 0:7, LSB-first).
    ctrlpf : jax.Array
        uint8 — TIA CTRLPF register; bit 0 = reflect right half.

    Returns
    -------
    mask : jax.Array
        bool[160] — True where playfield is active.
    """
    # Left half: 20 groups of 4 pixels each.
    # PF0 bits 7,6,5,4  → groups 0-3   (bit 7 first)
    # PF1 bits 7,6,5,4,3,2,1,0 → groups 4-11 (bit 7 first)
    # PF2 bits 0,1,2,3,4,5,6,7 → groups 12-19 (bit 0 first)
    pf0_shifts = jnp.array([7, 6, 5, 4], dtype=jnp.uint8)
    pf1_shifts = jnp.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=jnp.uint8)
    pf2_shifts = jnp.arange(8, dtype=jnp.uint8)

    pf0_bits = ((pf0 >> pf0_shifts) & jnp.uint8(1)).astype(jnp.bool_)
    pf1_bits = ((pf1 >> pf1_shifts) & jnp.uint8(1)).astype(jnp.bool_)
    pf2_bits = ((pf2 >> pf2_shifts) & jnp.uint8(1)).astype(jnp.bool_)

    left_20 = jnp.concatenate([pf0_bits, pf1_bits, pf2_bits])  # bool[20]
    right_20 = jax.lax.cond(
        (ctrlpf & jnp.uint8(1)).astype(jnp.bool_),
        lambda: jnp.flip(left_20),
        lambda: left_20,
    )
    pf_groups = jnp.concatenate([left_20, right_20])  # bool[40]
    return jnp.repeat(pf_groups, 4)  # bool[160]


def _player_mask(
    grp: chex.Array,
    pos: chex.Array,
    refp: chex.Array,
    nusiz: chex.Array,
    px: chex.Array,
) -> chex.Array:
    """
    Build a 160-pixel mask for a player sprite.

    Parameters
    ----------
    grp : jax.Array
        uint8 — GRP0/GRP1 pattern register.
    pos : jax.Array
        uint8 — Horizontal pixel position (0–159).
    refp : jax.Array
        bool — Reflect the pattern (REFP0/REFP1 bit 3).
    nusiz : jax.Array
        uint8 — NUSIZ0/NUSIZ1 register.
    px : jax.Array
        int32[160] — Pixel indices `jnp.arange(160)`.

    Returns
    -------
    mask : jax.Array
        bool[160] — True where the player sprite is active.
    """
    sz = (nusiz & jnp.uint8(7)).astype(jnp.int32)
    n_copies = _NUSIZ_COPIES[sz]
    offsets = _NUSIZ_OFFSETS[sz]  # int32[3]
    pos_i = pos.astype(jnp.int32)

    # Stretch width: modes 5=double, 7=quadruple
    width = jnp.where(sz == jnp.int32(5), jnp.int32(2), jnp.int32(1))
    width = jnp.where(sz == jnp.int32(7), jnp.int32(4), width)

    # Bit-reverse uint8 for REFP using three swap passes
    grp_b = grp.astype(jnp.uint8)
    grp_b = jax.lax.cond(
        refp.astype(jnp.bool_),
        lambda g: (
            ((g & jnp.uint8(0xF0)) >> jnp.uint8(4))
            | ((g & jnp.uint8(0x0F)) << jnp.uint8(4))
        ).astype(jnp.uint8),
        lambda g: g,
        grp_b,
    )
    grp_b = jax.lax.cond(
        refp.astype(jnp.bool_),
        lambda g: (
            ((g & jnp.uint8(0xCC)) >> jnp.uint8(2))
            | ((g & jnp.uint8(0x33)) << jnp.uint8(2))
        ).astype(jnp.uint8),
        lambda g: g,
        grp_b,
    )
    grp_b = jax.lax.cond(
        refp.astype(jnp.bool_),
        lambda g: (
            ((g & jnp.uint8(0xAA)) >> jnp.uint8(1))
            | ((g & jnp.uint8(0x55)) << jnp.uint8(1))
        ).astype(jnp.uint8),
        lambda g: g,
        grp_b,
    )

    def _copy_mask(copy_idx: int) -> jax.Array:
        start = (pos_i + offsets[copy_idx]) % jnp.int32(160)
        rel = (px - start) % jnp.int32(160)
        bit_idx = rel // width
        in_range = (rel >= jnp.int32(0)) & (rel < jnp.int32(8) * width)
        bit_val = (grp_b >> (jnp.int32(7) - bit_idx).astype(jnp.uint8)) & jnp.uint8(1)
        return in_range & bit_val.astype(jnp.bool_)

    mask = _copy_mask(0)
    mask = mask | jnp.where(
        n_copies >= jnp.int32(2),
        _copy_mask(1),
        jnp.zeros(160, dtype=jnp.bool_),
    )
    mask = mask | jnp.where(
        n_copies >= jnp.int32(3),
        _copy_mask(2),
        jnp.zeros(160, dtype=jnp.bool_),
    )
    return mask


def _missile_mask(
    enabled: jax.Array,
    pos: jax.Array,
    nusiz: jax.Array,
    px: jax.Array,
) -> jax.Array:
    """
    Build a 160-pixel mask for a missile sprite.

    Parameters
    ----------
    enabled : jax.Array
        bool — ENAM0/ENAM1 bit 1.
    pos : jax.Array
        uint8 — Horizontal pixel position (0–159).
    nusiz : jax.Array
        uint8 — NUSIZ0/NUSIZ1 register; width = 1 << bits[5:4].
    px : jax.Array
        int32[160] — Pixel indices.

    Returns
    -------
    mask : jax.Array
        bool[160] — True where the missile is active.
    """
    mw = jnp.int32(1) << ((nusiz >> jnp.uint8(4)) & jnp.uint8(3)).astype(jnp.int32)
    pos_i = pos.astype(jnp.int32)
    active = (px >= pos_i) & (px < pos_i + mw)
    return active & enabled.astype(jnp.bool_)


def _ball_mask(
    enabled: jax.Array,
    pos: jax.Array,
    ctrlpf: jax.Array,
    px: jax.Array,
) -> jax.Array:
    """
    Build a 160-pixel mask for the ball sprite.

    Parameters
    ----------
    enabled : jax.Array
        bool — ENABL bit 1.
    pos : jax.Array
        uint8 — Horizontal pixel position (0–159).
    ctrlpf : jax.Array
        uint8 — CTRLPF register; ball width = 1 << bits[5:4].
    px : jax.Array
        int32[160] — Pixel indices.

    Returns
    -------
    mask : jax.Array
        bool[160] — True where the ball is active.
    """
    bw = jnp.int32(1) << ((ctrlpf >> jnp.uint8(4)) & jnp.uint8(3)).astype(jnp.int32)
    pos_i = pos.astype(jnp.int32)
    active = (px >= pos_i) & (px < pos_i + bw)
    return active & enabled.astype(jnp.bool_)


def _update_collisions(
    collisions: jax.Array,
    p0: jax.Array,
    p1: jax.Array,
    m0: jax.Array,
    m1: jax.Array,
    bl: jax.Array,
    pf: jax.Array,
) -> jax.Array:
    """
    OR new collision bits into the accumulated collision word.

    Parameters
    ----------
    collisions : jax.Array
        uint16 — Existing latched collision bits.
    p0 : jax.Array
        bool[160] — Player 0 mask.
    p1 : jax.Array
        bool[160] — Player 1 mask.
    m0 : jax.Array
        bool[160] — Missile 0 mask.
    m1 : jax.Array
        bool[160] — Missile 1 mask.
    bl : jax.Array
        bool[160] — Ball mask.
    pf : jax.Array
        bool[160] — Playfield mask.

    Returns
    -------
    collisions : jax.Array
        uint16 — Updated collision word.
    """

    def _hit(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.any(a & b).astype(jnp.uint16)

    new_bits = (
        (_hit(m0, p1) << jnp.uint16(15))
        | (_hit(m0, p0) << jnp.uint16(14))
        | (_hit(m1, p1) << jnp.uint16(13))
        | (_hit(m1, p0) << jnp.uint16(12))
        | (_hit(p0, bl) << jnp.uint16(11))
        | (_hit(p0, pf) << jnp.uint16(10))
        | (_hit(p1, bl) << jnp.uint16(9))
        | (_hit(p1, pf) << jnp.uint16(8))
        | (_hit(m0, bl) << jnp.uint16(7))
        | (_hit(m0, pf) << jnp.uint16(6))
        | (_hit(m1, bl) << jnp.uint16(5))
        | (_hit(m1, pf) << jnp.uint16(4))
        | (_hit(bl, pf) << jnp.uint16(3))
        | (_hit(p0, p1) << jnp.uint16(1))
        | (_hit(m0, m1) << jnp.uint16(0))
    )
    return (collisions | new_bits).astype(jnp.uint16)


def _cx_byte(collisions: jax.Array, addr4: jax.Array) -> jax.Array:
    """
    Return the TIA collision-latch byte for a given 4-bit read address.

    Parameters
    ----------
    collisions : jax.Array
        uint16 — Packed collision word.
    addr4 : jax.Array
        int32 — 4-bit TIA read address (0x00–0x07).

    Returns
    -------
    byte : jax.Array
        uint8 — Bits 7 and 6 set according to the two relevant collision flags.
    """
    # Each CX register packs two collision flags into bits 7 and 6:
    #   0x00 CXM0P : bit15 M0∩P1 → b7, bit14 M0∩P0 → b6
    #   0x01 CXM1P : bit13 M1∩P1 → b7, bit12 M1∩P0 → b6
    #   0x02 CXP0FB: bit11 P0∩BL → b7, bit10 P0∩PF → b6
    #   0x03 CXP1FB: bit9  P1∩BL → b7, bit8  P1∩PF → b6
    #   0x04 CXM0FB: bit7  M0∩BL → b7, bit6  M0∩PF → b6
    #   0x05 CXM1FB: bit5  M1∩BL → b7, bit4  M1∩PF → b6
    #   0x06 CXBLPF: bit3  BL∩PF → b7, 0 → b6
    #   0x07 CXPPMM: bit1  P0∩P1 → b7, bit0 M0∩M1 → b6
    col = collisions.astype(jnp.int32)
    hi_shifts = jnp.array([15, 13, 11, 9, 7, 5, 3, 1], dtype=jnp.int32)
    lo_shifts = jnp.array([14, 12, 10, 8, 6, 4, 0, 0], dtype=jnp.int32)
    hi_shift = hi_shifts[addr4]
    lo_shift = lo_shifts[addr4]
    hi_bit = ((col >> hi_shift) & 1).astype(jnp.uint8)
    lo_bit = jnp.where(
        addr4 == jnp.int32(6),
        jnp.uint8(0),
        ((col >> lo_shift) & 1).astype(jnp.uint8),
    )
    return (hi_bit << jnp.uint8(7)) | (lo_bit << jnp.uint8(6))


def tia_read(state: AtariState, addr13: chex.Array) -> chex.Array:
    """
    Read one byte from the TIA address space.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address.

    Returns
    -------
    byte : jax.Array
        uint8 — Collision latch, INPT, or 0x00 for write-only registers.
    """
    addr4 = (addr13 & jnp.int32(0x0F)).astype(jnp.int32)
    is_cx = addr4 <= jnp.int32(0x07)
    fire_byte = jax.lax.select(state.tia.fire, jnp.uint8(0x00), jnp.uint8(0x80))
    cx_val = _cx_byte(state.tia.collisions, addr4).astype(jnp.uint8)
    result = jax.lax.select(is_cx, cx_val, jnp.uint8(0))
    result = jax.lax.select(addr4 == jnp.int32(0x0C), fire_byte, result)
    result = jax.lax.select(addr4 == jnp.int32(0x0D), fire_byte, result)
    return result


def tia_write(state: AtariState, addr13: chex.Array, value: chex.Array) -> AtariState:
    """
    Write one byte to the TIA, shadowing the value and triggering side-effects.

    Handles WSYNC (0x02), RESP0–RESBL (0x10–0x14), HMOVE (0x2A),
    HMCLR (0x2B), and CXCLR (0x2C).  All writes are also shadowed into
    `tia.regs[addr & 0x3F]`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address.
    value : jax.Array
        uint8 — Value to write.

    Returns
    -------
    state : AtariState
        Updated machine state.
    """
    reg_idx = (addr13 & jnp.int32(0x3F)).astype(jnp.int32)
    val8 = value.astype(jnp.uint8)

    # Shadow into regs
    new_regs = state.tia.regs.at[reg_idx].set(val8)
    tia = state.tia.__replace__(regs=new_regs)

    # WSYNC (0x02) — stall CPU until end of scanline
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x02),
        lambda t: t.__replace__(wsync=jnp.bool_(True)),
        lambda t: t,
        tia,
    )

    # RESP0 (0x10)
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x10),
        lambda t: t.__replace__(p0_pos=_hpos_to_pixel(t.hpos)),
        lambda t: t,
        tia,
    )

    # RESP1 (0x11)
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x11),
        lambda t: t.__replace__(p1_pos=_hpos_to_pixel(t.hpos)),
        lambda t: t,
        tia,
    )

    # RESM0 (0x12)
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x12),
        lambda t: t.__replace__(m0_pos=_hpos_to_pixel(t.hpos)),
        lambda t: t,
        tia,
    )

    # RESM1 (0x13)
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x13),
        lambda t: t.__replace__(m1_pos=_hpos_to_pixel(t.hpos)),
        lambda t: t,
        tia,
    )

    # RESBL (0x14)
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x14),
        lambda t: t.__replace__(bl_pos=_hpos_to_pixel(t.hpos)),
        lambda t: t,
        tia,
    )

    # HMOVE (0x2A) — apply horizontal motion to all sprites
    def _apply_hmove(t):
        def _delta(hm_reg_idx: int) -> jax.Array:
            nibble = (t.regs[hm_reg_idx] >> jnp.uint8(4)).astype(jnp.int32)
            return jnp.where(nibble > jnp.int32(7), nibble - jnp.int32(16), nibble)

        p0 = (t.p0_pos.astype(jnp.int32) - _delta(0x20)) % jnp.int32(160)
        p1 = (t.p1_pos.astype(jnp.int32) - _delta(0x21)) % jnp.int32(160)
        m0 = (t.m0_pos.astype(jnp.int32) - _delta(0x22)) % jnp.int32(160)
        m1 = (t.m1_pos.astype(jnp.int32) - _delta(0x23)) % jnp.int32(160)
        bl = (t.bl_pos.astype(jnp.int32) - _delta(0x24)) % jnp.int32(160)
        return t.__replace__(
            p0_pos=p0.astype(jnp.uint8),
            p1_pos=p1.astype(jnp.uint8),
            m0_pos=m0.astype(jnp.uint8),
            m1_pos=m1.astype(jnp.uint8),
            bl_pos=bl.astype(jnp.uint8),
        )

    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x2A),
        _apply_hmove,
        lambda t: t,
        tia,
    )

    # HMCLR (0x2B) — zero HMP0/HMP1/HMM0/HMM1/HMBL (regs 0x20–0x24)
    def _hmclr(t):
        r = t.regs
        for i in range(0x20, 0x25):
            r = r.at[i].set(jnp.uint8(0))
        return t.__replace__(regs=r)

    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x2B),
        _hmclr,
        lambda t: t,
        tia,
    )

    # CXCLR (0x2C) — clear all collision latches
    tia = jax.lax.cond(
        reg_idx == jnp.int32(0x2C),
        lambda t: t.__replace__(collisions=jnp.uint16(0)),
        lambda t: t,
        tia,
    )

    return state.__replace__(tia=tia)


def render_scanline(state: AtariState) -> Tuple[AtariState, chex.Array]:
    """
    Render one scanline and update collision latches.

    Parameters
    ----------
    state : AtariState
        Current machine state; `tia.regs` supplies all TIA register values.

    Returns
    -------
    state : AtariState
        Updated state with collision bits OR'd into `tia.collisions`.
    pixels : jax.Array
        uint8[160, 3] — RGB scanline.
    """
    tia = state.tia
    regs = tia.regs
    px = jnp.arange(160, dtype=jnp.int32)

    # Playfield
    pf_mask = _build_pf_mask(regs[0x0D], regs[0x0E], regs[0x0F], regs[0x0A])

    # Players
    refp0 = ((regs[0x0B] >> jnp.uint8(3)) & jnp.uint8(1)).astype(jnp.bool_)
    refp1 = ((regs[0x0C] >> jnp.uint8(3)) & jnp.uint8(1)).astype(jnp.bool_)
    p0_mask = _player_mask(regs[0x1B], tia.p0_pos, refp0, regs[0x04], px)
    p1_mask = _player_mask(regs[0x1C], tia.p1_pos, refp1, regs[0x05], px)

    # Missiles
    enam0 = ((regs[0x1D] >> jnp.uint8(1)) & jnp.uint8(1)).astype(jnp.bool_)
    enam1 = ((regs[0x1E] >> jnp.uint8(1)) & jnp.uint8(1)).astype(jnp.bool_)
    m0_mask = _missile_mask(enam0, tia.m0_pos, regs[0x04], px)
    m1_mask = _missile_mask(enam1, tia.m1_pos, regs[0x05], px)

    # Ball
    enabl = ((regs[0x1F] >> jnp.uint8(1)) & jnp.uint8(1)).astype(jnp.bool_)
    bl_mask = _ball_mask(enabl, tia.bl_pos, regs[0x0A], px)

    # Colour registers
    colup0 = regs[0x06]
    colup1 = regs[0x07]
    colupf = regs[0x08]
    colubk = regs[0x09]

    ctrlpf = regs[0x0A]
    pf_priority = ((ctrlpf >> jnp.uint8(2)) & jnp.uint8(1)).astype(jnp.bool_)
    score_mode = ((ctrlpf >> jnp.uint8(1)) & jnp.uint8(1)).astype(jnp.bool_)

    # Score mode: left half PF uses COLUP0, right half uses COLUP1
    pf_color = jnp.where(
        score_mode,
        jnp.where(px < jnp.int32(80), colup0, colup1),
        colupf,
    )

    # Default priority (sprites above PF)
    colors = jnp.full(160, colubk, dtype=jnp.uint8)
    colors = jnp.where(pf_mask, pf_color, colors)
    colors = jnp.where(m1_mask, colup1, colors)
    colors = jnp.where(m0_mask, colup0, colors)
    colors = jnp.where(bl_mask, colupf, colors)
    colors = jnp.where(p1_mask, colup1, colors)
    colors = jnp.where(p0_mask, colup0, colors)

    # PF-priority mode (PF above sprites)
    colors_pfpri = jnp.full(160, colubk, dtype=jnp.uint8)
    colors_pfpri = jnp.where(m1_mask, colup1, colors_pfpri)
    colors_pfpri = jnp.where(m0_mask, colup0, colors_pfpri)
    colors_pfpri = jnp.where(bl_mask, colupf, colors_pfpri)
    colors_pfpri = jnp.where(p1_mask, colup1, colors_pfpri)
    colors_pfpri = jnp.where(p0_mask, colup0, colors_pfpri)
    colors_pfpri = jnp.where(pf_mask, pf_color, colors_pfpri)

    colors = jnp.where(pf_priority, colors_pfpri, colors)

    # NTSC look-up: 7-bit index from bits 7:1
    col_idx = ((colors >> jnp.uint8(1)) & jnp.uint8(0x7F)).astype(jnp.int32)
    rgb_row = _NTSC_PALETTE[col_idx]  # uint8[160, 3]

    # Update collision latches (OR-only, cleared only by CXCLR)
    new_collisions = _update_collisions(
        tia.collisions, p0_mask, p1_mask, m0_mask, m1_mask, bl_mask, pf_mask
    )
    new_state = state.__replace__(tia=tia.__replace__(collisions=new_collisions))

    return new_state, rgb_row
