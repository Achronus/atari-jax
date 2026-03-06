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

"""Shared HUD rendering utilities for all games.

Provides:

- 7-segment score digit rendering (6 digits, top-centre of the 210×160 frame)
- Life pip rendering with per-game SDF pip functions

Usage (in each game's `render()` method):

    from atarax.env.hud import render_score, render_life_pips, HUD_PIP_XS, HUD_PIP_Y
    from atarax.env.sdf import sdf_circle

    canvas = render_score(canvas, state.score, colour=_MY_SCORE_COLOUR)
    canvas = render_life_pips(
        canvas, state.lives,
        pip_xs=HUD_PIP_XS,
        pip_y=HUD_PIP_Y,
        pip_sdf_fn=lambda cx, cy: sdf_circle(cx, cy, 3.5),
        pip_colour=_MY_COLOUR,
    )
"""

from typing import Callable

import chex
import jax.numpy as jnp
import numpy as np

from atarax.env.sdf import paint_layer

# ── 7-segment digit layout ──────────────────────────────────────────────────
# Each digit occupies a 5×7 px cell.
# Segments: top(0) top-left(1) top-right(2) middle(3) bottom-left(4) bottom-right(5) bottom(6)
# _SEG_RECTS_PY: (ox, oy, w, h) relative to digit top-left corner.
_SEG_RECTS_PY: list[tuple[int, int, int, int]] = [
    (1, 0, 3, 1),  # 0 top
    (0, 1, 1, 2),  # 1 top-left
    (4, 1, 1, 2),  # 2 top-right
    (1, 3, 3, 1),  # 3 middle
    (0, 4, 1, 2),  # 4 bottom-left
    (4, 4, 1, 2),  # 5 bottom-right
    (1, 6, 3, 1),  # 6 bottom
]

# Which of the 7 segments are ON for digits 0–9.
DIGIT_SEGS = jnp.array(
    [
        [1, 1, 1, 0, 1, 1, 1],  # 0
        [0, 0, 1, 0, 0, 1, 0],  # 1
        [1, 0, 1, 1, 1, 0, 1],  # 2
        [1, 0, 1, 1, 0, 1, 1],  # 3
        [0, 1, 1, 1, 0, 1, 0],  # 4
        [1, 1, 0, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 1, 1],  # 6
        [1, 0, 1, 0, 0, 1, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ],
    dtype=jnp.bool_,
)


def _make_seg_masks(bx: int, by: int) -> np.ndarray:
    """Return (7, 210, 160) bool array: per-segment pixel masks for a digit at (bx, by)."""
    out = np.zeros((7, 210, 160), dtype=bool)
    for i, (ox, oy, w, h) in enumerate(_SEG_RECTS_PY):
        out[i, by + oy : by + oy + h, bx + ox : bx + ox + w] = True
    return out


# ── Default score layout — 6 digits centred in the HUD strip (y=0..29) ─────
# Each digit: 5 px wide, 7 px tall.  Digits are 6 px apart (5 px + 1 px gap).
# 6 digits × 6 px = 36 px; starting at x=62 centres them on the 160 px width.
HUD_SCORE_Y: int = 11
HUD_SCORE_XS: list[int] = [62 + di * 6 for di in range(6)]
# Shape: (6, 7, 210, 160) bool — one (7, 210, 160) segment mask per digit position.
# Stored as a JAX array so it lives on the target device and avoids
# CPU→device transfers inside render_score().
SCORE_SEG_MASKS: chex.Array = jnp.array(
    np.stack([_make_seg_masks(x, HUD_SCORE_Y) for x in HUD_SCORE_XS])
)

# ── Default pip layout — 3 pips, left side of HUD strip ────────────────────
HUD_PIP_XS: list[float] = [8.0, 16.0, 24.0]
HUD_PIP_Y: float = 15.0

# ── Default score colour (cream, matches ALE palette) ──────────────────────
_COL_SCORE_DEFAULT = jnp.array([1.0, 0.902, 0.725], dtype=jnp.float32)


def render_score(
    canvas: chex.Array,
    score: chex.Array,
    colour: chex.Array | None = None,
    seg_masks: chex.Array | None = None,
) -> chex.Array:
    """Render a 6-digit 7-segment score display onto *canvas*.

    Parameters
    ----------
    canvas : chex.Array
        `(210, 160, 3)` float32 canvas to draw onto.
    score : chex.Array
        int32 scalar — current score (clamped to `[0, 999 999]`).
    colour : chex.Array (optional)
        `float32[3]` RGB colour for the digit segments.
        Default is `None` (cream `(1.0, 0.902, 0.725)`).
    seg_masks : chex.Array (optional)
        `(6, 7, 210, 160)` bool — segment pixel masks for each digit position.
        Default is `None` (uses `SCORE_SEG_MASKS`, centred at HUD `y=11`).

    Returns
    -------
    canvas : chex.Array
        Updated canvas with the score rendered.
    """
    if colour is None:
        colour = _COL_SCORE_DEFAULT
    if seg_masks is None:
        seg_masks = SCORE_SEG_MASKS

    score_clamped = jnp.clip(score, jnp.int32(0), jnp.int32(999999))
    divisors = [100000, 10000, 1000, 100, 10, 1]
    for di, div in enumerate(divisors):
        digit_val = (score_clamped // jnp.int32(div)) % jnp.int32(10)
        segs = DIGIT_SEGS[digit_val]  # (7,) bool
        seg_px = seg_masks[di]  # (7, 210, 160) — already a JAX array
        digit_mask = jnp.any(seg_px & segs[:, None, None], axis=0)
        canvas = paint_layer(canvas, digit_mask, colour)
    return canvas


def render_life_pips(
    canvas: chex.Array,
    lives: chex.Array,
    pip_sdf_fn: Callable[[float, float], chex.Array],
    pip_colour: chex.Array,
    pip_xs: list[float] | None = None,
    pip_y: float | None = None,
) -> chex.Array:
    """Render life pip icons onto *canvas*.

    Each pip is shown only when the player has more than `i` lives
    remaining (pip 0 requires `lives > 0`, etc.).

    Parameters
    ----------
    canvas : chex.Array
        `(210, 160, 3)` float32 canvas to draw onto.
    lives : chex.Array
        int32 scalar — remaining lives.
    pip_sdf_fn : callable
        `(cx: float, cy: float) → (210, 160) float32` SDF array.
        Negative values are inside the shape.  Use any SDF primitive
        (`sdf_circle`, `sdf_ship_triangle`, `sdf_rect`, …).
    pip_colour : chex.Array
        `float32[3]` RGB colour for the pips.
    pip_xs : list of float, optional
        X centres for each pip position.  Defaults to `HUD_PIP_XS`
        `[8.0, 16.0, 24.0]`.
    pip_y : float, optional
        Y centre for all pips.  Defaults to `HUD_PIP_Y` (`15.0`).

    Returns
    -------
    canvas : chex.Array
        Updated canvas with up to `len(pip_xs)` pip icons rendered.
    """
    if pip_xs is None:
        pip_xs = HUD_PIP_XS
    if pip_y is None:
        pip_y = HUD_PIP_Y

    for li, cx in enumerate(pip_xs):
        pip_active = lives > jnp.int32(li)
        sdf = pip_sdf_fn(cx, pip_y)
        pip_mask = (sdf < jnp.float32(0.0)) & pip_active
        canvas = paint_layer(canvas, pip_mask, pip_colour)
    return canvas
