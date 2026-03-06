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

"""SDF rendering utilities for JAX Atari environments.

Compositing pipeline:

1. `make_canvas` — create blank `(210, 160, 3)` float32 frame
2. `paint_layer` / `paint_sdf` / `paint_soft` — stamp layers
3. Entity pool helpers — `render_rect_pool`, `render_circle_pool`,
   `render_variable_circle_pool`, `render_bool_grid`
4. `finalise_rgb` — convert float32 → uint8

All functions are `jit`- and `vmap`-compatible.
Outputs: `(210, 160, 3) uint8` RGB at ALE native resolution.
"""

import chex
import jax.numpy as jnp

from atarax.env.sdf.primitives import _PX, _PY


def make_canvas(bg_colour: chex.Array) -> chex.Array:
    """
    Create a blank `(210, 160, 3)` float32 canvas filled with `bg_colour`.

    Parameters
    ----------
    bg_colour : chex.Array
        (3,) float32 RGB in `[0, 1]`.

    Returns
    -------
    canvas : chex.Array
        (210, 160, 3) float32 — blank canvas at the given background colour.
    """
    return jnp.ones((210, 160, 3), dtype=jnp.float32) * bg_colour[None, None, :]


def finalise_rgb(canvas: chex.Array) -> chex.Array:
    """
    Convert a float32 canvas to `uint8` RGB output.

    Clips values to `[0, 1]`, scales to `[0, 255]`, and casts to `uint8`.
    This is the final step in every `render()` function.

    Parameters
    ----------
    canvas : chex.Array
        (210, 160, 3) float32 in `[0, 1]`.

    Returns
    -------
    frame : chex.Array
        (210, 160, 3) uint8 — ALE-resolution RGB frame.
    """
    return jnp.clip(canvas * 255.0, 0.0, 255.0).astype(jnp.uint8)


def paint_layer(
    canvas: chex.Array,
    mask: chex.Array,
    colour: chex.Array,
) -> chex.Array:
    """
    Paint a boolean mask onto the canvas with the given colour.

    Uses painter's algorithm — overwrites existing pixels wherever `mask` is
    `True`. Layers should be composited back-to-front.

    Parameters
    ----------
    canvas : chex.Array
        (210, 160, 3) float32 — current canvas state.
    mask : chex.Array
        (210, 160) bool — pixels to paint.
    colour : chex.Array
        (3,) float32 RGB in `[0, 1]`.

    Returns
    -------
    canvas : chex.Array
        (210, 160, 3) float32 — updated canvas.
    """
    m = mask[:, :, None]
    return jnp.where(m, colour[None, None, :], canvas)


def paint_sdf(
    canvas: chex.Array,
    sdf: chex.Array,
    colour: chex.Array,
) -> chex.Array:
    """
    Paint wherever `sdf < 0` (hard-edge threshold).

    Convenience wrapper around :func:`paint_layer` for direct SDF compositing.

    Parameters
    ----------
    canvas : chex.Array
        (210, 160, 3) float32.
    sdf : chex.Array
        (210, 160) float32 — signed distance field.
    colour : chex.Array
        (3,) float32 RGB in `[0, 1]`.

    Returns
    -------
    canvas : chex.Array
        (210, 160, 3) float32 — updated canvas.
    """
    return paint_layer(canvas, sdf < 0.0, colour)


def paint_soft(
    canvas: chex.Array,
    sdf: chex.Array,
    colour: chex.Array,
) -> chex.Array:
    """
    Paint with sub-pixel anti-aliasing — smooth transition over ±0.5 world pixels.

    Encodes fractional position information as a continuous alpha gradient,
    useful for CNN training (entity sub-pixel position becomes differentiable).

    Parameters
    ----------
    canvas : chex.Array
        (210, 160, 3) float32.
    sdf : chex.Array
        (210, 160) float32 — signed distance field.
    colour : chex.Array
        (3,) float32 RGB in `[0, 1]`.

    Returns
    -------
    canvas : chex.Array
        (210, 160, 3) float32 — updated canvas with soft-edge entity.
    """
    alpha = jnp.clip(0.5 - sdf, 0.0, 1.0)[:, :, None]  # (210, 160, 1)
    c = colour[None, None, :]  # (1, 1, 3)
    return canvas * (1.0 - alpha) + c * alpha


def render_rect_pool(
    pool: chex.Array,
    hw: float,
    hh: float,
) -> chex.Array:
    """
    Render all active entities in a pool as axis-aligned rectangles.

    Uses an `(H, W, N)` broadcast — all entities are rendered simultaneously
    without any Python-level loop. `jit`- and `vmap`-safe.

    Parameters
    ----------
    pool : chex.Array
        (N, 3) float32 — entity table with columns `[x, y, active]`.
        `active` must be `1.0` (alive) or `0.0` (dead).
    hw : float
        Half-width of each rectangle in world pixels.
    hh : float
        Half-height of each rectangle in world pixels.

    Returns
    -------
    mask : chex.Array
        (210, 160) bool — pixel mask; `True` where at least one live entity covers.
    """
    ex = pool[:, 0]  # (N,) — entity x centres
    ey = pool[:, 1]  # (N,) — entity y centres
    ea = pool[:, 2]  # (N,) — active flags

    # _PX: (1, 160) → with None → (1, 160, 1) broadcasts vs (N,) → (1, 160, N)
    # _PY: (210, 1) → with None → (210, 1, 1) broadcasts vs (N,) → (210, 1, N)
    # (1, 160, N) & (210, 1, N) → (210, 160, N)
    dx = jnp.abs(_PX[:, :, None] - ex)  # (1, 160, N)
    dy = jnp.abs(_PY[:, :, None] - ey)  # (210, 1, N)

    inside = (dx < hw) & (dy < hh)  # (210, 160, N)
    active = ea[None, None, :]  # (1, 1, N)
    return jnp.any(inside & (active > 0.0), axis=-1)  # (210, 160)


def render_circle_pool(
    pool: chex.Array,
    radius: float,
) -> chex.Array:
    """
    Render all active entities in a pool as circles of uniform radius.

    Parameters
    ----------
    pool : chex.Array
        (N, 3) float32 — entity table with columns `[x, y, active]`.
    radius : float
        World-space radius for all entities.

    Returns
    -------
    mask : chex.Array
        (210, 160) bool — pixel mask.
    """
    ex = pool[:, 0]
    ey = pool[:, 1]
    ea = pool[:, 2]

    dx = _PX[:, :, None] - ex  # (1, 160, N)
    dy = _PY[:, :, None] - ey  # (210, 1, N)

    inside = (dx**2 + dy**2) < radius**2
    active = ea[None, None, :]
    return jnp.any(inside & (active > 0.0), axis=-1)


def render_variable_circle_pool(pool: chex.Array) -> chex.Array:
    """
    Render circles with per-entity radii.

    Used for Asteroids rocks (three sizes) and explosion effects.

    Parameters
    ----------
    pool : chex.Array
        (N, 4) float32 — entity table with columns `[x, y, radius, active]`.

    Returns
    -------
    mask : chex.Array
        (210, 160) bool — pixel mask.
    """
    ex = pool[:, 0]
    ey = pool[:, 1]
    er = pool[:, 2]
    ea = pool[:, 3]

    dx = _PX[:, :, None] - ex  # (1, 160, N)
    dy = _PY[:, :, None] - ey  # (210, 1, N)
    dist_sq = dx**2 + dy**2
    r_sq = (er**2)[None, None, :]
    inside = dist_sq < r_sq
    active = ea[None, None, :]
    return jnp.any(inside & (active > 0.0), axis=-1)


def render_bool_grid(
    grid: chex.Array,
    cell_x0: float,
    cell_y0: float,
    cell_w: float,
    cell_h: float,
    *,
    draw_w: float | None = None,
    draw_h: float | None = None,
) -> chex.Array:
    """
    Render a boolean occupancy grid as a field of same-sized rectangles.

    Efficient for sparse grids (many `False` entries) such as Breakout bricks,
    Space Invaders shields, or Pac-Man pellets. Internally builds a pool from
    active cell centres and delegates to :func:`render_rect_pool`.

    Parameters
    ----------
    grid : chex.Array
        (rows, cols) bool — `True` = cell is alive/active.
    cell_x0 : float
        World x coordinate of the grid's left edge.
    cell_y0 : float
        World y coordinate of the grid's top edge.
    cell_w : float
        Horizontal spacing between cell centres in world pixels.
    cell_h : float
        Vertical spacing between cell centres in world pixels.
    draw_w : float | None (optional)
        Width of each drawn rectangle in world pixels.
        Defaults to `cell_w * 0.96` (nearly fills the full cell).
        Set smaller than `cell_w` to draw small markers (e.g. Pac-Man dots)
        centred in larger tile cells.
    draw_h : float | None (optional)
        Height of each drawn rectangle in world pixels.
        Defaults to `cell_h * 0.96`.

    Returns
    -------
    mask : chex.Array
        (210, 160) bool — pixel mask for all live cells.
    """
    rows, cols = grid.shape
    col_idx = jnp.arange(cols, dtype=jnp.float32)
    row_idx = jnp.arange(rows, dtype=jnp.float32)
    cx = cell_x0 + (col_idx + 0.5) * cell_w  # (cols,)
    cy = cell_y0 + (row_idx + 0.5) * cell_h  # (rows,)
    all_cx = jnp.tile(cx, rows)  # (rows*cols,)
    all_cy = jnp.repeat(cy, cols)  # (rows*cols,)
    active = grid.ravel().astype(jnp.float32)
    pool = jnp.stack([all_cx, all_cy, active], axis=1)  # (rows*cols, 3)
    hw = (draw_w if draw_w is not None else cell_w * 0.96) / 2.0
    hh = (draw_h if draw_h is not None else cell_h * 0.96) / 2.0
    return render_rect_pool(pool, hw=hw, hh=hh)
