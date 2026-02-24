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

"""Shared JAX-native physics primitives.

All functions are pure JAX — no Python-level branching on dynamic values.
Every conditional uses `jnp.where` so the functions stay branch-free and
compile cleanly under `jax.jit`, `jax.vmap`, and `jax.lax.scan`.
"""

from typing import Tuple

import chex
import jax.numpy as jnp


def aabb_overlap(
    pos_a: chex.Array,
    size_a: chex.Array,
    pos_b: chex.Array,
    size_b: chex.Array,
) -> chex.Array:
    """
    Test whether two axis-aligned bounding boxes overlap.

    Parameters
    ----------
    pos_a : chex.Array
        float32[2] — (x, y) top-left corner of box A.
    size_a : chex.Array
        float32[2] — (width, height) of box A.
    pos_b : chex.Array
        float32[2] — (x, y) top-left corner of box B.
    size_b : chex.Array
        float32[2] — (width, height) of box B.

    Returns
    -------
    overlap : chex.Array
        bool — `True` when the two boxes overlap.
    """
    return (
        (pos_a[0] < pos_b[0] + size_b[0])
        & (pos_a[0] + size_a[0] > pos_b[0])
        & (pos_a[1] < pos_b[1] + size_b[1])
        & (pos_a[1] + size_a[1] > pos_b[1])
    )


def reflect_axis(
    vel: chex.Array,
    hit_x: chex.Array,
    hit_y: chex.Array,
) -> chex.Array:
    """
    Reflect a 2-D velocity vector along the struck axis.

    Parameters
    ----------
    vel : chex.Array
        float32[2] — (vx, vy) velocity.
    hit_x : chex.Array
        bool — Negate the x component.
    hit_y : chex.Array
        bool — Negate the y component.

    Returns
    -------
    new_vel : chex.Array
        float32[2] — Reflected velocity.
    """
    vx = jnp.where(hit_x, -vel[0], vel[0])
    vy = jnp.where(hit_y, -vel[1], vel[1])
    return jnp.array([vx, vy], dtype=jnp.float32)


def grid_hit_test(
    ball_x: chex.Array,
    ball_y: chex.Array,
    ball_w: chex.Array,
    ball_h: chex.Array,
    grid: chex.Array,
    cell_y0: chex.Array,
    cell_x0: chex.Array,
    cell_h: chex.Array,
    cell_w: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Test a ball AABB against a 2-D boolean grid of cells.

    The ball is treated as an axis-aligned rectangle of size `ball_w` ×
    `ball_h` whose top-left corner is at (`ball_x`, `ball_y`).  Each cell
    `(row, col)` occupies the rectangle:

        x ∈ [cell_x0 + col * cell_w, cell_x0 + (col+1) * cell_w)
        y ∈ [cell_y0 + row * cell_h, cell_y0 + (row+1) * cell_h)

    Only active cells (where `grid[row, col]` is `True`) are tested.

    Parameters
    ----------
    ball_x : chex.Array
        float32 — Ball left edge.
    ball_y : chex.Array
        float32 — Ball top edge.
    ball_w : chex.Array
        float32 — Ball width in pixels.
    ball_h : chex.Array
        float32 — Ball height in pixels.
    grid : chex.Array
        bool[rows, cols] — Active cell flags.
    cell_y0 : chex.Array
        float32 — Y coordinate of the top edge of row 0.
    cell_x0 : chex.Array
        float32 — X coordinate of the left edge of column 0.
    cell_h : chex.Array
        float32 — Height of each cell in pixels.
    cell_w : chex.Array
        float32 — Width of each cell in pixels.

    Returns
    -------
    hit_mask : chex.Array
        bool[rows, cols] — `True` for every active cell the ball overlaps.
    any_hit : chex.Array
        bool — `True` when at least one cell was hit.
    """
    rows, cols = jnp.shape(grid)
    row_idx = jnp.arange(rows, dtype=jnp.float32)
    col_idx = jnp.arange(cols, dtype=jnp.float32)

    # Top-left corner of every cell, broadcast to [rows, cols]
    cy = (cell_y0 + row_idx * cell_h)[:, None]  # [rows, 1]
    cx = (cell_x0 + col_idx * cell_w)[None, :]  # [1, cols]

    hit_mask = (
        (ball_x < cx + cell_w)
        & (ball_x + ball_w > cx)
        & (ball_y < cy + cell_h)
        & (ball_y + ball_h > cy)
        & grid
    )
    return hit_mask, jnp.any(hit_mask)


def wrap_position(
    pos: chex.Array,
    lo: chex.Array,
    hi: chex.Array,
) -> chex.Array:
    """
    Wrap a scalar position into the half-open interval [lo, hi).

    Parameters
    ----------
    pos : chex.Array
        float32 — Position to wrap.
    lo : chex.Array
        float32 — Lower bound (inclusive).
    hi : chex.Array
        float32 — Upper bound (exclusive).

    Returns
    -------
    wrapped : chex.Array
        float32 — Position modulo the interval width.
    """
    span = hi - lo
    return lo + jnp.mod(pos - lo, span)


def clamp(
    val: chex.Array,
    lo: chex.Array,
    hi: chex.Array,
) -> chex.Array:
    """
    Clamp `val` to the closed interval [lo, hi].

    Parameters
    ----------
    val : chex.Array
        Value to clamp.
    lo : chex.Array
        Lower bound.
    hi : chex.Array
        Upper bound.

    Returns
    -------
    clamped : chex.Array
        Value clipped to [lo, hi].
    """
    return jnp.clip(val, lo, hi)
