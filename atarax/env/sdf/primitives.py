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

"""SDF shape primitives for JAX Atari rendering.

All functions operate in world space (= ALE native pixel space: 160 × 210).
Pixel `(row, col)` maps to world coordinates `(x=col, y=row)`.

Module-level coordinate grids are built once at import time:

    _PX : (1, 160) float32 — x coordinate of each pixel column
    _PY : (210, 1) float32 — y coordinate of each pixel row

All SDF primitives accept scalar position/size parameters as `chex.Array`
(JAX scalar tensors) or Python `float` literals — JAX coerces them transparently.

Return value convention for all primitives:

    value < 0  →  inside  the shape
    value = 0  →  on      the boundary
    value > 0  →  outside the shape
"""

import chex
import jax.numpy as jnp

# Pixel coordinate grids
_PX: chex.Array = jnp.arange(160, dtype=jnp.float32)[None, :]  # (1, 160)
_PY: chex.Array = jnp.arange(210, dtype=jnp.float32)[:, None]  # (210, 1)


def sdf_rect(
    cx: chex.Array,
    cy: chex.Array,
    hw: chex.Array,
    hh: chex.Array,
) -> chex.Array:
    """
    Axis-aligned rectangle signed distance field.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x in world coordinates.
    cy : chex.Array
        float32 scalar — centre y in world coordinates.
    hw : chex.Array
        float32 scalar — half-width of the rectangle.
    hh : chex.Array
        float32 scalar — half-height of the rectangle.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32 — negative inside, positive outside.
    """
    dx = jnp.abs(_PX - cx) - hw
    dy = jnp.abs(_PY - cy) - hh
    outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
    return outside + inside


def sdf_rect_topleft(
    x0: chex.Array,
    y0: chex.Array,
    w: chex.Array,
    h: chex.Array,
) -> chex.Array:
    """
    Axis-aligned rectangle defined by its top-left corner and dimensions.

    Convenience wrapper around :func:`sdf_rect`.

    Parameters
    ----------
    x0 : chex.Array
        float32 scalar — left edge x in world coordinates.
    y0 : chex.Array
        float32 scalar — top edge y in world coordinates.
    w : chex.Array
        float32 scalar — full width.
    h : chex.Array
        float32 scalar — full height.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    return sdf_rect(cx=x0 + w / 2, cy=y0 + h / 2, hw=w / 2, hh=h / 2)


def sdf_oriented_rect(
    cx: chex.Array,
    cy: chex.Array,
    hw: chex.Array,
    hh: chex.Array,
    angle: chex.Array,
) -> chex.Array:
    """
    Rectangle rotated by `angle` radians around its centre.

    Achieved by rotating pixel coordinates into the rect's local frame before
    comparing against axis-aligned extents.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x in world coordinates.
    cy : chex.Array
        float32 scalar — centre y in world coordinates.
    hw : chex.Array
        float32 scalar — half-width in the local (rotated) frame.
    hh : chex.Array
        float32 scalar — half-height in the local (rotated) frame.
    angle : chex.Array
        float32 scalar — rotation angle in radians (CCW positive).

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    cos_a = jnp.cos(-angle)
    sin_a = jnp.sin(-angle)
    tx = _PX - cx
    ty = _PY - cy
    lx = tx * cos_a - ty * sin_a
    ly = tx * sin_a + ty * cos_a
    dx = jnp.abs(lx) - hw
    dy = jnp.abs(ly) - hh
    outside = jnp.sqrt(jnp.maximum(dx, 0.0) ** 2 + jnp.maximum(dy, 0.0) ** 2)
    inside = jnp.minimum(jnp.maximum(dx, dy), 0.0)
    return outside + inside


def sdf_circle(
    cx: chex.Array,
    cy: chex.Array,
    r: chex.Array,
) -> chex.Array:
    """
    Filled circle signed distance field.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x in world coordinates.
    cy : chex.Array
        float32 scalar — centre y in world coordinates.
    r : chex.Array
        float32 scalar — radius in world coordinates.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32 — negative inside, positive outside.
    """
    return jnp.sqrt((_PX - cx) ** 2 + (_PY - cy) ** 2) - r


def sdf_ring(
    cx: chex.Array,
    cy: chex.Array,
    r_outer: chex.Array,
    r_inner: chex.Array,
) -> chex.Array:
    """
    Hollow ring — annulus between `r_inner` and `r_outer`.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x.
    cy : chex.Array
        float32 scalar — centre y.
    r_outer : chex.Array
        float32 scalar — outer radius.
    r_inner : chex.Array
        float32 scalar — inner radius (must be < `r_outer`).

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    d = jnp.sqrt((_PX - cx) ** 2 + (_PY - cy) ** 2)
    return jnp.maximum(d - r_outer, r_inner - d)


def sdf_triangle(
    ax: chex.Array,
    ay: chex.Array,
    bx: chex.Array,
    by: chex.Array,
    cx: chex.Array,
    cy: chex.Array,
) -> chex.Array:
    """
    Arbitrary triangle defined by three vertices.

    Parameters
    ----------
    ax, ay : chex.Array
        float32 scalar — first vertex in world coordinates.
    bx, by : chex.Array
        float32 scalar — second vertex in world coordinates.
    cx, cy : chex.Array
        float32 scalar — third vertex in world coordinates.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32 — negative inside the triangle.
    """

    def _edge_dist(
        ex0: chex.Array, ey0: chex.Array, ex1: chex.Array, ey1: chex.Array
    ) -> chex.Array:
        dx_e = ex1 - ex0
        dy_e = ey1 - ey0
        dx_p = _PX - ex0
        dy_p = _PY - ey0
        t = jnp.clip(
            (dx_p * dx_e + dy_p * dy_e) / (dx_e**2 + dy_e**2 + 1e-8),
            0.0,
            1.0,
        )
        nx = dx_p - t * dx_e
        ny = dy_p - t * dy_e
        return jnp.sqrt(nx**2 + ny**2)

    def _cross(
        ex0: chex.Array, ey0: chex.Array, ex1: chex.Array, ey1: chex.Array
    ) -> chex.Array:
        return (_PX - ex0) * (ey1 - ey0) - (_PY - ey0) * (ex1 - ex0)

    d0 = _edge_dist(ax, ay, bx, by)
    d1 = _edge_dist(bx, by, cx, cy)
    d2 = _edge_dist(cx, cy, ax, ay)
    s0 = _cross(ax, ay, bx, by)
    s1 = _cross(bx, by, cx, cy)
    s2 = _cross(cx, cy, ax, ay)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    dist = jnp.minimum(jnp.minimum(d0, d1), d2)
    return jnp.where(inside, -dist, dist)


def sdf_ship_triangle(
    cx: chex.Array,
    cy: chex.Array,
    angle: chex.Array,
    size: float = 8.0,
) -> chex.Array:
    """
    Equilateral triangle pointing in the direction of `angle`.

    Convenience wrapper around :func:`sdf_triangle` for Asteroids / Gravitar ships.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x in world coordinates.
    cy : chex.Array
        float32 scalar — centre y in world coordinates.
    angle : chex.Array
        float32 scalar — heading angle in radians.
    size : float
        Distance from centre to tip in world pixels. Default `8.0`.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    tip_x = cx + jnp.cos(angle) * size
    tip_y = cy + jnp.sin(angle) * size
    lft_x = cx + jnp.cos(angle + 2.4) * size * 0.6
    lft_y = cy + jnp.sin(angle + 2.4) * size * 0.6
    rgt_x = cx + jnp.cos(angle - 2.4) * size * 0.6
    rgt_y = cy + jnp.sin(angle - 2.4) * size * 0.6
    return sdf_triangle(tip_x, tip_y, lft_x, lft_y, rgt_x, rgt_y)


def sdf_capsule(
    ax: chex.Array,
    ay: chex.Array,
    bx: chex.Array,
    by: chex.Array,
    r: chex.Array,
) -> chex.Array:
    """
    Capsule — a line segment with hemispherical rounded ends.

    Used for bullets, tentacles, and laser beams.

    Parameters
    ----------
    ax : chex.Array
        float32 scalar — start point x.
    ay : chex.Array
        float32 scalar — start point y.
    bx : chex.Array
        float32 scalar — end point x.
    by : chex.Array
        float32 scalar — end point y.
    r : chex.Array
        float32 scalar — radius of the capsule.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    dx_e = bx - ax
    dy_e = by - ay
    dx_p = _PX - ax
    dy_p = _PY - ay
    t = jnp.clip(
        (dx_p * dx_e + dy_p * dy_e) / (dx_e**2 + dy_e**2 + 1e-8),
        0.0,
        1.0,
    )
    nx = _PX - (ax + t * dx_e)
    ny = _PY - (ay + t * dy_e)
    return jnp.sqrt(nx**2 + ny**2) - r


def sdf_diamond(
    cx: chex.Array,
    cy: chex.Array,
    hw: chex.Array,
    hh: chex.Array,
) -> chex.Array:
    """
    Diamond (rhombus) signed distance field.

    Used for gems, power-up indicators, and angular projectiles.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x.
    cy : chex.Array
        float32 scalar — centre y.
    hw : chex.Array
        float32 scalar — half-extent along the x axis.
    hh : chex.Array
        float32 scalar — half-extent along the y axis.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32.
    """
    dx = jnp.abs(_PX - cx) / hw
    dy = jnp.abs(_PY - cy) / hh
    return (dx + dy - 1.0) * (hw * hh) / jnp.sqrt(hw**2 + hh**2)


def sdf_ghost(
    cx: chex.Array,
    cy: chex.Array,
    r: float = 5.0,
) -> chex.Array:
    """
    Pac-Man ghost silhouette — circle head merged with a rectangular body.

    Parameters
    ----------
    cx : chex.Array
        float32 scalar — centre x of the ghost.
    cy : chex.Array
        float32 scalar — centre y of the ghost (vertical midpoint).
    r : float
        Head radius in world pixels. Default `5.0`.

    Returns
    -------
    distance_field : chex.Array
        (210, 160) float32 — combined ghost shape SDF.
    """
    from atarax.env.sdf.operators import sdf_union

    head = sdf_circle(cx, cy - r * 0.4, r)
    body = sdf_rect(cx, cy + r * 0.6, hw=r, hh=r * 0.6)
    return sdf_union(head, body)
