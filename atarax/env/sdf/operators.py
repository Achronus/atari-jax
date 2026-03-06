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

"""SDF combination operators.

SDFs compose algebraically — these operators build complex shapes from primitives
without any loops or conditional logic. All functions are `jit`- and `vmap`-safe.
"""

import chex
import jax.numpy as jnp


def sdf_union(a: chex.Array, b: chex.Array) -> chex.Array:
    """
    Boolean OR — region covered by A or B.

    Parameters
    ----------
    a : chex.Array
        (H, W) float32 — first distance field.
    b : chex.Array
        (H, W) float32 — second distance field.

    Returns
    -------
    result : chex.Array
        (H, W) float32 — combined distance field (negative inside A or B).

    Example
    -------
    >>> head = sdf_circle(cx, cy - 5, r=3)
    >>> body = sdf_rect(cx, cy + 2, hw=2.5, hh=5)
    >>> humanoid = sdf_union(head, body)
    """
    return jnp.minimum(a, b)


def sdf_intersect(a: chex.Array, b: chex.Array) -> chex.Array:
    """
    Boolean AND — region covered by both A and B.

    Parameters
    ----------
    a : chex.Array
        (H, W) float32 — first distance field.
    b : chex.Array
        (H, W) float32 — second distance field.

    Returns
    -------
    result : chex.Array
        (H, W) float32 — negative only where both are inside.

    Example
    -------
    >>> shape = sdf_circle(cx, cy, r=20)
    >>> viewport = sdf_rect(80, 105, hw=80, hh=105)
    >>> clipped = sdf_intersect(shape, viewport)
    """
    return jnp.maximum(a, b)


def sdf_subtract(a: chex.Array, b: chex.Array) -> chex.Array:
    """
    Boolean MINUS — region in A but not B.

    Parameters
    ----------
    a : chex.Array
        (H, W) float32 — base shape distance field.
    b : chex.Array
        (H, W) float32 — shape to subtract distance field.

    Returns
    -------
    result : chex.Array
        (H, W) float32 — A with B hollowed out.

    Example
    -------
    >>> pacman = sdf_subtract(sdf_circle(cx, cy, r=6), wedge_sdf)
    """
    return jnp.maximum(a, -b)


def sdf_smooth_union(a: chex.Array, b: chex.Array, k: float = 4.0) -> chex.Array:
    """
    Organic smooth blend between two shapes.

    Produces a rounded merge at the boundary — useful for organic or fluid shapes.

    Parameters
    ----------
    a : chex.Array
        (H, W) float32 — first distance field.
    b : chex.Array
        (H, W) float32 — second distance field.
    k : float
        Blend radius in world pixels. Larger values produce wider blending zones.
        Default `4.0` gives a noticeable but subtle blend.

    Returns
    -------
    result : chex.Array
        (H, W) float32 — smoothly blended distance field.

    Example
    -------
    >>> blob = sdf_smooth_union(sdf_circle(40, 80, 10), sdf_circle(55, 80, 8), k=6.0)
    """
    h = jnp.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return a * (1.0 - h) + b * h - k * h * (1.0 - h)
