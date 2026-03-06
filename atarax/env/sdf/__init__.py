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

"""SDF rendering library for JAX-native Atari environments.

Provides:

- **Primitives** — SDF shape functions
  (rect, circle, triangle, capsule, …)
- **Operators** — algebraic SDF combiners
  (union, subtract, smooth blend)
- **Rendering** — canvas management, pool
  renderers, and grid renderer
"""

from atarax.env.sdf.operators import (
    sdf_intersect,
    sdf_smooth_union,
    sdf_subtract,
    sdf_union,
)
from atarax.env.sdf.primitives import (
    _PX,
    _PY,
    sdf_capsule,
    sdf_circle,
    sdf_diamond,
    sdf_ghost,
    sdf_oriented_rect,
    sdf_rect,
    sdf_rect_topleft,
    sdf_ring,
    sdf_ship_triangle,
    sdf_triangle,
)
from atarax.env.sdf.render import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    paint_soft,
    render_bool_grid,
    render_circle_pool,
    render_rect_pool,
    render_variable_circle_pool,
)

__all__ = [
    # coordinate grids
    "_PX",
    "_PY",
    # primitives
    "sdf_capsule",
    "sdf_circle",
    "sdf_diamond",
    "sdf_ghost",
    "sdf_oriented_rect",
    "sdf_rect",
    "sdf_rect_topleft",
    "sdf_ring",
    "sdf_ship_triangle",
    "sdf_triangle",
    # operators
    "sdf_intersect",
    "sdf_smooth_union",
    "sdf_subtract",
    "sdf_union",
    # rendering
    "make_canvas",
    "finalise_rgb",
    "paint_layer",
    "paint_sdf",
    "paint_soft",
    "render_bool_grid",
    "render_circle_pool",
    "render_rect_pool",
    "render_variable_circle_pool",
]
