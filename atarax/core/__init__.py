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

"""atarax.core — shared JAX-native physics primitives."""

from atarax.core.mechanics import (
    aabb_overlap,
    clamp,
    grid_hit_test,
    reflect_axis,
    wrap_position,
)

__all__ = [
    "aabb_overlap",
    "clamp",
    "grid_hit_test",
    "reflect_axis",
    "wrap_position",
]
