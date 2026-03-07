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

"""Template 7 — Traversal / Racing base classes.

Games in this template (Freeway, Road Runner, Skiing, …):

- Player navigates across or through a field of moving obstacles
- Obstacles are organised into lanes, each with a signed speed
- Collision pushes the player back or ends the episode
- Progress is measured by crossings, gates passed, or distance covered

Concrete games inherit `TraversalState` to add game-specific fields,
and `TraversalGame` to gain shared obstacle-movement and collision helpers.
"""

import chex
import jax.numpy as jnp

from atarax.game import AtaraxGame
from atarax.state import AtariState


@chex.dataclass
class TraversalState(AtariState):
    """
    Shared state for Template 7 Traversal / Racing games.

    Inherits `reward`, `done`, `step`, `episode_step`, `lives`,
    `score`, `level`, and `key` from `~atarax.state.AtariState`.

    Concrete game states (e.g. `FreewayState`) inherit from this class
    and add any game-specific fields on top.

    Parameters
    ----------
    player_x : chex.Array
        float32 scalar — player horizontal position (fixed for Freeway).
    player_y : chex.Array
        float32 scalar — player vertical position.
    jump_vy : chex.Array
        float32 scalar — vertical velocity impulse; used for jumps or
        pushback depending on the game (0.0 if unused).
    obstacles : chex.Array
        `(LANES, MAX_PER_LANE, 2)` float32 — per-lane obstacle pool.
        Axis -1 layout: `[x, active]`.
    obstacle_speed : chex.Array
        `(LANES,)` float32 — signed per-lane speed (positive = right).
    crossings : chex.Array
        int32 scalar — successful crossings / goals reached so far.
    timer : chex.Array
        int32 scalar — countdown timer; 0 means expired (used by Freeway).
    """

    player_x: chex.Array
    player_y: chex.Array
    jump_vy: chex.Array
    obstacles: chex.Array
    obstacle_speed: chex.Array
    crossings: chex.Array
    timer: chex.Array


class TraversalGame(AtaraxGame):
    """
    Abstract base class for Template 7 Traversal / Racing games.

    Provides shared, branch-free helpers for lane-based obstacle movement,
    toroidal x-wrapping, and player–obstacle AABB collision detection.
    Concrete games inherit this class and implement `_reset`, `_step`,
    and `render`.
    """

    def _move_obstacles(
        self,
        obstacles: chex.Array,
        speeds: chex.Array,
        world_w: float,
        obj_hw: float,
    ) -> chex.Array:
        """
        Advance all obstacles by their per-lane speed and wrap toroidally.

        The wrap range is `[-obj_hw, world_w + obj_hw]` so objects smoothly
        re-enter from the opposite edge once fully off-screen.

        Parameters
        ----------
        obstacles : chex.Array
            `(LANES, N, 2)` float32 — `[x, active]` per obstacle.
        speeds : chex.Array
            `(LANES,)` float32 — signed per-lane speed.
        world_w : float
            World width in pixels.
        obj_hw : float
            Object half-width; determines off-screen margin before wrapping.

        Returns
        -------
        new_obstacles : chex.Array
            Updated obstacle array with x positions advanced and wrapped.
        """
        x = obstacles[..., 0]  # (LANES, N)
        a = obstacles[..., 1]  # (LANES, N)

        # Advance x by per-lane speed (broadcast speeds over N dimension)
        new_x = x + speeds[:, None]

        # Toroidal wrap: shift into [0, world_w + 2*obj_hw], mod, shift back
        wrap_range = jnp.float32(world_w + 2.0 * obj_hw)
        new_x = ((new_x + jnp.float32(obj_hw)) % wrap_range) - jnp.float32(obj_hw)

        return jnp.stack([new_x, a], axis=-1)

    def _wrap_x(self, x: chex.Array, world_w: float) -> chex.Array:
        """
        Wrap a single x coordinate into `[0, world_w)`.

        Parameters
        ----------
        x : chex.Array
            float32 scalar or array.
        world_w : float
            World width in pixels.

        Returns
        -------
        wrapped_x : chex.Array
            x modulo world_w, always in `[0, world_w)`.
        """
        return x % jnp.float32(world_w)

    def _player_obstacle_hit(
        self,
        player_x: chex.Array,
        player_y: chex.Array,
        player_hw: float,
        player_hh: float,
        obstacle_xs: chex.Array,
        obstacle_ys: chex.Array,
        obstacle_active: chex.Array,
        obj_hw: float,
        obj_hh: float,
    ) -> chex.Array:
        """
        AABB collision between the player and a pool of obstacles.

        Parameters
        ----------
        player_x, player_y : chex.Array
            float32 scalars — player centre.
        player_hw, player_hh : float
            Player half-dimensions.
        obstacle_xs : chex.Array
            `(N,)` float32 — obstacle x centres.
        obstacle_ys : chex.Array
            `(N,)` float32 — obstacle y centres (lane centre y values).
        obstacle_active : chex.Array
            `(N,)` bool — active flag per obstacle.
        obj_hw, obj_hh : float
            Obstacle half-dimensions.

        Returns
        -------
        hit_mask : chex.Array
            `(N,)` bool — True where the player overlaps an active obstacle.
        """
        dx = jnp.abs(player_x - obstacle_xs) < jnp.float32(player_hw + obj_hw)
        dy = jnp.abs(player_y - obstacle_ys) < jnp.float32(player_hh + obj_hh)
        return dx & dy & obstacle_active
