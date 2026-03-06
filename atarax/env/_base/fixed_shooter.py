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

"""Template 1 — Fixed Shooter base classes.

Games in this template (Space Invaders, Phoenix, Galaxian, …):

- Player moves horizontally only
- Enemies form a grid/formation that moves as a unit
- Both sides fire bullets from fixed-size pools
- Optional destructible shields

Concrete games inherit `FixedShooterState` to add game-specific fields,
and `FixedShooterGame` to gain shared physics helpers.
"""

import chex
import jax.numpy as jnp

from atarax.game import AtaraxGame
from atarax.state import AtariState


@chex.dataclass
class FixedShooterState(AtariState):
    """
    Shared state for Template 1 Fixed Shooter games.

    Inherits `reward`, `done`, `step`, `episode_step`, `lives`,
    `score`, `level`, and `key` from `~atarax.state.AtariState`.

    Concrete game states (e.g. `SpaceInvadersState`) inherit from this class
    and add any game-specific fields on top.

    Parameters
    ----------
    player_x : chex.Array
        float32 scalar — player cannon horizontal centre `(world x)`.
    fire_cooldown : chex.Array
        int32 scalar — frames remaining until the next player shot is allowed.
    enemy_grid : chex.Array
        `(rows, cols)` bool — alive/dead flag for each enemy cell.
    fleet_x : chex.Array
        float32 scalar — x anchor of the leftmost enemy column.
    fleet_y : chex.Array
        float32 scalar — y anchor of the topmost enemy row.
    fleet_dir : chex.Array
        int32 scalar — horizontal direction: `+1` = right, `-1` = left.
    fleet_speed : chex.Array
        float32 scalar — pixels moved per physics step.
    player_bullets : chex.Array
        `(N, 3)` float32 — player projectile pool `[x, y, active]`.
    enemy_bullets : chex.Array
        `(M, 3)` float32 — enemy projectile pool `[x, y, active]`.
    """

    player_x: chex.Array
    fire_cooldown: chex.Array
    enemy_grid: chex.Array
    fleet_x: chex.Array
    fleet_y: chex.Array
    fleet_dir: chex.Array
    fleet_speed: chex.Array
    player_bullets: chex.Array
    enemy_bullets: chex.Array


class FixedShooterGame(AtaraxGame):
    """
    Abstract base class for Template 1 Fixed Shooter games.

    Provides shared, branch-free physics helpers that all T1 games can reuse.
    Concrete games inherit this class and implement `_reset`,
    `_step`, and `render`.
    """

    def _move_bullets(
        self,
        bullets: chex.Array,
        speed_y: chex.Array,
    ) -> chex.Array:
        """
        Advance all projectiles in a pool by `speed_y` pixels and deactivate
        any that leave the world boundaries (y < 0 or y > 210).

        Movement is gated by the active flag — inactive bullets are not moved
        (arithmetic gate avoids branching).

        Parameters
        ----------
        bullets : chex.Array
            (N, 3) float32 — pool with columns `[x, y, active]`.
        speed_y : chex.Array
            float32 scalar — pixels per step; positive = moving downward.

        Returns
        -------
        updated : chex.Array
            (N, 3) float32 — updated pool with new y positions and deactivated
            out-of-bounds projectiles.
        """
        new_y = bullets[:, 1] + speed_y * bullets[:, 2]
        oob = (new_y < 0.0) | (new_y > 210.0)
        new_active = bullets[:, 2] * (~oob).astype(jnp.float32)
        return jnp.stack([bullets[:, 0], new_y, new_active], axis=1)

    def _compute_fleet_speed(
        self,
        n_alive: chex.Array,
        base_speed: chex.Array,
        speed_gain: chex.Array,
        total_enemies: int,
    ) -> chex.Array:
        """
        Compute fleet speed as a linear function of the number of enemies
        remaining — the formation accelerates as enemies are destroyed.

        Parameters
        ----------
        n_alive : chex.Array
            int32 scalar — number of alive enemies.
        base_speed : chex.Array
            float32 scalar — speed at full enemy count.
        speed_gain : chex.Array
            float32 scalar — extra speed added per enemy destroyed.
        total_enemies : int
            Total enemy count at the start of the wave.

        Returns
        -------
        speed : chex.Array
            float32 scalar — current fleet speed in pixels per step.
        """
        n_dead = jnp.float32(total_enemies) - n_alive.astype(jnp.float32)
        return base_speed + n_dead * speed_gain

    def _move_fleet(
        self,
        fleet_x: chex.Array,
        fleet_y: chex.Array,
        fleet_dir: chex.Array,
        fleet_speed: chex.Array,
        x_min: chex.Array,
        x_max: chex.Array,
        drop: chex.Array,
        left_edge: chex.Array,
        right_edge: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """
        Move the enemy formation one step and handle edge reversal.

        The fleet moves horizontally. When the leftmost alive column reaches
        `x_min` (moving left) or the rightmost alive column reaches `x_max`
        (moving right), the direction reverses and the fleet drops `drop`
        pixels downward.

        Parameters
        ----------
        fleet_x : chex.Array
            float32 scalar — current x anchor.
        fleet_y : chex.Array
            float32 scalar — current y anchor.
        fleet_dir : chex.Array
            int32 scalar — current direction: `+1` = right, `-1` = left.
        fleet_speed : chex.Array
            float32 scalar — pixels per step.
        x_min : chex.Array
            float32 scalar — left reversal boundary.
        x_max : chex.Array
            float32 scalar — right reversal boundary.
        drop : chex.Array
            float32 scalar — y pixels dropped on each reversal.
        left_edge : chex.Array
            float32 scalar — current world x of the leftmost alive alien.
        right_edge : chex.Array
            float32 scalar — current world x of the rightmost alive alien.

        Returns
        -------
        new_fleet_x : chex.Array
            float32 scalar — updated x anchor.
        new_fleet_y : chex.Array
            float32 scalar — updated y anchor (incremented on reversal).
        new_fleet_dir : chex.Array
            int32 scalar — updated direction (flipped on reversal).
        """
        hit_left = (left_edge <= x_min) & (fleet_dir < 0)
        hit_right = (right_edge >= x_max) & (fleet_dir > 0)
        reverse = hit_left | hit_right

        new_dir = jnp.where(reverse, -fleet_dir, fleet_dir)
        new_y = fleet_y + jnp.where(reverse, drop, jnp.float32(0.0))
        new_x = fleet_x + fleet_speed * new_dir.astype(jnp.float32)
        return new_x, new_y, new_dir

    def _bullet_rect_hits(
        self,
        bullets: chex.Array,
        entity_xs: chex.Array,
        entity_ys: chex.Array,
        hit_hw: chex.Array,
        hit_hh: chex.Array,
    ) -> chex.Array:
        """
        Compute which `(bullet, entity)` pairs are in AABB contact.

        Fully vectorised — no Python loops. Broadcasts `(N_bullets,)` against
        `(N_entities,)` to produce a `(N_bullets, N_entities)` hit matrix.

        Parameters
        ----------
        bullets : chex.Array
            (N, 3) float32 — bullet pool `[x, y, active]`.
        entity_xs : chex.Array
            (M,) float32 — entity centre x positions.
        entity_ys : chex.Array
            (M,) float32 — entity centre y positions.
        hit_hw : chex.Array
            float32 scalar — half-width of the hit box.
        hit_hh : chex.Array
            float32 scalar — half-height of the hit box.

        Returns
        -------
        hit_matrix : chex.Array
            (N, M) bool — `True` where bullet `n` overlaps entity `m`,
            subject to `bullets[n, 2] > 0` (bullet active).
        """
        bx = bullets[:, 0][:, None]  # (N, 1)
        by = bullets[:, 1][:, None]  # (N, 1)
        ba = bullets[:, 2][:, None]  # (N, 1)  active flag

        ex = entity_xs[None, :]  # (1, M)
        ey = entity_ys[None, :]  # (1, M)

        dx = jnp.abs(bx - ex)
        dy = jnp.abs(by - ey)
        return (dx < hit_hw) & (dy < hit_hh) & (ba > 0.0)
