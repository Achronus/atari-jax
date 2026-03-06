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

"""Template 5 — Maze Navigator base classes.

Games in this template (Pac-Man, Ms. Pac-Man, …):

- Player navigates a fixed tile-based maze
- Movement is grid-snapped — one tile per step at a configurable speed
- Collectibles (pellets, coins) are tracked as a boolean grid
- Enemy agents (ghosts) traverse the maze with AI-driven tile selection

Concrete games inherit `MazeNavigatorState` to add game-specific fields,
and `MazeNavigatorGame` to gain shared movement helpers.
"""

import chex
import jax.numpy as jnp

from atarax.game import AtaraxGame
from atarax.state import AtariState

# Direction encoding used throughout the template.
# DIR_UP=0, DIR_RIGHT=1, DIR_DOWN=2, DIR_LEFT=3
_DR = jnp.array([-1, 0, 1, 0], dtype=jnp.int32)  # row delta per direction
_DC = jnp.array([0, 1, 0, -1], dtype=jnp.int32)  # col delta per direction


@chex.dataclass
class MazeNavigatorState(AtariState):
    """
    Shared state for Template 5 Maze Navigator games.

    Inherits `reward`, `done`, `step`, `episode_step`, `lives`,
    `score`, `level`, and `key` from `~atarax.state.AtariState`.

    Concrete game states (e.g. `PacManState`) inherit from this class
    and add any game-specific fields on top.

    Parameters
    ----------
    player_row : chex.Array
        int32 scalar — player tile row index.
    player_col : chex.Array
        int32 scalar — player tile column index.
    player_dir : chex.Array
        int32 scalar — current movement direction: `0`=up, `1`=right,
        `2`=down, `3`=left.
    tile_map : chex.Array
        `(R, C)` int32 — static maze layout. `0`=open, `1`=wall;
        additional values are game-specific (e.g. ghost house).
    collectibles : chex.Array
        `(R, C)` bool — `True` where a collectible (pellet, coin) is present.
    power_timer : chex.Array
        int32 scalar — countdown frames of active power-up; `0` = inactive.
    ghost_row : chex.Array
        `(N,)` int32 — ghost tile row indices.
    ghost_col : chex.Array
        `(N,)` int32 — ghost tile column indices.
    ghost_dir : chex.Array
        `(N,)` int32 — ghost movement directions (same encoding as `player_dir`).
    ghost_mode : chex.Array
        `(N,)` int32 — ghost AI mode: `0`=scatter, `1`=chase, `2`=fright,
        `3`=eaten, `4`=house.
    """

    player_row: chex.Array
    player_col: chex.Array
    player_dir: chex.Array
    tile_map: chex.Array
    collectibles: chex.Array
    power_timer: chex.Array
    ghost_row: chex.Array
    ghost_col: chex.Array
    ghost_dir: chex.Array
    ghost_mode: chex.Array


class MazeNavigatorGame(AtaraxGame):
    """
    Abstract base class for Template 5 Maze Navigator games.

    Provides shared, branch-free grid movement helpers that all T5 games
    can reuse. Concrete games inherit this class and implement `_reset`,
    `_step`, and `render`.
    """

    def _can_move(
        self,
        row: chex.Array,
        col: chex.Array,
        direction: chex.Array,
        tile_map: chex.Array,
    ) -> chex.Array:
        """
        Check whether an entity can move one tile in `direction`.

        The target tile is considered passable when its value in `tile_map`
        is `0` (open). Out-of-bounds accesses are safe because `jnp.clip`
        keeps indices within the array; out-of-bounds tiles are treated as
        walls (`tile_map[clipped] != 0` is always possible — concrete
        subclasses must ensure the maze has a wall border).

        Parameters
        ----------
        row : chex.Array
            int32 scalar — current tile row.
        col : chex.Array
            int32 scalar — current tile column.
        direction : chex.Array
            int32 scalar — direction to test: `0`=up, `1`=right,
            `2`=down, `3`=left.
        tile_map : chex.Array
            (R, C) int32 — maze layout; `0`=passable.

        Returns
        -------
        passable : chex.Array
            bool scalar — `True` if the next tile is open.
        """
        rows, cols = tile_map.shape
        next_row = jnp.clip(row + _DR[direction], 0, rows - 1)
        next_col = jnp.clip(col + _DC[direction], 0, cols - 1)
        return tile_map[next_row, next_col] == 0

    def _step_entity(
        self,
        row: chex.Array,
        col: chex.Array,
        direction: chex.Array,
        tile_map: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Move an entity one tile in `direction` if the target is passable.

        No movement occurs when the tile is blocked — the entity stays in
        place. All arithmetic is branch-free; the move is gated by a
        `jnp.where` on the passable flag.

        Parameters
        ----------
        row : chex.Array
            int32 scalar — current tile row.
        col : chex.Array
            int32 scalar — current tile column.
        direction : chex.Array
            int32 scalar — direction to move: `0`=up, `1`=right,
            `2`=down, `3`=left.
        tile_map : chex.Array
            (R, C) int32 — maze layout; `0`=passable.

        Returns
        -------
        new_row : chex.Array
            int32 scalar — updated tile row.
        new_col : chex.Array
            int32 scalar — updated tile column.
        """
        passable = self._can_move(row, col, direction, tile_map)
        new_row = jnp.where(passable, row + _DR[direction], row)
        new_col = jnp.where(passable, col + _DC[direction], col)
        return new_row, new_col

    def _grid_to_pixel(
        self,
        row: chex.Array,
        col: chex.Array,
        tile_h: chex.Array,
        tile_w: chex.Array,
        offset_y: chex.Array,
        offset_x: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Convert tile grid coordinates to world-space pixel centres.

        The pixel centre is the middle of the tile, offset by the grid
        origin `(offset_x, offset_y)` in world space.

        Parameters
        ----------
        row : chex.Array
            int32 scalar (or array) — tile row index(es).
        col : chex.Array
            int32 scalar (or array) — tile column index(es).
        tile_h : chex.Array
            float32 scalar — tile height in world pixels.
        tile_w : chex.Array
            float32 scalar — tile width in world pixels.
        offset_y : chex.Array
            float32 scalar — world y of the grid's top edge.
        offset_x : chex.Array
            float32 scalar — world x of the grid's left edge.

        Returns
        -------
        pixel_x : chex.Array
            float32 — world x centre(s) of the tile(s).
        pixel_y : chex.Array
            float32 — world y centre(s) of the tile(s).
        """
        pixel_x = (
            offset_x + col.astype(jnp.float32) * tile_w + tile_w * jnp.float32(0.5)
        )
        pixel_y = (
            offset_y + row.astype(jnp.float32) * tile_h + tile_h * jnp.float32(0.5)
        )
        return pixel_x, pixel_y

    def _pick_ghost_direction(
        self,
        row: chex.Array,
        col: chex.Array,
        current_dir: chex.Array,
        target_row: chex.Array,
        target_col: chex.Array,
        tile_map: chex.Array,
    ) -> chex.Array:
        """
        Select the ghost's next direction using Manhattan-distance targeting.

        At each tile the ghost evaluates all four directions. The direction
        that minimises Manhattan distance to the target tile is chosen,
        subject to two constraints: the direction must lead to a passable tile
        and ghosts cannot reverse direction (180° turns are forbidden).

        All selection is done via `jnp.argmin` + `jnp.where` — no Python
        branching, fully JIT-safe.

        Parameters
        ----------
        row : chex.Array
            int32 scalar — ghost current tile row.
        col : chex.Array
            int32 scalar — ghost current tile column.
        current_dir : chex.Array
            int32 scalar — ghost current direction (reversal is forbidden).
        target_row : chex.Array
            int32 scalar — AI target tile row.
        target_col : chex.Array
            int32 scalar — AI target tile col.
        tile_map : chex.Array
            (R, C) int32 — maze layout.

        Returns
        -------
        new_dir : chex.Array
            int32 scalar — chosen direction for next step.
        """
        rows, cols = tile_map.shape
        reverse_dir = (current_dir + 2) % 4

        dists = jnp.full((4,), jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
        for d in range(4):
            next_row = jnp.clip(row + _DR[d], 0, rows - 1)
            next_col = jnp.clip(col + _DC[d], 0, cols - 1)
            passable = tile_map[next_row, next_col] == 0
            not_reverse = d != reverse_dir
            dist = jnp.abs(next_row - target_row) + jnp.abs(next_col - target_col)
            blocked_val = jnp.iinfo(jnp.int32).max
            dists = dists.at[d].set(
                jnp.where(passable & not_reverse, dist, blocked_val)
            )

        return jnp.argmin(dists).astype(jnp.int32)
