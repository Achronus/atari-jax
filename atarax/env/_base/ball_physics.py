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

"""Template 6 — Ball / Physics base classes.

Games in this template (Breakout, Video Pinball, …):

- A ball bounces around the play field under Newtonian reflection
- The player controls a paddle or flipper to keep the ball in play
- Targets (bricks, bumpers, pins) are destroyed on contact
- Speed may escalate as targets are cleared

Concrete games inherit `BallPhysicsState` to add game-specific fields,
and `BallPhysicsGame` to gain shared bounce and collision helpers.
"""

import chex
import jax.numpy as jnp

from atarax.game import AtaraxGame
from atarax.state import AtariState


@chex.dataclass
class BallPhysicsState(AtariState):
    """
    Shared state for Template 6 Ball / Physics games.

    Inherits `reward`, `done`, `step`, `episode_step`, `lives`,
    `score`, `level`, and `key` from `~atarax.state.AtariState`.

    Concrete game states (e.g. `BreakoutState`) inherit from this class
    and add any game-specific fields on top.

    Parameters
    ----------
    ball_x : chex.Array
        float32 scalar — ball centre x in world coordinates.
    ball_y : chex.Array
        float32 scalar — ball centre y in world coordinates.
    ball_vx : chex.Array
        float32 scalar — ball horizontal velocity (pixels per step); positive = right.
    ball_vy : chex.Array
        float32 scalar — ball vertical velocity; positive = moving downward.
    ball_in_play : chex.Array
        bool scalar — `False` when ball is held on the paddle before launch.
    paddle_x : chex.Array
        float32 scalar — paddle centre x in world coordinates.
    targets : chex.Array
        `(R, C)` bool — alive flag for each target cell (bricks, bumpers, etc.).
    """

    ball_x: chex.Array
    ball_y: chex.Array
    ball_vx: chex.Array
    ball_vy: chex.Array
    ball_in_play: chex.Array
    paddle_x: chex.Array
    targets: chex.Array


class BallPhysicsGame(AtaraxGame):
    """
    Abstract base class for Template 6 Ball / Physics games.

    Provides shared, branch-free physics helpers that all T6 games can reuse.
    Concrete games inherit this class and implement `_reset`,
    `_step`, and `render`.
    """

    def _bounce_walls(
        self,
        bx: chex.Array,
        by: chex.Array,
        vx: chex.Array,
        vy: chex.Array,
        r: chex.Array,
        x_lo: chex.Array,
        x_hi: chex.Array,
        y_lo: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Reflect the ball off the left/right walls and ceiling.

        The ball is clamped back inside the boundary after reflection so it
        cannot clip through at high speed. No bottom boundary is handled here
        — ball loss is game-specific logic.

        Parameters
        ----------
        bx : chex.Array
            float32 scalar — ball centre x after movement.
        by : chex.Array
            float32 scalar — ball centre y after movement.
        vx : chex.Array
            float32 scalar — horizontal velocity.
        vy : chex.Array
            float32 scalar — vertical velocity.
        r : chex.Array
            float32 scalar — ball radius.
        x_lo : chex.Array
            float32 scalar — left wall inner x boundary.
        x_hi : chex.Array
            float32 scalar — right wall inner x boundary.
        y_lo : chex.Array
            float32 scalar — ceiling inner y boundary.

        Returns
        -------
        new_bx : chex.Array
            float32 scalar — clamped ball x.
        new_by : chex.Array
            float32 scalar — clamped ball y.
        new_vx : chex.Array
            float32 scalar — reflected horizontal velocity.
        new_vy : chex.Array
            float32 scalar — reflected vertical velocity (ceiling bounce only).
        """
        hit_left = bx - r < x_lo
        hit_right = bx + r > x_hi
        hit_ceil = by - r < y_lo

        new_vx = jnp.where(hit_left | hit_right, -vx, vx)
        new_vy = jnp.where(hit_ceil, jnp.abs(vy), vy)
        new_bx = jnp.clip(bx, x_lo + r, x_hi - r)
        new_by = jnp.where(hit_ceil, y_lo + r, by)
        return new_bx, new_by, new_vx, new_vy

    def _paddle_reflect(
        self,
        bx: chex.Array,
        by: chex.Array,
        vx: chex.Array,
        vy: chex.Array,
        r: chex.Array,
        px: chex.Array,
        paddle_hw: chex.Array,
        paddle_hh: chex.Array,
        paddle_y: chex.Array,
        speed: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Reflect the ball off the paddle with spin-based angle deflection.

        Horizontal deflection depends on where along the paddle the ball
        lands. The hit position `hit_pos` is in `[-1, +1]`; `-1` = far left,
        `+1` = far right. Speed magnitude is preserved. Only activates when
        the ball is moving downward and overlaps the paddle face.

        Parameters
        ----------
        bx : chex.Array
            float32 scalar — ball centre x.
        by : chex.Array
            float32 scalar — ball centre y.
        vx : chex.Array
            float32 scalar — horizontal velocity.
        vy : chex.Array
            float32 scalar — vertical velocity; positive = down.
        r : chex.Array
            float32 scalar — ball radius.
        px : chex.Array
            float32 scalar — paddle centre x.
        paddle_hw : chex.Array
            float32 scalar — paddle half-width.
        paddle_hh : chex.Array
            float32 scalar — paddle half-height.
        paddle_y : chex.Array
            float32 scalar — paddle centre y.
        speed : chex.Array
            float32 scalar — target ball speed magnitude after reflection.

        Returns
        -------
        new_bx : chex.Array
            float32 scalar — ball x (unchanged).
        new_by : chex.Array
            float32 scalar — ball y repositioned just above the paddle.
        new_vx : chex.Array
            float32 scalar — deflected horizontal velocity.
        new_vy : chex.Array
            float32 scalar — upward vertical velocity preserving speed.
        """
        hit = (
            (vy > jnp.float32(0.0))
            & (by + r >= paddle_y - paddle_hh)
            & (jnp.abs(bx - px) <= paddle_hw + r)
        )
        hit_pos = (bx - px) / paddle_hw
        new_vx_val = hit_pos * speed * jnp.float32(0.75)
        new_vy_val = -jnp.sqrt(jnp.maximum(speed**2 - new_vx_val**2, jnp.float32(0.5)))
        new_vx = jnp.where(hit, new_vx_val, vx)
        new_vy = jnp.where(hit, new_vy_val, vy)
        new_by = jnp.where(hit, paddle_y - paddle_hh - r, by)
        return bx, new_by, new_vx, new_vy

    def _grid_collide(
        self,
        bx: chex.Array,
        by: chex.Array,
        vx: chex.Array,
        vy: chex.Array,
        r: chex.Array,
        targets: chex.Array,
        x0: chex.Array,
        y0: chex.Array,
        cell_w: chex.Array,
        cell_h: chex.Array,
        row_scores: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Detect and resolve ball collisions with a grid of rectangular targets.

        Fully vectorised — no Python loops. Computes AABB overlap for all
        `(R, C)` cells simultaneously, removes hit cells, accumulates score,
        and determines the bounce direction from the hit face.

        Parameters
        ----------
        bx : chex.Array
            float32 scalar — ball centre x.
        by : chex.Array
            float32 scalar — ball centre y.
        vx : chex.Array
            float32 scalar — horizontal velocity.
        vy : chex.Array
            float32 scalar — vertical velocity.
        r : chex.Array
            float32 scalar — ball radius.
        targets : chex.Array
            (R, C) bool — alive flag for each target cell.
        x0 : chex.Array
            float32 scalar — world x of the grid's left edge.
        y0 : chex.Array
            float32 scalar — world y of the grid's top edge.
        cell_w : chex.Array
            float32 scalar — width of each cell in world pixels.
        cell_h : chex.Array
            float32 scalar — height of each cell in world pixels.
        row_scores : chex.Array
            (R,) int32 — point value for destroying one cell in each row.

        Returns
        -------
        new_targets : chex.Array
            (R, C) bool — targets with hit cells cleared.
        delta_score : chex.Array
            int32 scalar — score gained this step.
        new_vx : chex.Array
            float32 scalar — reflected horizontal velocity.
        new_vy : chex.Array
            float32 scalar — reflected vertical velocity.
        hit_any : chex.Array
            bool scalar — `True` if at least one target was hit.
        """
        rows, cols = targets.shape
        col_idx = jnp.arange(cols, dtype=jnp.float32)
        row_idx = jnp.arange(rows, dtype=jnp.float32)

        hw = cell_w * jnp.float32(0.5)
        hh = cell_h * jnp.float32(0.5)
        cx = (x0 + col_idx * cell_w + hw)[None, :]  # (1, C)
        cy = (y0 + row_idx * cell_h + hh)[:, None]  # (R, 1)

        overlap = (
            (bx - r < cx + hw)
            & (bx + r > cx - hw)
            & (by - r < cy + hh)
            & (by + r > cy - hh)
            & targets
        )
        hit_any = jnp.any(overlap)
        new_targets = targets & ~overlap
        delta_score = jnp.sum(overlap * row_scores[:, None]).astype(jnp.int32)

        # Bounce direction: if ball centre is within the x-range of a hit cell it
        # struck the top/bottom face (reverse vy); otherwise it hit a side (reverse vx).
        n_hits = jnp.maximum(jnp.sum(overlap).astype(jnp.float32), jnp.float32(1.0))
        hit_cx = jnp.sum(overlap * cx) / n_hits
        in_x_range = jnp.abs(bx - hit_cx) < hw
        new_vx = jnp.where(hit_any & ~in_x_range, -vx, vx)
        new_vy = jnp.where(hit_any & in_x_range, -vy, vy)
        return new_targets, delta_score, new_vx, new_vy, hit_any
