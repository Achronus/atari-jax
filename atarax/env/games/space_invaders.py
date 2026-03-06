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

"""Space Invaders — SDF-rendered JAX-native game implementation.

Template: T1 — Fixed Shooter.
Action space (6 discrete actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — RIGHT + FIRE
    5 — LEFT + FIRE
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.fixed_shooter import FixedShooterGame, FixedShooterState
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_bool_grid,
    render_circle_pool,
    render_rect_pool,
    sdf_rect,
)
from atarax.game import AtaraxParams

_ROWS: int = 6
_COLS: int = 6
_TOTAL_ALIENS: int = _ROWS * _COLS  # 36

# Row score values (row 0 = top = highest value).
# Atari 2600 version: top 2 rows = 30 pts, middle 2 = 20 pts, bottom 2 = 10 pts.
_ROW_SCORES = jnp.array([30, 30, 20, 20, 10, 10], dtype=jnp.int32)

# Shield layout: 3 shields, each 22 cells wide at 1 px/cell → 22 px per shield.
# Positioned at x=[34, 69, 104] with 13 px gaps between them, centred on 160 px screen.
_SHIELD_ANCHOR_XS = jnp.array([34.0, 69.0, 104.0], dtype=jnp.float32)
_SHIELD_CELL_W: float = 1.0  # world-px per shield cell (horizontal)
_SHIELD_CELL_H: float = 5.0  # world-px per shield cell (vertical)

# Pre-computed column / row offset arrays for alien position derivation
_COL_OFFSETS = jnp.arange(_COLS, dtype=jnp.float32)  # (6,)
_ROW_OFFSETS = jnp.arange(_ROWS, dtype=jnp.float32)  # (6,)

# Rendering colours (float32 RGB [0, 1])
_COL_BG = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
_COL_GROUND = jnp.array([0.3, 0.3, 0.3], dtype=jnp.float32)
_COL_SHIELD = jnp.array([0.2, 0.5, 1.0], dtype=jnp.float32)
# Per-type alien colours: row 0 = squid (yellow), rows 1-2 = crab (red), rows 3-4 = octopus (orange)
_COL_ALIEN_TOP = jnp.array([1.0, 0.90, 0.15], dtype=jnp.float32)
_COL_ALIEN_MID = jnp.array([0.85, 0.20, 0.20], dtype=jnp.float32)
_COL_ALIEN_BOT = jnp.array([1.0, 0.55, 0.10], dtype=jnp.float32)
_COL_ALIEN_BULLET = jnp.array([1.0, 0.5, 0.1], dtype=jnp.float32)
_COL_PLAYER_BULLET = jnp.array([1.0, 1.0, 0.2], dtype=jnp.float32)
_COL_PLAYER = jnp.array([0.2, 1.0, 0.35], dtype=jnp.float32)
_COL_HUD = jnp.array([0.12, 0.6, 0.21], dtype=jnp.float32)  # player colour × 0.6


@chex.dataclass
class SpaceInvadersParams(AtaraxParams):
    """
    Space Invaders environment parameters.

    Extends `AtaraxParams` with game-specific constants for world geometry,
    fleet behaviour, projectile speeds, and scoring.

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode. Default: 10000.
    num_lives : int
        Starting lives. Default: 3.
    ground_y : float
        Y coordinate of the ground line. Episode ends if the fleet reaches it.
    player_speed : float
        Pixels the cannon moves per physics step.
    player_x_min : float
        Left clamp boundary for the cannon.
    player_x_max : float
        Right clamp boundary for the cannon.
    player_y : float
        Fixed cannon y coordinate (never changes).
    fire_cooldown : int
        Physics steps between consecutive player shots.
    fleet_x_init : float
        Initial x anchor of the leftmost alien column.
    fleet_y_init : float
        Initial y anchor of the top alien row.
    fleet_x_min : float
        Left reversal boundary for the fleet.
    fleet_x_max : float
        Right reversal boundary for the fleet.
    fleet_col_gap : float
        Horizontal spacing between alien column centres.
    fleet_row_gap : float
        Vertical spacing between alien row centres.
    fleet_speed_base : float
        Fleet speed at full enemy count (pixels per step).
    fleet_speed_gain : float
        Extra speed added per alien destroyed.
    fleet_drop : float
        Pixels the fleet drops on each direction reversal.
    alien_fire_prob : float
        Probability per physics step that a random alien fires.
    alien_bullet_speed : float
        Downward speed of alien bullets (pixels per step).
    player_bullet_speed : float
        Upward speed of player bullets (pixels per step).
    max_player_bullets : int
        Maximum simultaneous player bullets.
    max_alien_bullets : int
        Maximum simultaneous alien bullets.
    shield_y : float
        Y coordinate of the shield row centre.
    score_row_0 : int
        Points for killing an alien in row 0 (top row, hardest to reach).
    score_row_1 : int
        Points for killing an alien in rows 1–2.
    score_row_3 : int
        Points for killing an alien in rows 3–4 (bottom rows, easiest).
    """

    max_steps: int = 10000
    num_lives: int = 3
    ground_y: float = 195.0
    player_speed: float = 2.0
    player_x_min: float = 8.0
    player_x_max: float = 152.0
    player_y: float = 185.0
    fire_cooldown: int = 8
    fleet_x_init: float = 16.0
    fleet_y_init: float = 40.0
    fleet_x_min: float = 8.0
    fleet_x_max: float = 152.0
    fleet_col_gap: float = 14.0
    fleet_row_gap: float = 12.0
    fleet_speed_base: float = 1.0
    fleet_speed_gain: float = 0.08
    fleet_drop: float = 6.0
    alien_fire_prob: float = 0.015
    alien_bullet_speed: float = 3.0
    player_bullet_speed: float = 6.0
    max_player_bullets: int = 3
    max_alien_bullets: int = 6
    shield_y: float = 156.0
    score_row_0: int = 30
    score_row_1: int = 20
    score_row_3: int = 10


@chex.dataclass
class SpaceInvadersState(FixedShooterState):
    """
    Space Invaders game state.

    Extends `FixedShooterState` with destructible shield grids.

    Inherited from `FixedShooterState`:
        `player_x`, `fire_cooldown`, `enemy_grid` (6×6 alien grid),
        `fleet_x`, `fleet_y`, `fleet_dir`, `fleet_speed`,
        `player_bullets` (3×3), `enemy_bullets` (6×3).

    Inherited from `AtariState`:
        `reward`, `done`, `step`, `episode_step`, `lives`, `score`, `level`, `key`.

    Parameters
    ----------
    shields : chex.Array
        (3, 22) bool — `True` where each shield cell is still intact.
        Three shields, each 22 cells wide; cells degrade when hit by bullets.
    """

    shields: chex.Array


class SpaceInvaders(FixedShooterGame):
    """
    Space Invaders implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    Alien positions are always derived from `fleet_x`, `fleet_y`, and grid indices
    — no per-alien position storage.
    """

    num_actions: int = 6
    game_id: ClassVar[str] = "space_invaders"

    def _reset(self, rng: chex.PRNGKey) -> SpaceInvadersState:
        return SpaceInvadersState(
            # FixedShooterState fields
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            enemy_grid=jnp.ones((_ROWS, _COLS), dtype=jnp.bool_),
            fleet_x=jnp.float32(16.0),
            fleet_y=jnp.float32(40.0),
            fleet_dir=jnp.int32(1),
            fleet_speed=jnp.float32(1.0),
            player_bullets=jnp.zeros((3, 3), dtype=jnp.float32),
            enemy_bullets=jnp.zeros((6, 3), dtype=jnp.float32),
            # SpaceInvadersState fields
            shields=jnp.ones((3, 22), dtype=jnp.bool_),
            # AtariState fields
            lives=jnp.int32(3),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=rng,
        )

    def _step_physics(
        self,
        state: SpaceInvadersState,
        action: chex.Array,
        params: SpaceInvadersParams,
        rng: chex.PRNGKey,
    ) -> SpaceInvadersState:
        rng, fire_rng, col_rng = jax.random.split(rng, 3)

        # ── 1. Player movement
        move_right = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_left = (action == jnp.int32(3)) | (action == jnp.int32(5))
        dx = jnp.where(
            move_right,
            jnp.float32(params.player_speed),
            jnp.where(move_left, jnp.float32(-params.player_speed), jnp.float32(0.0)),
        )
        player_x = jnp.clip(
            state.player_x + dx,
            jnp.float32(params.player_x_min),
            jnp.float32(params.player_x_max),
        )

        # ── 2. Fire player bullet
        want_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(4))
            | (action == jnp.int32(5))
        )
        can_fire = want_fire & (state.fire_cooldown <= jnp.int32(0))
        slot = jnp.argmin(state.player_bullets[:, 2])
        slot_free = state.player_bullets[slot, 2] < jnp.float32(0.5)
        do_fire = can_fire & slot_free
        new_pb = jnp.array(
            [
                player_x,
                jnp.float32(params.player_y) - jnp.float32(4.0),
                jnp.float32(1.0),
            ]
        )
        player_bullets = jnp.where(
            do_fire, state.player_bullets.at[slot].set(new_pb), state.player_bullets
        )
        fire_cooldown = jnp.where(
            do_fire, jnp.int32(params.fire_cooldown), state.fire_cooldown - jnp.int32(1)
        )
        fire_cooldown = jnp.maximum(fire_cooldown, jnp.int32(0))

        # ── 3. Move bullets
        player_bullets = self._move_bullets(
            player_bullets, jnp.float32(-params.player_bullet_speed)
        )
        enemy_bullets = self._move_bullets(
            state.enemy_bullets, jnp.float32(params.alien_bullet_speed)
        )

        # ── 4. Fleet speed
        n_alive = jnp.sum(state.enemy_grid).astype(jnp.int32)
        fleet_speed = self._compute_fleet_speed(
            n_alive,
            jnp.float32(params.fleet_speed_base),
            jnp.float32(params.fleet_speed_gain),
            _TOTAL_ALIENS,
        )

        # ── 5. Fleet movement
        alive_cols = jnp.any(state.enemy_grid, axis=0)  # (11,) bool
        left_col = jnp.argmax(alive_cols).astype(jnp.float32)
        right_col = (jnp.int32(_COLS - 1) - jnp.argmax(alive_cols[::-1])).astype(
            jnp.float32
        )
        left_edge = state.fleet_x + left_col * jnp.float32(params.fleet_col_gap)
        right_edge = (
            state.fleet_x
            + right_col * jnp.float32(params.fleet_col_gap)
            + jnp.float32(9.0)
        )

        fleet_x, fleet_y, fleet_dir = self._move_fleet(
            state.fleet_x,
            state.fleet_y,
            state.fleet_dir,
            fleet_speed,
            jnp.float32(params.fleet_x_min),
            jnp.float32(params.fleet_x_max),
            jnp.float32(params.fleet_drop),
            left_edge,
            right_edge,
        )

        # ── 6. Alien fires
        rand_col = jax.random.randint(col_rng, (), minval=0, maxval=_COLS)
        alive_in_col = state.enemy_grid[:, rand_col]  # (5,) bool
        # Bottom-most alive alien in chosen column
        bottom_row = (jnp.int32(_ROWS - 1) - jnp.argmax(alive_in_col[::-1])).astype(
            jnp.float32
        )
        bx = fleet_x + rand_col.astype(jnp.float32) * jnp.float32(params.fleet_col_gap)
        by = fleet_y + bottom_row * jnp.float32(params.fleet_row_gap) + jnp.float32(6.0)
        fire_prob = jax.random.uniform(fire_rng)
        alien_slot = jnp.argmin(enemy_bullets[:, 2])
        alien_slot_free = enemy_bullets[alien_slot, 2] < jnp.float32(0.5)
        do_alien_fire = (
            jnp.any(alive_in_col)
            & (fire_prob < jnp.float32(params.alien_fire_prob))
            & alien_slot_free
        )
        new_ab = jnp.array([bx, by, jnp.float32(1.0)])
        enemy_bullets = jnp.where(
            do_alien_fire, enemy_bullets.at[alien_slot].set(new_ab), enemy_bullets
        )

        # ── 7. Player bullets vs alien grid
        flat_xs = jnp.tile(
            fleet_x + _COL_OFFSETS * jnp.float32(params.fleet_col_gap), _ROWS
        )  # (36,)
        flat_ys = jnp.repeat(
            fleet_y + _ROW_OFFSETS * jnp.float32(params.fleet_row_gap), _COLS
        )  # (36,)
        pb_hits = self._bullet_rect_hits(
            player_bullets,
            flat_xs,
            flat_ys,
            jnp.float32(4.5),
            jnp.float32(4.0),  # >speed/2=3 to prevent skip-through at 6px/step
        )  # (3, 36)
        alive_flat = state.enemy_grid.ravel()
        pb_alien_hits = pb_hits & alive_flat[None, :]  # (3, 55) — only alive aliens
        alien_hit_flat = jnp.any(pb_alien_hits, axis=0)  # (55,)
        alien_hit_grid = alien_hit_flat.reshape(_ROWS, _COLS)
        enemy_grid = state.enemy_grid & ~alien_hit_grid
        # Score: aliens destroyed × per-row points
        delta_score = jnp.sum(
            jnp.sum(alien_hit_grid, axis=1).astype(jnp.int32) * _ROW_SCORES
        )
        # Deactivate bullets that hit an alien
        pb_killed_any = jnp.any(pb_alien_hits, axis=1)  # (3,)
        player_bullets = player_bullets.at[:, 2].set(
            player_bullets[:, 2] * (~pb_killed_any).astype(jnp.float32)
        )

        # ── 8. Bullets vs shields
        shields = state.shields
        cell_j = jnp.arange(22, dtype=jnp.float32)
        shield_ys = jnp.full((22,), jnp.float32(params.shield_y))
        hw = jnp.float32(_SHIELD_CELL_W * 0.6)
        hh = jnp.float32(_SHIELD_CELL_H * 0.6)
        for s in range(3):
            cell_xs = _SHIELD_ANCHOR_XS[s] + cell_j * jnp.float32(_SHIELD_CELL_W)
            # Player bullets vs this shield
            pb_s_hits = self._bullet_rect_hits(
                player_bullets, cell_xs, shield_ys, hw, hh
            )  # (3, 22)
            alive_cells = shields[s]  # (22,) bool
            pb_s_live = pb_s_hits & alive_cells[None, :]
            hit_cells = jnp.any(pb_s_live, axis=0)  # (22,)
            pb_hit_shield = jnp.any(pb_s_live, axis=1)  # (3,)
            shields = shields.at[s].set(shields[s] & ~hit_cells)
            player_bullets = player_bullets.at[:, 2].set(
                player_bullets[:, 2] * (~pb_hit_shield).astype(jnp.float32)
            )
            # Alien bullets vs this shield
            ab_s_hits = self._bullet_rect_hits(
                enemy_bullets, cell_xs, shield_ys, hw, hh
            )  # (6, 22)
            ab_s_live = ab_s_hits & alive_cells[None, :]
            hit_cells_ab = jnp.any(ab_s_live, axis=0)  # (22,)
            ab_hit_shield = jnp.any(ab_s_live, axis=1)  # (6,)
            shields = shields.at[s].set(shields[s] & ~hit_cells_ab)
            enemy_bullets = enemy_bullets.at[:, 2].set(
                enemy_bullets[:, 2] * (~ab_hit_shield).astype(jnp.float32)
            )

        # ── 9. Alien bullets vs player
        p_hits = self._bullet_rect_hits(
            enemy_bullets,
            jnp.array([player_x]),
            jnp.array([jnp.float32(params.player_y)]),
            jnp.float32(7.0),
            jnp.float32(4.0),  # >speed/2=1.5 to prevent skip-through
        )  # (6, 1)
        player_hit = jnp.any(p_hits)
        lives = state.lives - player_hit.astype(jnp.int32)
        enemy_bullets = enemy_bullets.at[:, 2].set(
            enemy_bullets[:, 2] * (~p_hits[:, 0]).astype(jnp.float32)
        )

        # ── 10. Wave clear
        wave_clear = jnp.sum(enemy_grid) == jnp.int32(0)
        new_level = state.level + wave_clear.astype(jnp.int32)
        y_offset = jnp.minimum(new_level, jnp.int32(4)).astype(
            jnp.float32
        ) * jnp.float32(4.0)
        fleet_x = jnp.where(wave_clear, jnp.float32(params.fleet_x_init), fleet_x)
        fleet_y = jnp.where(
            wave_clear, jnp.float32(params.fleet_y_init) + y_offset, fleet_y
        )
        fleet_dir = jnp.where(wave_clear, jnp.int32(1), fleet_dir)
        enemy_grid = jnp.where(
            wave_clear, jnp.ones((_ROWS, _COLS), dtype=jnp.bool_), enemy_grid
        )

        # ── 11. Done conditions
        fleet_bottom = fleet_y + jnp.float32(_ROWS - 1) * jnp.float32(
            params.fleet_row_gap
        )
        done = (lives <= jnp.int32(0)) | (fleet_bottom >= jnp.float32(params.ground_y))

        return state.__replace__(
            player_x=player_x,
            fire_cooldown=fire_cooldown,
            enemy_grid=enemy_grid,
            fleet_x=fleet_x,
            fleet_y=fleet_y,
            fleet_dir=fleet_dir,
            fleet_speed=fleet_speed,
            player_bullets=player_bullets,
            enemy_bullets=enemy_bullets,
            shields=shields,
            lives=lives,
            score=state.score + delta_score,
            reward=state.reward + delta_score.astype(jnp.float32),
            level=new_level,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: SpaceInvadersState,
        action: chex.Array,
        params: SpaceInvadersParams,
    ) -> SpaceInvadersState:
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: SpaceInvadersState) -> SpaceInvadersState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: SpaceInvadersState) -> chex.Array:
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Ground line
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(80.0),
                jnp.float32(197.0),
                jnp.float32(80.0),
                jnp.float32(1.0),
            ),
            _COL_GROUND,
        )

        # Layer 2 — Shields (3 shields × 22 cells each)
        # cell_x0 shifted by -0.5 so centres land on integer pixel columns
        for s in range(3):
            shield_row = state.shields[s].reshape(1, 22)
            shield_mask = render_bool_grid(
                shield_row,
                cell_x0=_SHIELD_ANCHOR_XS[s] - 0.5,
                cell_y0=153.5,
                cell_w=_SHIELD_CELL_W,
                cell_h=_SHIELD_CELL_H,
            )
            canvas = paint_layer(canvas, shield_mask, _COL_SHIELD)

        # Layer 3 — Alien grid (per-row shapes: circle/flat/tall for squid/crab/octopus)
        # Atari 2600: 6 rows × 6 cols. Top 2=squid(30pt), mid 2=crab(20pt), bot 2=octopus(10pt).
        row_xs = state.fleet_x + _COL_OFFSETS * jnp.float32(14.0)  # (6,) x-centres
        row_ys = state.fleet_y + _ROW_OFFSETS * jnp.float32(12.0)  # (6,) y-centres

        # Rows 0-1 — squid (top, 30 pts): circles
        for _r in (0, 1):
            _rp = jnp.stack(
                [row_xs, jnp.full(_COLS, row_ys[_r]), state.enemy_grid[_r].astype(jnp.float32)],
                axis=1,
            )
            canvas = paint_layer(canvas, render_circle_pool(_rp, radius=3.5), _COL_ALIEN_TOP)

        # Rows 2-3 — crab (20 pts): wide flat rectangles
        for _r in (2, 3):
            _rp = jnp.stack(
                [row_xs, jnp.full(_COLS, row_ys[_r]), state.enemy_grid[_r].astype(jnp.float32)],
                axis=1,
            )
            canvas = paint_layer(canvas, render_rect_pool(_rp, hw=5.5, hh=1.8), _COL_ALIEN_MID)

        # Rows 4-5 — octopus (10 pts): taller rectangles
        for _r in (4, 5):
            _rp = jnp.stack(
                [row_xs, jnp.full(_COLS, row_ys[_r]), state.enemy_grid[_r].astype(jnp.float32)],
                axis=1,
            )
            canvas = paint_layer(canvas, render_rect_pool(_rp, hw=3.5, hh=3.0), _COL_ALIEN_BOT)

        # Layer 4 — Alien bullets
        canvas = paint_layer(
            canvas,
            render_rect_pool(state.enemy_bullets, hw=1.2, hh=3.0),
            _COL_ALIEN_BULLET,
        )

        # Layer 5 — Player bullets
        canvas = paint_layer(
            canvas,
            render_rect_pool(state.player_bullets, hw=1.2, hh=3.5),
            _COL_PLAYER_BULLET,
        )

        # Layer 6 — Player cannon
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.player_x, jnp.float32(185.0), jnp.float32(7.0), jnp.float32(3.0)
            ),
            _COL_PLAYER,
        )

        # Layer 7 — Lives HUD (small cannon icons bottom-left)
        for i in range(3):
            hud_x = jnp.float32(10.0 + i * 18.0)
            alive = (state.lives > jnp.int32(i)).astype(jnp.float32)
            hud_sdf = sdf_rect(
                hud_x, jnp.float32(204.0), jnp.float32(4.0), jnp.float32(2.0)
            )
            canvas = paint_layer(
                canvas,
                (hud_sdf < jnp.float32(0.0)) & (alive > jnp.float32(0.5)),
                _COL_HUD,
            )

        return finalise_rgb(canvas)
