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

"""Breakout — JAX-native SDF game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top, coordinates are ball/entity centres):
    Side walls : x ∈ [0, 8) left, x ∈ [152, 160) right — gray, full screen height
    Playfield  : x ∈ [8, 152),  y ∈ [19, 210)
    Brick area : y ∈ [57, 93),  x ∈ [8, 152)  — 6 rows × 18 cols × 8×6 px
    Paddle     : centre_y = 191,  half-width = 8,  half-height = 2
    Ball       : radius = 1 (2×2 px footprint)

Action space (4 actions):
    0 — NOOP
    1 — FIRE (serve ball when inactive)
    2 — RIGHT
    3 — LEFT
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.ball_physics import BallPhysicsGame, BallPhysicsState
from atarax.env.hud import render_life_pips, render_score
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_bool_grid,
    sdf_circle,
    sdf_rect,
)
from atarax.game import AtaraxParams

# ── Geometry
_PLAY_LEFT: float = 8.0
_PLAY_RIGHT: float = 152.0
_PLAY_TOP: float = 19.0
_PLAY_BOTTOM: float = 193.0

_BRICK_X0: float = 8.0
_BRICK_Y0: float = 57.0
_BRICK_W: float = 8.0
_BRICK_H: float = 6.0
_BRICK_ROWS: int = 6
_BRICK_COLS: int = 18

_PADDLE_Y: float = 191.0
_PADDLE_HW: float = 8.0
_PADDLE_HH: float = 2.0

_BALL_R: float = 2.0

# ── Physics
# Speed tiers (pixels per emulated frame) — indexed by state.speed_tier.
_SPEED_TIERS = jnp.array([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)

# Row score values, top row to bottom.
_ROW_SCORES = jnp.array([7, 7, 4, 4, 1, 1], dtype=jnp.int32)

# ── Colours (float32 RGB in [0, 1])
_COL_BG = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
_COL_WALL = jnp.array([0.502, 0.502, 0.502], dtype=jnp.float32)
_COL_BALL = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
_COL_PADDLE = jnp.array([0.765, 0.565, 0.239], dtype=jnp.float32)

# Per-row brick colours matching ALE Breakout (one distinct colour per row, top→bottom).
# Scores: rows 0-1 = 7pts, rows 2-3 = 4pts, rows 4-5 = 1pt.
_COL_BRICKS = jnp.array(
    [
        [0.90, 0.20, 0.20],   # Row 0: Red      (7 pts)
        [0.95, 0.55, 0.10],   # Row 1: Orange   (7 pts)
        [0.95, 0.90, 0.10],   # Row 2: Yellow   (4 pts)
        [0.25, 0.75, 0.25],   # Row 3: Green    (4 pts)
        [0.10, 0.75, 0.85],   # Row 4: Cyan     (1 pt)
        [0.30, 0.30, 0.90],   # Row 5: Blue     (1 pt)
    ],
    dtype=jnp.float32,
)


@chex.dataclass
class BreakoutParams(AtaraxParams):
    """
    Static configuration for Breakout.

    Parameters
    ----------
    max_steps : int
        Maximum agent steps per episode.
    paddle_speed : float
        Paddle movement in pixels per emulated frame.
    num_lives : int
        Lives at episode start.
    """

    max_steps: int = 10000
    paddle_speed: float = 2.0
    num_lives: int = 5


@chex.dataclass
class BreakoutState(BallPhysicsState):
    """
    Breakout game state.

    Extends `BallPhysicsState` with a speed tier tracker and death timer.

    Inherited from `BallPhysicsState`:
        `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `ball_in_play`,
        `paddle_x`, `targets` (6×18 bool brick grid).

    Inherited from `AtariState`:
        `reward`, `done`, `step`, `episode_step`, `lives`, `score`, `level`, `key`.

    Parameters
    ----------
    speed_tier : chex.Array
        int32 scalar — current speed tier index (0–3). Advances as bricks are
        cleared, keyed to bricks-remaining thresholds matching the ALE ROM.
    death_timer : chex.Array
        int32 scalar — emulated frames remaining before the player may serve
        after losing a life (~36 frames, matching ALE ROM death-animation delay).
        0 = serve is allowed.
    """

    speed_tier: chex.Array
    death_timer: chex.Array


class Breakout(BallPhysicsGame):
    """
    Breakout implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    Ball physics are delegated to `BallPhysicsGame` helpers.
    """

    num_actions: int = 4
    game_id: ClassVar[str] = "breakout"

    def _reset(self, rng: chex.PRNGKey) -> BreakoutState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : BreakoutState
            All bricks present, ball resting on the paddle, 5 lives.
        """
        return BreakoutState(
            # BallPhysicsState fields
            ball_x=jnp.float32(80.0),
            ball_y=jnp.float32(_PADDLE_Y - _PADDLE_HH - _BALL_R - 0.5),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(0.0),
            ball_in_play=jnp.bool_(False),
            paddle_x=jnp.float32(80.0),
            targets=jnp.ones((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_),
            # BreakoutState fields
            speed_tier=jnp.int32(0),
            death_timer=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(5),
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
        state: BreakoutState,
        action: chex.Array,
        params: BreakoutParams,
        rng: chex.PRNGKey,
    ) -> BreakoutState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : BreakoutState
            Current game state.
        action : chex.Array
            int32 — Action for this frame (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT).
        params : BreakoutParams
            Static environment parameters.
        rng : chex.PRNGKey
            PRNG key for serve angle randomisation.

        Returns
        -------
        new_state : BreakoutState
            State after one emulated frame.
        """
        # ── 1. Paddle movement
        move = jnp.where(
            action == jnp.int32(2),
            jnp.float32(params.paddle_speed),
            jnp.where(
                action == jnp.int32(3),
                jnp.float32(-params.paddle_speed),
                jnp.float32(0.0),
            ),
        )
        paddle_x = jnp.clip(
            state.paddle_x + move,
            jnp.float32(_PLAY_LEFT + _PADDLE_HW),
            jnp.float32(_PLAY_RIGHT - _PADDLE_HW),
        )

        # ── 2. Ball rests on paddle when inactive
        bx = jnp.where(state.ball_in_play, state.ball_x, paddle_x)
        by = jnp.where(
            state.ball_in_play,
            state.ball_y,
            jnp.float32(_PADDLE_Y - _PADDLE_HH - _BALL_R - 0.5),
        )
        vx = state.ball_vx
        vy = state.ball_vy

        # ── 3. Serve on FIRE — blocked while death animation timer is running
        fire = (action == jnp.int32(1)) & ~state.ball_in_play & (state.death_timer == jnp.int32(0))
        angle = jax.random.uniform(rng, minval=-jnp.pi / 4.0, maxval=jnp.pi / 4.0)
        tier_speed = _SPEED_TIERS[state.speed_tier]
        serve_vx = tier_speed * jnp.sin(angle)
        serve_vy = -tier_speed * jnp.cos(angle)
        ball_in_play = state.ball_in_play | fire
        vx = jnp.where(fire, serve_vx, vx)
        vy = jnp.where(fire, serve_vy, vy)

        # ── 4. Move ball (gated by active flag)
        new_bx = jnp.where(ball_in_play, bx + vx, bx)
        new_by = jnp.where(ball_in_play, by + vy, by)

        # ── 5. Wall and ceiling bounce
        new_bx, new_by, vx, vy = self._bounce_walls(
            new_bx,
            new_by,
            vx,
            vy,
            jnp.float32(_BALL_R),
            jnp.float32(_PLAY_LEFT),
            jnp.float32(_PLAY_RIGHT),
            jnp.float32(_PLAY_TOP),
        )

        # ── 6. Brick collision
        targets, delta_score, vx, vy, _ = self._grid_collide(
            new_bx,
            new_by,
            vx,
            vy,
            jnp.float32(_BALL_R),
            state.targets,
            jnp.float32(_BRICK_X0),
            jnp.float32(_BRICK_Y0),
            jnp.float32(_BRICK_W),
            jnp.float32(_BRICK_H),
            _ROW_SCORES,
        )

        # ── 7. Speed tier — bricks-remaining thresholds (matching ALE ROM):
        #   108–54 remaining → tier 0 (2 px/frame)
        #    53–38 remaining → tier 1 (3 px/frame)
        #    37–12 remaining → tier 2 (4 px/frame)
        #    11–0  remaining → tier 3 (5 px/frame)
        remaining = jnp.sum(targets).astype(jnp.int32)
        tier_from_remaining = jnp.where(
            remaining <= jnp.int32(11), jnp.int32(3),
            jnp.where(
                remaining <= jnp.int32(37), jnp.int32(2),
                jnp.where(remaining <= jnp.int32(53), jnp.int32(1), jnp.int32(0)),
            ),
        )
        new_speed_tier = jnp.maximum(state.speed_tier, tier_from_remaining)

        # Level clear: reset brick grid, increment level; tier does NOT reset
        all_cleared = ~jnp.any(targets)
        targets = jnp.where(
            all_cleared,
            jnp.ones((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_),
            targets,
        )
        new_level = state.level + jnp.where(all_cleared, jnp.int32(1), jnp.int32(0))

        # Scale velocity when the tier advances
        old_speed = _SPEED_TIERS[state.speed_tier]
        new_speed = _SPEED_TIERS[new_speed_tier]
        scale = new_speed / jnp.maximum(old_speed, jnp.float32(1e-6))
        tier_changed = (new_speed_tier != state.speed_tier) & ball_in_play
        vx = jnp.where(tier_changed, vx * scale, vx)
        vy = jnp.where(tier_changed, vy * scale, vy)

        # ── 8. Paddle reflect
        new_bx, new_by, vx, vy = self._paddle_reflect(
            new_bx,
            new_by,
            vx,
            vy,
            jnp.float32(_BALL_R),
            paddle_x,
            jnp.float32(_PADDLE_HW),
            jnp.float32(_PADDLE_HH),
            jnp.float32(_PADDLE_Y),
            new_speed,
        )

        # ── 9. Ball lost (missed the paddle)
        ball_lost = ball_in_play & (
            new_by + jnp.float32(_BALL_R) > jnp.float32(_PLAY_BOTTOM)
        )
        new_lives = state.lives - jnp.where(ball_lost, jnp.int32(1), jnp.int32(0))
        ball_in_play = ball_in_play & ~ball_lost
        new_bx = jnp.where(ball_lost, paddle_x, new_bx)
        new_by = jnp.where(
            ball_lost,
            jnp.float32(_PADDLE_Y - _PADDLE_HH - _BALL_R - 0.5),
            new_by,
        )
        vx = jnp.where(ball_lost, jnp.float32(0.0), vx)
        vy = jnp.where(ball_lost, jnp.float32(0.0), vy)

        # Death animation timer: set to 36 frames on life loss (≈ ALE ROM delay),
        # otherwise decrement toward 0 each frame.
        new_death_timer = jnp.where(
            ball_lost,
            jnp.int32(36),
            jnp.maximum(state.death_timer - jnp.int32(1), jnp.int32(0)),
        )

        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            ball_x=new_bx,
            ball_y=new_by,
            ball_vx=vx,
            ball_vy=vy,
            ball_in_play=ball_in_play,
            paddle_x=paddle_x,
            targets=targets,
            speed_tier=new_speed_tier,
            death_timer=new_death_timer,
            lives=new_lives,
            score=state.score + delta_score,
            reward=state.reward + delta_score.astype(jnp.float32),
            level=new_level,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: BreakoutState,
        action: chex.Array,
        params: BreakoutParams,
    ) -> BreakoutState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for in-step randomness.
        state : BreakoutState
            Current game state.
        action : chex.Array
            int32 — Action index.
        params : BreakoutParams
            Static environment parameters.

        Returns
        -------
        new_state : BreakoutState
            State after 4 emulated frames with `episode_step` incremented once.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: BreakoutState) -> BreakoutState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: BreakoutState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BreakoutState
            Current game state.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Cage walls — 8 px wide on all three sides (left, right, top).
        # The three segments share the same top edge at y=_PLAY_TOP and the same
        # thickness (hw/hh = 4.0 → 8 px), forming a flush rectangular frame.
        #
        #   Ceiling  : x ∈ [8, 152),  y ∈ [_PLAY_TOP, _PLAY_TOP+8)
        #              centre (80, _PLAY_TOP+4), hw=72, hh=4
        #   Left wall: x ∈ [0, 8),    y ∈ [_PLAY_TOP, _PLAY_BOTTOM]
        #              centre (4, mid), hw=4, hh=(BOTTOM-TOP)/2
        #   Right wall: x ∈ [152,160), same height
        _CAGE_HW = jnp.float32(4.0)          # half-thickness of each wall segment
        _CAGE_TOP = jnp.float32(_PLAY_TOP)
        _CAGE_BOT = jnp.float32(_PLAY_BOTTOM)
        _WALL_CY  = (_CAGE_TOP + _CAGE_BOT) * jnp.float32(0.5)
        _WALL_HH  = (_CAGE_BOT - _CAGE_TOP) * jnp.float32(0.5)
        _CEIL_CY  = _CAGE_TOP + _CAGE_HW    # ceiling centre y

        # Layer 1 — Side walls (top-aligned with ceiling)
        canvas = paint_sdf(canvas, sdf_rect(jnp.float32(4.0),   _WALL_CY, _CAGE_HW, _WALL_HH), _COL_WALL)
        canvas = paint_sdf(canvas, sdf_rect(jnp.float32(156.0), _WALL_CY, _CAGE_HW, _WALL_HH), _COL_WALL)

        # Layer 2 — Ceiling bar (same thickness, spanning play width only)
        canvas = paint_sdf(canvas, sdf_rect(jnp.float32(80.0), _CEIL_CY, jnp.float32(72.0), _CAGE_HW), _COL_WALL)

        # Layer 3 — Bricks (one pass per row; +1px overdraw eliminates the 1-px
        # background gap that strict `<` in render_rect_pool leaves at cell edges)
        for row in range(_BRICK_ROWS):
            row_grid = state.targets[row : row + 1, :]
            row_mask = render_bool_grid(
                row_grid,
                cell_x0=_BRICK_X0,
                cell_y0=_BRICK_Y0 + row * _BRICK_H,
                cell_w=_BRICK_W,
                cell_h=_BRICK_H,
                draw_w=_BRICK_W + 1.0,
                draw_h=_BRICK_H + 1.0,
            )
            canvas = paint_layer(canvas, row_mask, _COL_BRICKS[row])

        # Layer 3 — Paddle
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.paddle_x,
                jnp.float32(_PADDLE_Y),
                jnp.float32(_PADDLE_HW),
                jnp.float32(_PADDLE_HH),
            ),
            _COL_PADDLE,
        )

        # Layer 4 — Ball
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.ball_x, state.ball_y, jnp.float32(_BALL_R), jnp.float32(_BALL_R)
            ),
            _COL_BALL,
        )

        # ── HUD (top 30 px) — ball life pips + score ───────────────────────
        canvas = render_life_pips(
            canvas,
            state.lives,
            pip_sdf_fn=lambda cx, cy: sdf_circle(cx, cy, 3.5),
            pip_colour=_COL_BALL,
        )
        canvas = render_score(canvas, state.score)

        return finalise_rgb(canvas)
