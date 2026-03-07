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

"""Pong — JAX-native SDF game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Action space (6 actions, matching ALE):
    0 — NOOP
    1 — FIRE  (no effect in Pong)
    2 — RIGHT → player paddle UP
    3 — LEFT  → player paddle DOWN
    4 — RIGHTFIRE → player paddle UP
    5 — LEFTFIRE  → player paddle DOWN
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp
import numpy as np

from atarax.env._base.ball_physics import BallPhysicsGame, BallPhysicsState
from atarax.env.hud import _make_seg_masks, render_score
from atarax.env.sdf import finalise_rgb, make_canvas, paint_sdf, sdf_rect
from atarax.game import AtaraxParams

# ── Geometry
_TOP_WALL: float = 34.0
_BOT_WALL: float = 194.0

_PLAYER_X: float = 144.0  # player (right) paddle centre x
_AI_X: float = 16.0  # AI (left) paddle centre x

_PADDLE_HW: float = 3.0
_PADDLE_HH: float = 14.0
_PADDLE_Y_MIN: float = 34.0
_PADDLE_Y_MAX: float = 186.0
_PADDLE_SPEED: float = 3.5

_BALL_R: float = 3.0
_BALL_SPEED_INIT: float = 3.5
_BALL_SPEED_MAX: float = 7.0
_BALL_SPEED_INC: float = 0.1

_CENTER_X: float = 80.0
_CENTER_Y: float = 114.0

# ── Colours
_COL_BG = jnp.array([0.04, 0.04, 0.10], dtype=jnp.float32)    # deep navy
_COL_BALL = jnp.array([1.0,  0.95, 0.6], dtype=jnp.float32)   # warm yellow
_COL_AI_PADDLE = jnp.array([0.9,  0.3, 0.2], dtype=jnp.float32)    # red (AI)
_COL_PLAYER_PADDLE = jnp.array([0.2, 0.75, 1.0], dtype=jnp.float32) # cyan (player)
_COL_CENTER_LINE = jnp.array([0.25, 0.25, 0.35], dtype=jnp.float32)
_COL_SCORE = jnp.array([1.0, 0.902, 0.725], dtype=jnp.float32)

# ── Score HUD segment masks — 2 visible digits per player at y=11 ───────────
# render_score iterates 6 divisors; the first 4 are mapped to an off-screen
# x=200 so no pixels are written.  Only the tens and ones positions are shown.
# AI score sits left-of-centre; player score sits right-of-centre.
_SCORE_Y: int = 11   # matches HUD_SCORE_Y used by other games
_OFFSCREEN: int = 200  # x > 160 → all mask pixels are False

_AI_SCORE_MASKS: chex.Array = jnp.array(
    np.stack(
        [
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 100000s — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 10000s  — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 1000s   — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 100s    — hidden
            _make_seg_masks(52, _SCORE_Y),           # 10s — AI left-of-centre
            _make_seg_masks(59, _SCORE_Y),           # 1s
        ]
    )
)

_PLAYER_SCORE_MASKS: chex.Array = jnp.array(
    np.stack(
        [
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 100000s — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 10000s  — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 1000s   — hidden
            _make_seg_masks(_OFFSCREEN, _SCORE_Y),  # 100s    — hidden
            _make_seg_masks(95, _SCORE_Y),           # 10s — player right-of-centre
            _make_seg_masks(102, _SCORE_Y),          # 1s
        ]
    )
)

# Y-centres for the centre-line dashes (static, 10 segments spanning y∈[34,194])
_DASH_CYS: tuple[float, ...] = tuple(_TOP_WALL + 8.0 + i * 16.0 for i in range(10))


@chex.dataclass
class PongParams(AtaraxParams):
    """Static configuration for Pong."""

    max_steps: int = 10000


@chex.dataclass
class PongState(BallPhysicsState):
    """
    Pong game state.

    Extends `BallPhysicsState`.  The inherited `paddle_x` and `targets`
    fields are stubbed (0.0 and zeros((1,1))) since Pong uses vertical
    paddles and has no brick grid.

    Inherited from `AtariState`:
        `reward`, `done`, `step`, `episode_step`, `lives`, `score`, `level`, `key`.

    Parameters
    ----------
    player_y : chex.Array
        float32 — right paddle centre y.
    ai_y : chex.Array
        float32 — left (AI) paddle centre y.
    player_score : chex.Array
        int32 — player points (0–21).
    ai_score : chex.Array
        int32 — AI points (0–21).
    volley_count : chex.Array
        int32 — consecutive paddle hits in the current rally (drives speed up).
    """

    player_y: chex.Array
    ai_y: chex.Array
    player_score: chex.Array
    ai_score: chex.Array
    volley_count: chex.Array


class Pong(BallPhysicsGame):
    """
    Pong implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    """

    num_actions: ClassVar[int] = 6
    game_id: ClassVar[str] = "pong"

    def _reset(self, rng: chex.PRNGKey) -> PongState:
        """Return the canonical initial game state with ball served toward the player."""
        rng, k = jax.random.split(rng)
        angle = jax.random.uniform(k, minval=-jnp.pi / 6.0, maxval=jnp.pi / 6.0)
        # Serve rightward (toward player) with slight angle variance
        vx = jnp.float32(_BALL_SPEED_INIT) * jnp.cos(angle)
        vy = jnp.float32(_BALL_SPEED_INIT) * jnp.sin(angle)

        return PongState(
            # BallPhysicsState fields
            ball_x=jnp.float32(_CENTER_X),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=vx,
            ball_vy=vy,
            ball_in_play=jnp.bool_(True),  # always in play in Pong
            paddle_x=jnp.float32(0.0),  # stubbed — Pong uses player_y/ai_y
            targets=jnp.zeros((1, 1), dtype=jnp.bool_),  # stubbed — no bricks
            # PongState fields
            player_y=jnp.float32(_CENTER_Y),
            ai_y=jnp.float32(_CENTER_Y),
            player_score=jnp.int32(0),
            ai_score=jnp.int32(0),
            volley_count=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(0),
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
        state: PongState,
        action: chex.Array,
        params: PongParams,
        rng: chex.PRNGKey,
    ) -> PongState:
        """Advance the game by one emulated frame (branch-free)."""

        # ── 1. Player paddle movement
        # RIGHT (2) / RIGHTFIRE (4) → move UP (decrease y)
        # LEFT  (3) / LEFTFIRE  (5) → move DOWN (increase y)
        is_up = (action == jnp.int32(2)) | (action == jnp.int32(4))
        is_dn = (action == jnp.int32(3)) | (action == jnp.int32(5))
        dy = jnp.where(
            is_up,
            jnp.float32(-_PADDLE_SPEED),
            jnp.where(is_dn, jnp.float32(_PADDLE_SPEED), jnp.float32(0.0)),
        )
        player_y = jnp.clip(
            state.player_y + dy,
            jnp.float32(_PADDLE_Y_MIN),
            jnp.float32(_PADDLE_Y_MAX),
        )

        # ── 2. AI paddle tracking (branch-free)
        ai_dy = jnp.clip(
            (state.ball_y - state.ai_y) * jnp.float32(0.6),
            jnp.float32(-_PADDLE_SPEED),
            jnp.float32(_PADDLE_SPEED),
        )
        ai_y = jnp.clip(
            state.ai_y + ai_dy,
            jnp.float32(_PADDLE_Y_MIN),
            jnp.float32(_PADDLE_Y_MAX),
        )

        # ── 3. Ball movement
        bx = state.ball_x + state.ball_vx
        by = state.ball_y + state.ball_vy
        vx = state.ball_vx
        vy = state.ball_vy
        volley_count = state.volley_count

        # ── 4. Top / bottom wall bounce
        hit_top = by - jnp.float32(_BALL_R) < jnp.float32(_TOP_WALL)
        hit_bot = by + jnp.float32(_BALL_R) > jnp.float32(_BOT_WALL)
        vy = jnp.where(hit_top, jnp.abs(vy), vy)
        vy = jnp.where(hit_bot, -jnp.abs(vy), vy)
        by = jnp.clip(
            by,
            jnp.float32(_TOP_WALL + _BALL_R),
            jnp.float32(_BOT_WALL - _BALL_R),
        )

        # ── 5. Player paddle (right) bounce
        # AABB test: ball must be within (PADDLE_HW + BALL_R) of paddle centre in
        # both axes.  The vx > 0 guard prevents double-bouncing after contact.
        hit_player = (
            (vx > jnp.float32(0.0))
            & (jnp.abs(bx - jnp.float32(_PLAYER_X)) < jnp.float32(_PADDLE_HW + _BALL_R))
            & (jnp.abs(by - player_y) < jnp.float32(_PADDLE_HH + _BALL_R))
        )
        curr_spd = jnp.sqrt(vx**2 + vy**2)
        new_spd_player = jnp.minimum(
            curr_spd + jnp.float32(_BALL_SPEED_INC),
            jnp.float32(_BALL_SPEED_MAX),
        )
        rel_p = (by - player_y) / jnp.float32(_PADDLE_HH)
        p_vy = rel_p * new_spd_player * jnp.float32(0.8)
        p_vx = -jnp.sqrt(jnp.maximum(new_spd_player**2 - p_vy**2, jnp.float32(0.01)))
        bx = jnp.where(hit_player, jnp.float32(_PLAYER_X - _PADDLE_HW - _BALL_R), bx)
        vx = jnp.where(hit_player, p_vx, vx)
        vy = jnp.where(hit_player, p_vy, vy)
        volley_count = jnp.where(hit_player, volley_count + jnp.int32(1), volley_count)

        # ── 6. AI paddle (left) bounce
        # AABB test: same symmetric test for the AI (left) paddle.
        hit_ai = (
            (vx < jnp.float32(0.0))
            & (jnp.abs(bx - jnp.float32(_AI_X)) < jnp.float32(_PADDLE_HW + _BALL_R))
            & (jnp.abs(by - ai_y) < jnp.float32(_PADDLE_HH + _BALL_R))
        )
        new_spd_ai = jnp.minimum(
            jnp.sqrt(vx**2 + vy**2) + jnp.float32(_BALL_SPEED_INC),
            jnp.float32(_BALL_SPEED_MAX),
        )
        rel_a = (by - ai_y) / jnp.float32(_PADDLE_HH)
        a_vy = rel_a * new_spd_ai * jnp.float32(0.8)
        a_vx = jnp.sqrt(jnp.maximum(new_spd_ai**2 - a_vy**2, jnp.float32(0.01)))
        bx = jnp.where(hit_ai, jnp.float32(_AI_X + _PADDLE_HW + _BALL_R), bx)
        vx = jnp.where(hit_ai, a_vx, vx)
        vy = jnp.where(hit_ai, a_vy, vy)
        volley_count = jnp.where(hit_ai, volley_count + jnp.int32(1), volley_count)

        # ── 7. Scoring — ball exits left or right
        ai_scored = bx + jnp.float32(_BALL_R) > jnp.float32(160.0)  # player missed
        player_scored = bx - jnp.float32(_BALL_R) < jnp.float32(0.0)  # AI missed

        new_player_score = state.player_score + jnp.where(
            player_scored, jnp.int32(1), jnp.int32(0)
        )
        new_ai_score = state.ai_score + jnp.where(ai_scored, jnp.int32(1), jnp.int32(0))
        step_reward = jnp.where(
            player_scored,
            jnp.float32(1.0),
            jnp.where(ai_scored, jnp.float32(-1.0), jnp.float32(0.0)),
        )

        # After scoring: reset ball to centre; serve toward loser's opponent
        scored = player_scored | ai_scored
        serve_angle = jax.random.uniform(rng, minval=-jnp.pi / 6.0, maxval=jnp.pi / 6.0)
        # Serve toward the player who just scored (so the loser has to react)
        serve_dir = jnp.where(player_scored, jnp.float32(1.0), jnp.float32(-1.0))
        serve_vx = serve_dir * jnp.float32(_BALL_SPEED_INIT) * jnp.cos(serve_angle)
        serve_vy = jnp.float32(_BALL_SPEED_INIT) * jnp.sin(serve_angle)

        bx = jnp.where(scored, jnp.float32(_CENTER_X), bx)
        by = jnp.where(scored, jnp.float32(_CENTER_Y), by)
        vx = jnp.where(scored, serve_vx, vx)
        vy = jnp.where(scored, serve_vy, vy)
        volley_count = jnp.where(scored, jnp.int32(0), volley_count)

        # ── 8. Done
        done = (new_player_score >= jnp.int32(21)) | (new_ai_score >= jnp.int32(21))

        return state.__replace__(
            ball_x=bx,
            ball_y=by,
            ball_vx=vx,
            ball_vy=vy,
            player_y=player_y,
            ai_y=ai_y,
            player_score=new_player_score,
            ai_score=new_ai_score,
            volley_count=volley_count,
            score=state.score + jnp.where(player_scored, jnp.int32(1), jnp.int32(0)),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: PongState,
        action: chex.Array,
        params: PongParams,
    ) -> PongState:
        """Advance the game by one agent step (4 emulated frames)."""
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: PongState) -> PongState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: PongState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Centre dashed line (10 evenly spaced segments)
        for cy in _DASH_CYS:
            canvas = paint_sdf(
                canvas,
                sdf_rect(
                    jnp.float32(80.0),
                    jnp.float32(cy),
                    jnp.float32(1.5),
                    jnp.float32(5.0),
                ),
                _COL_CENTER_LINE,
            )

        # Layer 2 — AI paddle (left)
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(_AI_X),
                state.ai_y,
                jnp.float32(_PADDLE_HW),
                jnp.float32(_PADDLE_HH),
            ),
            _COL_AI_PADDLE,
        )

        # Layer 3 — Player paddle (right)
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(_PLAYER_X),
                state.player_y,
                jnp.float32(_PADDLE_HW),
                jnp.float32(_PADDLE_HH),
            ),
            _COL_PLAYER_PADDLE,
        )

        # Layer 4 — Ball (square, matching ALE Pong pixel art)
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.ball_x,
                state.ball_y,
                jnp.float32(_BALL_R),
                jnp.float32(_BALL_R),
            ),
            _COL_BALL,
        )

        # Layer 5 — Score HUD: AI score in red (left), player score in cyan (right)
        canvas = render_score(
            canvas, state.ai_score, colour=_COL_AI_PADDLE, seg_masks=_AI_SCORE_MASKS
        )
        canvas = render_score(
            canvas, state.player_score, colour=_COL_PLAYER_PADDLE, seg_masks=_PLAYER_SCORE_MASKS
        )

        return finalise_rgb(canvas)
