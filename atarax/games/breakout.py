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

"""Breakout — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Playfield  : x ∈ [8, 152),  y ∈ [19, 210)
    Brick area : y ∈ [57, 93),  x ∈ [8, 152)  — 6 rows × 18 cols × 8×6 px
    Paddle     : y = 189,  16 × 4 px
    Ball       : 2 × 2 px

Action space (4 actions):
    0 — NOOP
    1 — FIRE  (serve ball when inactive)
    2 — RIGHT
    3 — LEFT
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.mechanics import grid_hit_test, grid_remove_hit
from atarax.state import AtariState

# Geometry constants
_PLAY_LEFT: int = 8
_PLAY_RIGHT: int = 152
_PLAY_TOP: int = 19
_PLAY_BOTTOM: int = 210

_BRICK_Y0: int = 57
_BRICK_X0: int = 8
_BRICK_ROWS: int = 6
_BRICK_COLS: int = 18
_BRICK_H: int = 6
_BRICK_W: int = 8

_PADDLE_Y: int = 189
_PADDLE_H: int = 4
_PADDLE_W: int = 16
_PADDLE_SPEED: float = 4.0

_BALL_SIZE: int = 2
_FRAME_SKIP: int = 4

# Ball speed tiers (px/emulated frame). Tier rises on upper-row hits + board clears.
_SPEED_TIERS = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

# Row point values (row 0 = top/hardest → highest score)
_ROW_SCORES = jnp.array([7, 7, 4, 4, 1, 1], dtype=jnp.float32)

# Brick RGB colours by row — NTSC-accurate (top → bottom: red, orange, green)
_BRICK_COLORS = jnp.array(
    [
        [200, 72, 72],  # rows 0-1 — red
        [200, 72, 72],
        [195, 144, 61],  # rows 2-3 — orange
        [195, 144, 61],
        [92, 186, 92],  # rows 4-5 — green
        [92, 186, 92],
    ],
    dtype=jnp.uint8,
)

_PADDLE_COLOR = jnp.array([195, 144, 61], dtype=jnp.uint8)
_SCORE_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

# 3×5 bitmap font for digits 0–9.  Shape: [10, 5, 3].
_DIGIT_FONT = jnp.array(
    [
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
        [[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 4
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 5
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 6
        [[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 7
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 8
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 9
    ],
    dtype=jnp.bool_,
)

# Precomputed scanline arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # (210, 1)
_COL_IDX = jnp.arange(160)[None, :]  # (1, 160)


def _blit_digit(frame: jax.Array, digit: jax.Array, x0: int, y0: int) -> jax.Array:
    """
    Blit a single 3×5 digit glyph onto *frame* at pixel (x0, y0).

    Parameters
    ----------
    frame : jax.Array
        uint8[210, 160, 3] — Frame to draw onto.
    digit : jax.Array
        int32 scalar — Digit value 0–9.
    x0 : int
        Left edge x coordinate.
    y0 : int
        Top edge y coordinate.

    Returns
    -------
    frame : jax.Array
        uint8[210, 160, 3] — Updated frame.
    """
    glyph = _DIGIT_FONT[digit]  # bool[5, 3]
    dr = jnp.clip(_ROW_IDX - y0, 0, 4)
    dc = jnp.clip(_COL_IDX - x0, 0, 2)
    in_box = (
        (_ROW_IDX >= y0) & (_ROW_IDX < y0 + 5) & (_COL_IDX >= x0) & (_COL_IDX < x0 + 3)
    )
    lit = glyph[dr, dc]
    return jnp.where((in_box & lit)[:, :, None], _SCORE_COLOR, frame)


@chex.dataclass
class BreakoutState(AtariState):
    """
    Complete Breakout game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    Parameters
    ----------
    bricks : jax.Array
        bool[6, 18] — Active bricks. `True` = brick present.
    ball_x : jax.Array
        float32 — Ball left-edge x coordinate.
    ball_y : jax.Array
        float32 — Ball top-edge y coordinate.
    ball_dx : jax.Array
        float32 — Ball x velocity (px per emulated frame).
    ball_dy : jax.Array
        float32 — Ball y velocity (negative = moving up).
    ball_active : jax.Array
        bool — `False` while waiting for FIRE to serve the ball.
    paddle_x : jax.Array
        float32 — Paddle left-edge x coordinate.
    speed_tier : jax.Array
        int32 — Current speed tier (0–3). Rises on upper-row hits and board clears.
    """

    bricks: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_dy: jax.Array
    ball_active: jax.Array
    paddle_x: jax.Array
    speed_tier: jax.Array


class Breakout(AtaraxGame):
    """
    Breakout implemented as a pure-JAX function suite.

    Physics: ball bounces off walls, bricks, and paddle. Bricks award
    row-dependent points. Speed increases when hitting upper rows or clearing
    the board.
    """

    num_actions: int = 4

    def _reset(self, key: chex.PRNGKey) -> BreakoutState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : BreakoutState
            All bricks present, ball inactive above the paddle, 5 lives.
        """
        paddle_x = jnp.float32((_PLAY_LEFT + _PLAY_RIGHT) / 2 - _PADDLE_W / 2)
        return BreakoutState(
            bricks=jnp.ones((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_),
            ball_x=paddle_x + jnp.float32(_PADDLE_W // 2 - _BALL_SIZE // 2),
            ball_y=jnp.float32(_PADDLE_Y - _BALL_SIZE - 1),
            ball_dx=jnp.float32(0.0),
            ball_dy=jnp.float32(0.0),
            ball_active=jnp.bool_(False),
            paddle_x=paddle_x,
            speed_tier=jnp.int32(0),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: BreakoutState, action: jax.Array) -> BreakoutState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : BreakoutState
            Current game state.
        action : jax.Array
            int32 — Action for this frame (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT).

        Returns
        -------
        new_state : BreakoutState
            State after one emulated frame. `episode_step` is NOT incremented
            here — it is incremented once per agent step in `_step`.
        """
        key, subkey = jax.random.split(state.key)

        # Paddle movement
        move = jnp.where(
            action == 2,
            jnp.float32(_PADDLE_SPEED),
            jnp.where(action == 3, jnp.float32(-_PADDLE_SPEED), jnp.float32(0.0)),
        )
        paddle_x = jnp.clip(
            state.paddle_x + move,
            jnp.float32(_PLAY_LEFT),
            jnp.float32(_PLAY_RIGHT - _PADDLE_W),
        )

        # Ball follows paddle when inactive
        ball_x = jnp.where(
            state.ball_active,
            state.ball_x,
            paddle_x + jnp.float32(_PADDLE_W // 2 - _BALL_SIZE // 2),
        )
        ball_y = jnp.where(
            state.ball_active, state.ball_y, jnp.float32(_PADDLE_Y - _BALL_SIZE - 1)
        )
        ball_dx = state.ball_dx
        ball_dy = state.ball_dy

        # Serve on FIRE
        fire = (action == 1) & ~state.ball_active
        angle = jax.random.uniform(subkey, minval=-jnp.pi / 4, maxval=jnp.pi / 4)
        serve_speed = _SPEED_TIERS[state.speed_tier]
        serve_dx = serve_speed * jnp.sin(angle)
        serve_dy = -serve_speed
        ball_active = state.ball_active | fire
        ball_dx = jnp.where(fire, serve_dx, ball_dx)
        ball_dy = jnp.where(fire, serve_dy, ball_dy)

        # Move ball (only when active)
        new_x = jnp.where(ball_active, ball_x + ball_dx, ball_x)
        new_y = jnp.where(ball_active, ball_y + ball_dy, ball_y)

        # Wall reflections
        left_hit = new_x < jnp.float32(_PLAY_LEFT)
        new_x = jnp.where(left_hit, 2.0 * _PLAY_LEFT - new_x, new_x)
        ball_dx = jnp.where(left_hit, -ball_dx, ball_dx)

        right_hit = new_x + _BALL_SIZE > jnp.float32(_PLAY_RIGHT)
        new_x = jnp.where(right_hit, 2.0 * (_PLAY_RIGHT - _BALL_SIZE) - new_x, new_x)
        ball_dx = jnp.where(right_hit, -ball_dx, ball_dx)

        top_hit = new_y < jnp.float32(_PLAY_TOP)
        new_y = jnp.where(top_hit, 2.0 * _PLAY_TOP - new_y, new_y)
        ball_dy = jnp.where(top_hit, -ball_dy, ball_dy)

        # Brick collision
        hit_mask, any_brick_hit = grid_hit_test(
            new_x,
            new_y,
            jnp.float32(_BALL_SIZE),
            jnp.float32(_BALL_SIZE),
            state.bricks,
            jnp.float32(_BRICK_Y0),
            jnp.float32(_BRICK_X0),
            jnp.float32(_BRICK_H),
            jnp.float32(_BRICK_W),
        )
        new_bricks = grid_remove_hit(state.bricks, hit_mask)
        step_reward = jnp.sum(hit_mask * _ROW_SCORES[:, None])
        ball_dy = jnp.where(any_brick_hit & ball_active, -ball_dy, ball_dy)

        # Speed tier: rise on upper-row (rows 0-1) brick hit
        hit_any_upper = jnp.any(hit_mask[:2, :])
        new_speed_tier = jnp.where(
            any_brick_hit & ball_active & hit_any_upper,
            jnp.maximum(state.speed_tier, jnp.int32(1)),
            state.speed_tier,
        )

        # Level clear: reset bricks; increment level and speed tier
        all_cleared = ~jnp.any(new_bricks)
        new_bricks = jnp.where(
            all_cleared,
            jnp.ones((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_),
            new_bricks,
        )
        new_level = state.level + jnp.where(all_cleared, jnp.int32(1), jnp.int32(0))
        new_speed_tier = jnp.minimum(
            new_speed_tier + jnp.where(all_cleared, jnp.int32(1), jnp.int32(0)),
            jnp.int32(3),
        )

        # Scale ball velocity when tier advances
        scale = _SPEED_TIERS[new_speed_tier] / _SPEED_TIERS[state.speed_tier]
        tier_up = (new_speed_tier != state.speed_tier) & ball_active
        ball_dx = jnp.where(tier_up, ball_dx * scale, ball_dx)
        ball_dy = jnp.where(tier_up, ball_dy * scale, ball_dy)

        # Paddle collision
        ball_bottom = new_y + jnp.float32(_BALL_SIZE)
        paddle_right = paddle_x + jnp.float32(_PADDLE_W)
        paddle_hit = (
            ball_active
            & (ball_bottom >= jnp.float32(_PADDLE_Y))
            & (new_y < jnp.float32(_PADDLE_Y + _PADDLE_H))
            & (new_x + _BALL_SIZE > paddle_x)
            & (new_x < paddle_right)
        )
        ball_dy = jnp.where(paddle_hit, -jnp.abs(ball_dy), ball_dy)
        hit_rel = jnp.clip(
            (new_x + _BALL_SIZE / 2.0 - paddle_x) / jnp.float32(_PADDLE_W),
            0.0,
            1.0,
        )
        spin_dx = _SPEED_TIERS[new_speed_tier] * (hit_rel * 2.0 - 1.0)
        ball_dx = jnp.where(paddle_hit, spin_dx, ball_dx)
        new_y = jnp.where(paddle_hit, jnp.float32(_PADDLE_Y - _BALL_SIZE - 0.5), new_y)

        # Ball out of bounds (missed)
        ball_oob = ball_active & (new_y > jnp.float32(_PLAY_BOTTOM))
        new_lives = state.lives - jnp.where(ball_oob, jnp.int32(1), jnp.int32(0))
        ball_active = jnp.where(ball_oob, jnp.bool_(False), ball_active)
        new_x = jnp.where(
            ball_oob,
            paddle_x + jnp.float32(_PADDLE_W // 2 - _BALL_SIZE // 2),
            new_x,
        )
        new_y = jnp.where(ball_oob, jnp.float32(_PADDLE_Y - _BALL_SIZE - 1), new_y)

        done = new_lives <= jnp.int32(0)

        return BreakoutState(
            bricks=new_bricks,
            ball_x=new_x,
            ball_y=new_y,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            ball_active=ball_active,
            paddle_x=paddle_x,
            speed_tier=new_speed_tier,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            level=new_level,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step,  # incremented once per agent step in _step
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: BreakoutState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> BreakoutState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for in-step randomness.
        state : BreakoutState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT).
        params : AtaraxParams
            Static environment parameters.

        Returns
        -------
        new_state : BreakoutState
            State after 4 emulated frames with `episode_step` incremented once.
        """
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: BreakoutState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BreakoutState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Bricks
        brick_pixel_mask = jnp.repeat(
            jnp.repeat(state.bricks, _BRICK_H, axis=0), _BRICK_W, axis=1
        )  # bool[36, 144]
        row_colors_exp = jnp.broadcast_to(
            _BRICK_COLORS[:, None, None, :],
            (_BRICK_ROWS, _BRICK_H, _BRICK_COLS * _BRICK_W, 3),
        ).reshape(_BRICK_ROWS * _BRICK_H, _BRICK_COLS * _BRICK_W, 3)  # [36, 144, 3]
        brick_pixels = (brick_pixel_mask[:, :, None] * row_colors_exp).astype(jnp.uint8)
        frame = frame.at[
            _BRICK_Y0 : _BRICK_Y0 + _BRICK_ROWS * _BRICK_H,
            _BRICK_X0 : _BRICK_X0 + _BRICK_COLS * _BRICK_W,
        ].set(brick_pixels)

        # Ball
        bx = jnp.int32(state.ball_x)
        by = jnp.int32(state.ball_y)
        ball_mask = (
            (_ROW_IDX >= by)
            & (_ROW_IDX < by + _BALL_SIZE)
            & (_COL_IDX >= bx)
            & (_COL_IDX < bx + _BALL_SIZE)
        )
        frame = jnp.where(ball_mask[:, :, None], _SCORE_COLOR, frame)

        # Paddle
        px = jnp.int32(state.paddle_x)
        paddle_mask = (
            (_ROW_IDX >= _PADDLE_Y)
            & (_ROW_IDX < _PADDLE_Y + _PADDLE_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _PADDLE_W)
        )
        frame = jnp.where(paddle_mask[:, :, None], _PADDLE_COLOR, frame)

        # Score (4-digit)
        score = state.score
        frame = _blit_digit(frame, (score // 1000) % 10, x0=8, y0=4)
        frame = _blit_digit(frame, (score // 100) % 10, x0=12, y0=4)
        frame = _blit_digit(frame, (score // 10) % 10, x0=16, y0=4)
        frame = _blit_digit(frame, score % 10, x0=20, y0=4)

        # Life indicators (4×4 squares, max 5)
        for i in range(5):
            life_x = 96 + i * 6
            life_mask = (
                (_ROW_IDX >= 4)
                & (_ROW_IDX < 8)
                & (_COL_IDX >= life_x)
                & (_COL_IDX < life_x + 4)
                & (state.lives > jnp.int32(i))
            )
            frame = jnp.where(life_mask[:, :, None], _SCORE_COLOR, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Breakout action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
        }
