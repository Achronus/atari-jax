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

Mechanics implemented directly in JAX with no hardware emulation.
All conditionals use `jnp.where`; the step loop uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Playfield  : x ∈ [8, 152),  y ∈ [19, 210)
    Brick area : y ∈ [57, 93),  x ∈ [8, 152)  — 6 rows × 18 cols
    Paddle     : y = 189,  height 4 px,  width 16 px
    Ball       : 2 × 2 px

Action space (minimal, 4 actions):
    0 — NOOP
    1 — FIRE  (launch ball when inactive)
    2 — RIGHT
    3 — LEFT
"""

import chex
import jax
import jax.numpy as jnp

from atarax.core.mechanics import grid_hit_test
from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
_PLAY_LEFT: int = 8
_PLAY_RIGHT: int = 152
_PLAY_TOP: int = 19
_PLAY_BOTTOM: int = 205

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
_BALL_SPEED: float = 2.0  # pixels per emulated frame (× 4 = 8 px per step)
_FRAME_SKIP: int = 4

# Row point values (row 0 = top, hardest to reach → highest score)
_ROW_SCORES = jnp.array([7, 7, 4, 4, 1, 1], dtype=jnp.float32)

# Brick RGB colours by row (top → bottom: red, orange, yellow, green, blue, purple)
_BRICK_COLORS = jnp.array(
    [
        [200, 72, 72],
        [198, 108, 58],
        [180, 122, 48],
        [162, 162, 42],
        [72, 160, 72],
        [72, 88, 200],
    ],
    dtype=jnp.uint8,
)

# Precomputed scanline and column index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]


@chex.dataclass
class BreakoutState(AtariState):
    """
    Complete Breakout game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, and `episode_step` from `GameState`,
    and `lives` and `score` from `AtariState`.

    Parameters
    ----------
    bricks : jax.Array
        bool[6, 18] — Active bricks.  `True` = brick present.
    ball_x : jax.Array
        float32 — Ball left-edge x coordinate.
    ball_y : jax.Array
        float32 — Ball top-edge y coordinate.
    ball_dx : jax.Array
        float32 — Ball x velocity (pixels per emulated frame).
    ball_dy : jax.Array
        float32 — Ball y velocity (negative = moving up).
    ball_active : jax.Array
        bool — `False` while waiting for a FIRE action to serve the ball.
    paddle_x : jax.Array
        float32 — Paddle left-edge x coordinate.
    key : jax.Array
        uint32[2] — PRNG key, evolved each frame for stochastic serve angles.
    """

    bricks: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_dy: jax.Array
    ball_active: jax.Array
    paddle_x: jax.Array
    key: jax.Array


class Breakout(AtariEnv):
    """
    Breakout implemented as a pure JAX function suite.

    No hardware emulation — game physics are computed directly using
    `jnp.where` for all conditionals and `jax.lax.fori_loop` for the
    4-frame skip inside `_step`.
    """

    num_actions: int = 4

    def _reset(self, key: jax.Array) -> BreakoutState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BreakoutState
            Initial state with all bricks present, ball inactive above the
            paddle, and 5 lives.
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
            lives=jnp.int32(5),
            score=jnp.int32(0),
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
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Paddle movement ---
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

        # --- Ball follows paddle when inactive ---
        ball_x = jnp.where(
            state.ball_active,
            state.ball_x,
            paddle_x + jnp.float32(_PADDLE_W // 2 - _BALL_SIZE // 2),
        )
        ball_y = jnp.where(
            state.ball_active,
            state.ball_y,
            jnp.float32(_PADDLE_Y - _BALL_SIZE - 1),
        )
        ball_dx = state.ball_dx
        ball_dy = state.ball_dy

        # --- Serve on FIRE ---
        fire = (action == 1) & ~state.ball_active
        angle = jax.random.uniform(subkey, minval=-jnp.pi / 4, maxval=jnp.pi / 4)
        serve_dx = jnp.float32(_BALL_SPEED) * jnp.sin(angle)
        serve_dy = jnp.float32(-_BALL_SPEED)
        ball_active = state.ball_active | fire
        ball_dx = jnp.where(fire, serve_dx, ball_dx)
        ball_dy = jnp.where(fire, serve_dy, ball_dy)

        # --- Move ball (only when active) ---
        new_x = jnp.where(ball_active, ball_x + ball_dx, ball_x)
        new_y = jnp.where(ball_active, ball_y + ball_dy, ball_y)

        # --- Wall reflections ---
        # Left wall
        left_hit = new_x < jnp.float32(_PLAY_LEFT)
        new_x = jnp.where(left_hit, 2.0 * _PLAY_LEFT - new_x, new_x)
        ball_dx = jnp.where(left_hit, -ball_dx, ball_dx)

        # Right wall (ball right edge > PLAY_RIGHT)
        right_hit = new_x + _BALL_SIZE > jnp.float32(_PLAY_RIGHT)
        new_x = jnp.where(right_hit, 2.0 * (_PLAY_RIGHT - _BALL_SIZE) - new_x, new_x)
        ball_dx = jnp.where(right_hit, -ball_dx, ball_dx)

        # Top wall
        top_hit = new_y < jnp.float32(_PLAY_TOP)
        new_y = jnp.where(top_hit, 2.0 * _PLAY_TOP - new_y, new_y)
        ball_dy = jnp.where(top_hit, -ball_dy, ball_dy)

        # --- Brick collision ---
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
        new_bricks = state.bricks & ~hit_mask
        step_reward = jnp.sum(hit_mask * _ROW_SCORES[:, None])
        ball_dy = jnp.where(any_brick_hit & ball_active, -ball_dy, ball_dy)

        # Level clear: reset brick grid (ball and lives unchanged)
        all_cleared = ~jnp.any(new_bricks)
        new_bricks = jnp.where(
            all_cleared,
            jnp.ones((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_),
            new_bricks,
        )

        # --- Paddle collision ---
        ball_right = new_x + jnp.float32(_BALL_SIZE)
        ball_bottom = new_y + jnp.float32(_BALL_SIZE)
        paddle_right = paddle_x + jnp.float32(_PADDLE_W)

        paddle_hit = (
            ball_active
            & (ball_bottom >= jnp.float32(_PADDLE_Y))
            & (new_y < jnp.float32(_PADDLE_Y + _PADDLE_H))
            & (ball_right > paddle_x)
            & (new_x < paddle_right)
        )
        # Reflect and apply spin based on contact point
        ball_dy = jnp.where(paddle_hit, -jnp.abs(ball_dy), ball_dy)
        hit_rel = jnp.clip(
            (new_x + _BALL_SIZE / 2.0 - paddle_x) / jnp.float32(_PADDLE_W),
            0.0,
            1.0,
        )
        spin_dx = jnp.float32(_BALL_SPEED) * (hit_rel * 2.0 - 1.0)
        ball_dx = jnp.where(paddle_hit, spin_dx, ball_dx)
        # Push ball above paddle surface on bounce
        new_y = jnp.where(
            paddle_hit,
            jnp.float32(_PADDLE_Y - _BALL_SIZE - 0.5),
            new_y,
        )

        # --- Ball out of bounds ---
        ball_oob = ball_active & (new_y > jnp.float32(_PLAY_BOTTOM))
        new_lives = state.lives - jnp.where(ball_oob, jnp.int32(1), jnp.int32(0))
        ball_active = jnp.where(ball_oob, jnp.bool_(False), ball_active)
        new_x = jnp.where(
            ball_oob,
            paddle_x + jnp.float32(_PADDLE_W // 2 - _BALL_SIZE // 2),
            new_x,
        )
        new_y = jnp.where(ball_oob, jnp.float32(_PADDLE_Y - _BALL_SIZE - 1), new_y)

        # --- Episode termination ---
        done = new_lives <= jnp.int32(0)

        return BreakoutState(
            bricks=new_bricks,
            ball_x=new_x,
            ball_y=new_y,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            ball_active=ball_active,
            paddle_x=paddle_x,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: BreakoutState, action: jax.Array) -> BreakoutState:
        """
        Advance the game by one agent step (4 emulated frames).

        The reward is accumulated across all 4 frames, matching the ALE
        frame-skip convention.

        Parameters
        ----------
        state : BreakoutState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT).

        Returns
        -------
        new_state : BreakoutState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BreakoutState) -> BreakoutState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, _FRAME_SKIP, body, state)

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
        # --- Background ---
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Bricks ---
        # Expand bricks bool[6,18] → pixel grid [6*BRICK_H, 18*BRICK_W]
        brick_pixel_mask = jnp.repeat(
            jnp.repeat(state.bricks, _BRICK_H, axis=0), _BRICK_W, axis=1
        )  # bool[36, 144]

        # Expand row colours to match pixel grid: [6, BRICK_H, BRICK_COLS*BRICK_W, 3]
        row_colors_exp = jnp.broadcast_to(
            _BRICK_COLORS[:, None, None, :],
            (_BRICK_ROWS, _BRICK_H, _BRICK_COLS * _BRICK_W, 3),
        ).reshape(_BRICK_ROWS * _BRICK_H, _BRICK_COLS * _BRICK_W, 3)  # [36, 144, 3]

        brick_pixels = (brick_pixel_mask[:, :, None] * row_colors_exp).astype(jnp.uint8)

        frame = frame.at[
            _BRICK_Y0 : _BRICK_Y0 + _BRICK_ROWS * _BRICK_H,
            _BRICK_X0 : _BRICK_X0 + _BRICK_COLS * _BRICK_W,
        ].set(brick_pixels)

        # --- Ball ---
        bx = jnp.int32(state.ball_x)
        by = jnp.int32(state.ball_y)
        ball_mask = (
            (_ROW_IDX >= by)
            & (_ROW_IDX < by + _BALL_SIZE)
            & (_COL_IDX >= bx)
            & (_COL_IDX < bx + _BALL_SIZE)
        )  # bool[210, 160] — position is always valid; show ball even when awaiting serve
        frame = jnp.where(ball_mask[:, :, None], jnp.uint8(255), frame)

        # --- Paddle ---
        px = jnp.int32(state.paddle_x)
        paddle_mask = (
            (_ROW_IDX >= _PADDLE_Y)
            & (_ROW_IDX < _PADDLE_Y + _PADDLE_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _PADDLE_W)
        )  # bool[210, 160]
        frame = jnp.where(paddle_mask[:, :, None], jnp.uint8(200), frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Breakout action indices.
            Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
        }
