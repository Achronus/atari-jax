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

"""Pong — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Court        : x ∈ [8, 152),  y ∈ [34, 194)
    CPU paddle   : x ∈ [8, 12),   h=16 px,  y ∈ [34, 178)
    Player paddle: x ∈ [148, 152), h=16 px,  y ∈ [34, 178)
    Ball         : 2 × 4 px
    Net          : x = 80, dashed every 4 px

Action space (6 actions — matches ALE minimal action set):
    0 — NOOP
    1 — FIRE         (no effect in Pong)
    2 — RIGHT        → move player paddle UP
    3 — LEFT         → move player paddle DOWN
    4 — RIGHTFIRE    → move player paddle UP   (same as RIGHT)
    5 — LEFTFIRE     → move player paddle DOWN (same as LEFT)
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry constants
_COURT_TOP: int = 34
_COURT_BOTTOM: int = 194
_COURT_LEFT: int = 8
_COURT_RIGHT: int = 152

_CPU_X: int = 8
_PLAYER_X: int = 148
_PADDLE_W: int = 4
_PADDLE_H: int = 16
_PADDLE_SPEED: float = 2.0
_CPU_SPEED: float = 1.5

_BALL_W: int = 2
_BALL_H: int = 4
_BALL_SPEED: float = 2.0

_NET_X: int = 80
_WIN_SCORE: int = 21
_FRAME_SKIP: int = 4

_CENTRE_X: float = (_COURT_LEFT + _COURT_RIGHT - _BALL_W) / 2.0
_CENTRE_Y: float = (_COURT_TOP + _COURT_BOTTOM - _BALL_H) / 2.0
_PADDLE_INIT_Y: float = (_COURT_TOP + _COURT_BOTTOM - _PADDLE_H) / 2.0

# Precomputed scanline arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # (210, 1)
_COL_IDX = jnp.arange(160)[None, :]  # (1, 160)


@chex.dataclass
class PongState(AtariState):
    """
    Complete Pong game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `score` holds the player's point total; `opp_score` the CPU's. `level`
    stays `0` for Pong (no board cycling).  `ball_active=False` means the
    ball is paused at court centre; the auto-serve fires on the next
    emulated sub-step.

    Parameters
    ----------
    ball_x : jax.Array
        float32 — Ball left-edge x coordinate.
    ball_y : jax.Array
        float32 — Ball top-edge y coordinate.
    ball_dx : jax.Array
        float32 — Ball x velocity (positive = rightward).
    ball_dy : jax.Array
        float32 — Ball y velocity (positive = downward).
    player_y : jax.Array
        float32 — Player paddle top-edge y (right side of court).
    cpu_y : jax.Array
        float32 — CPU paddle top-edge y (left side of court).
    opp_score : jax.Array
        int32 — CPU point total.
    ball_active : jax.Array
        bool — `False` during the one-frame pause after each point.
    """

    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_dy: jax.Array
    player_y: jax.Array
    cpu_y: jax.Array
    opp_score: jax.Array
    ball_active: jax.Array


class Pong(AtaraxGame):
    """
    Pong implemented as a pure-JAX function suite.

    The CPU opponent tracks the ball centre at `_CPU_SPEED` px/sub-step
    (1.5 px), making it beatable but non-trivial.
    """

    num_actions: int = 6

    def _reset(self, key: chex.PRNGKey) -> PongState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : PongState
            Both paddles centred, ball inactive at court centre, scores zero.
        """
        return PongState(
            ball_x=jnp.float32(_CENTRE_X),
            ball_y=jnp.float32(_CENTRE_Y),
            ball_dx=jnp.float32(0.0),
            ball_dy=jnp.float32(0.0),
            player_y=jnp.float32(_PADDLE_INIT_Y),
            cpu_y=jnp.float32(_PADDLE_INIT_Y),
            opp_score=jnp.int32(0),
            ball_active=jnp.bool_(False),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: PongState, action: jax.Array) -> PongState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : PongState
            Current game state.
        action : jax.Array
            int32 — Action for this frame (0=NOOP, 1=FIRE/no-op, 2=RIGHT→UP,
            3=LEFT→DOWN, 4=RIGHTFIRE→UP, 5=LEFTFIRE→DOWN).

        Returns
        -------
        new_state : PongState
            State after one emulated frame. `episode_step` is NOT incremented
            here — it is incremented once per agent step in `_step`.
        """
        key, subkey = jax.random.split(state.key)

        # Auto-serve: random angle, alternate direction each point
        angle = jax.random.uniform(subkey, minval=-jnp.pi / 6, maxval=jnp.pi / 6)
        serve_dir = jnp.where(
            (state.score + state.opp_score) % 2 == 0,
            jnp.float32(1.0),
            jnp.float32(-1.0),
        )
        serve_dx = jnp.float32(_BALL_SPEED) * jnp.cos(angle) * serve_dir
        serve_dy = jnp.float32(_BALL_SPEED) * jnp.sin(angle)

        bx = jnp.where(state.ball_active, state.ball_x, jnp.float32(_CENTRE_X))
        by = jnp.where(state.ball_active, state.ball_y, jnp.float32(_CENTRE_Y))
        bdx = jnp.where(state.ball_active, state.ball_dx, serve_dx)
        bdy = jnp.where(state.ball_active, state.ball_dy, serve_dy)

        # Player paddle movement
        # ALE action mapping: RIGHT/RIGHTFIRE → up; LEFT/LEFTFIRE → down; FIRE/NOOP → stay
        move_up = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_down = (action == jnp.int32(3)) | (action == jnp.int32(5))
        p_move = jnp.where(
            move_up,
            jnp.float32(-_PADDLE_SPEED),
            jnp.where(move_down, jnp.float32(_PADDLE_SPEED), jnp.float32(0.0)),
        )
        player_y = jnp.clip(
            state.player_y + p_move,
            jnp.float32(_COURT_TOP),
            jnp.float32(_COURT_BOTTOM - _PADDLE_H),
        )

        # CPU paddle: track ball centre at limited speed
        ball_cy = by + jnp.float32(_BALL_H / 2)
        cpu_cy = state.cpu_y + jnp.float32(_PADDLE_H / 2)
        cpu_delta = jnp.clip(
            ball_cy - cpu_cy,
            jnp.float32(-_CPU_SPEED),
            jnp.float32(_CPU_SPEED),
        )
        cpu_y = jnp.clip(
            state.cpu_y + cpu_delta,
            jnp.float32(_COURT_TOP),
            jnp.float32(_COURT_BOTTOM - _PADDLE_H),
        )

        # Move ball
        nx = bx + bdx
        ny = by + bdy

        # Top wall bounce
        top_hit = ny < jnp.float32(_COURT_TOP)
        ny = jnp.where(top_hit, 2.0 * _COURT_TOP - ny, ny)
        bdy = jnp.where(top_hit, -bdy, bdy)

        # Bottom wall bounce
        bot_hit = ny + _BALL_H > jnp.float32(_COURT_BOTTOM)
        ny = jnp.where(bot_hit, 2.0 * (_COURT_BOTTOM - _BALL_H) - ny, ny)
        bdy = jnp.where(bot_hit, -bdy, bdy)

        # Player paddle collision (right side)
        p_overlap = (
            (nx + _BALL_W > jnp.float32(_PLAYER_X))
            & (nx < jnp.float32(_PLAYER_X + _PADDLE_W))
            & (ny + _BALL_H > player_y)
            & (ny < player_y + _PADDLE_H)
        )
        p_hit = (bdx > 0) & p_overlap
        hit_rel_p = jnp.clip(
            (ny + jnp.float32(_BALL_H / 2.0) - player_y) / jnp.float32(_PADDLE_H),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        deflect_dy_p = jnp.float32(_BALL_SPEED) * (hit_rel_p * 2.0 - 1.0)
        bdx = jnp.where(p_hit, -jnp.abs(bdx), bdx)
        bdy = jnp.where(p_hit, deflect_dy_p, bdy)
        nx = jnp.where(p_hit, jnp.float32(_PLAYER_X - _BALL_W - 0.5), nx)

        # CPU paddle collision (left side)
        c_overlap = (
            (nx + _BALL_W > jnp.float32(_CPU_X))
            & (nx < jnp.float32(_CPU_X + _PADDLE_W))
            & (ny + _BALL_H > cpu_y)
            & (ny < cpu_y + _PADDLE_H)
        )
        c_hit = (bdx < 0) & c_overlap
        hit_rel_c = jnp.clip(
            (ny + jnp.float32(_BALL_H / 2.0) - cpu_y) / jnp.float32(_PADDLE_H),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        deflect_dy_c = jnp.float32(_BALL_SPEED) * (hit_rel_c * 2.0 - 1.0)
        bdx = jnp.where(c_hit, jnp.abs(bdx), bdx)
        bdy = jnp.where(c_hit, deflect_dy_c, bdy)
        nx = jnp.where(c_hit, jnp.float32(_CPU_X + _PADDLE_W + 0.5), nx)

        # Scoring: ball leaves the court horizontally
        player_scored = nx < jnp.float32(_COURT_LEFT)
        cpu_scored = nx + _BALL_W > jnp.float32(_COURT_RIGHT)

        new_score = state.score + jnp.where(player_scored, jnp.int32(1), jnp.int32(0))
        new_opp = state.opp_score + jnp.where(cpu_scored, jnp.int32(1), jnp.int32(0))
        step_reward = (new_score - state.score).astype(jnp.float32) - (
            new_opp - state.opp_score
        ).astype(jnp.float32)

        any_scored = player_scored | cpu_scored
        nx = jnp.where(any_scored, jnp.float32(_CENTRE_X), nx)
        ny = jnp.where(any_scored, jnp.float32(_CENTRE_Y), ny)
        ball_active = ~any_scored

        done = (new_score >= jnp.int32(_WIN_SCORE)) | (new_opp >= jnp.int32(_WIN_SCORE))

        return PongState(
            ball_x=nx,
            ball_y=ny,
            ball_dx=bdx,
            ball_dy=bdy,
            player_y=player_y,
            cpu_y=cpu_y,
            opp_score=new_opp,
            ball_active=ball_active,
            lives=jnp.int32(0),
            score=new_score,
            level=state.level,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step,
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: PongState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> PongState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for in-step randomness.
        state : PongState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5; see module docstring).
        params : AtaraxParams
            Static environment parameters (unused in physics, consumed by
            `AtaraxGame.step` for truncation).

        Returns
        -------
        new_state : PongState
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

    def render(self, state: PongState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : PongState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Court border (dim grey horizontal lines)
        in_court_x = (_COL_IDX >= _COURT_LEFT) & (_COL_IDX < _COURT_RIGHT)
        top_border = (_ROW_IDX == _COURT_TOP) & in_court_x
        bot_border = (_ROW_IDX == _COURT_BOTTOM - 1) & in_court_x
        frame = jnp.where((top_border | bot_border)[:, :, None], jnp.uint8(80), frame)

        # Net (dashed white line at x=_NET_X)
        net_col = _COL_IDX == _NET_X
        in_court_y = (_ROW_IDX >= _COURT_TOP) & (_ROW_IDX < _COURT_BOTTOM)
        net_dashed = (_ROW_IDX // 4) % 2 == 0
        net_mask = net_col & in_court_y & net_dashed
        frame = jnp.where(net_mask[:, :, None], jnp.uint8(180), frame)

        # CPU paddle (left, red)
        cpu_y_int = jnp.int32(state.cpu_y)
        cpu_mask = (
            (_ROW_IDX >= cpu_y_int)
            & (_ROW_IDX < cpu_y_int + _PADDLE_H)
            & (_COL_IDX >= _CPU_X)
            & (_COL_IDX < _CPU_X + _PADDLE_W)
        )
        frame = jnp.where(
            cpu_mask[:, :, None], jnp.array([200, 72, 72], jnp.uint8), frame
        )

        # Player paddle (right, green)
        py_int = jnp.int32(state.player_y)
        player_mask = (
            (_ROW_IDX >= py_int)
            & (_ROW_IDX < py_int + _PADDLE_H)
            & (_COL_IDX >= _PLAYER_X)
            & (_COL_IDX < _PLAYER_X + _PADDLE_W)
        )
        frame = jnp.where(
            player_mask[:, :, None], jnp.array([72, 160, 72], jnp.uint8), frame
        )

        # Ball (white)
        bx = jnp.int32(state.ball_x)
        by = jnp.int32(state.ball_y)
        ball_mask = (
            (_ROW_IDX >= by)
            & (_ROW_IDX < by + _BALL_H)
            & (_COL_IDX >= bx)
            & (_COL_IDX < bx + _BALL_W)
        )
        frame = jnp.where(ball_mask[:, :, None], jnp.uint8(255), frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Pong action indices.
            UP/W → action 2 (RIGHT→up); DOWN/S → action 3 (LEFT→down).
        """
        import pygame

        return {
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 3,
            pygame.K_s: 3,
        }
