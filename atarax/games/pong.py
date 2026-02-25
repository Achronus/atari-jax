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

Mechanics implemented directly in JAX with no hardware emulation.
All conditionals use `jnp.where`; the step loop uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Court        : x ∈ [8, 152),  y ∈ [34, 194)
    CPU paddle   : x ∈ [8, 12),   h=16 px,  y ∈ [34, 178)
    Player paddle: x ∈ [148, 152), h=16 px,  y ∈ [34, 178)
    Ball         : 2 × 4 px
    Net          : x = 80,  dashed every 4 px

Action space (minimal, 3 actions):
    0 — NOOP
    1 — UP   (move player paddle up)
    2 — DOWN (move player paddle down)
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
_COURT_TOP: int = 34
_COURT_BOTTOM: int = 194
_COURT_LEFT: int = 8
_COURT_RIGHT: int = 152

_CPU_X: int = 8  # CPU paddle left edge
_PLAYER_X: int = 148  # Player paddle left edge
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

# Initial positions
_CENTRE_X: float = (_COURT_LEFT + _COURT_RIGHT - _BALL_W) / 2.0
_CENTRE_Y: float = (_COURT_TOP + _COURT_BOTTOM - _BALL_H) / 2.0
_PADDLE_INIT_Y: float = (_COURT_TOP + _COURT_BOTTOM - _PADDLE_H) / 2.0

# Precomputed scanline / column index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]


@chex.dataclass
class PongState(AtariState):
    """
    Complete Pong game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score` from `AtariState`.  `score` holds the player's point
    total; `opp_score` holds the CPU's point total.

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
        int32 — CPU score.
    ball_active : jax.Array
        bool — `False` during the one-frame pause after a point is scored;
        auto-serve fires on the next sub-step.
    key : jax.Array
        uint32[2] — PRNG key evolved each frame for stochastic serve angles.
    """

    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_dy: jax.Array
    player_y: jax.Array
    cpu_y: jax.Array
    opp_score: jax.Array
    ball_active: jax.Array
    key: jax.Array


class Pong(AtariEnv):
    """
    Pong implemented as a pure JAX function suite.

    No hardware emulation — game physics are computed directly using
    `jnp.where` for all conditionals and `jax.lax.fori_loop` for the
    4-frame skip inside `_step`.

    The CPU opponent tracks the ball centre at a limited speed (`_CPU_SPEED`
    pixels per sub-step), making it beatable but non-trivial.
    """

    num_actions: int = 3

    def _reset(self, key: jax.Array) -> PongState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

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
            int32 — Action for this frame (0=NOOP, 1=UP, 2=DOWN).

        Returns
        -------
        new_state : PongState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # Auto-serve when ball is inactive: pick a random angle and alternate direction
        angle = jax.random.uniform(subkey, minval=-jnp.pi / 6, maxval=jnp.pi / 6)
        serve_dir = jnp.where(
            (state.score + state.opp_score) % 2 == 0,
            jnp.float32(1.0),
            jnp.float32(-1.0),
        )
        serve_dx = jnp.float32(_BALL_SPEED) * jnp.cos(angle) * serve_dir
        serve_dy = jnp.float32(_BALL_SPEED) * jnp.sin(angle)

        ball_x = jnp.where(state.ball_active, state.ball_x, jnp.float32(_CENTRE_X))
        ball_y = jnp.where(state.ball_active, state.ball_y, jnp.float32(_CENTRE_Y))
        ball_dx = jnp.where(state.ball_active, state.ball_dx, serve_dx)
        ball_dy = jnp.where(state.ball_active, state.ball_dy, serve_dy)

        # --- Player paddle movement (UP = decrease y, DOWN = increase y) ---
        p_move = jnp.where(
            action == 1,
            jnp.float32(-_PADDLE_SPEED),
            jnp.where(action == 2, jnp.float32(_PADDLE_SPEED), jnp.float32(0.0)),
        )
        player_y = jnp.clip(
            state.player_y + p_move,
            jnp.float32(_COURT_TOP),
            jnp.float32(_COURT_BOTTOM - _PADDLE_H),
        )

        # --- CPU paddle: track ball centre (limited speed) ---
        ball_centre_y = ball_y + jnp.float32(_BALL_H / 2)
        cpu_centre_y = state.cpu_y + jnp.float32(_PADDLE_H / 2)
        cpu_delta = jnp.clip(
            ball_centre_y - cpu_centre_y,
            jnp.float32(-_CPU_SPEED),
            jnp.float32(_CPU_SPEED),
        )
        cpu_y = jnp.clip(
            state.cpu_y + cpu_delta,
            jnp.float32(_COURT_TOP),
            jnp.float32(_COURT_BOTTOM - _PADDLE_H),
        )

        # --- Move ball ---
        new_x = ball_x + ball_dx
        new_y = ball_y + ball_dy

        # --- Top wall bounce ---
        top_hit = new_y < jnp.float32(_COURT_TOP)
        new_y = jnp.where(top_hit, 2.0 * _COURT_TOP - new_y, new_y)
        ball_dy = jnp.where(top_hit, -ball_dy, ball_dy)

        # --- Bottom wall bounce ---
        bottom_hit = new_y + _BALL_H > jnp.float32(_COURT_BOTTOM)
        new_y = jnp.where(bottom_hit, 2.0 * (_COURT_BOTTOM - _BALL_H) - new_y, new_y)
        ball_dy = jnp.where(bottom_hit, -ball_dy, ball_dy)

        # --- Player paddle collision (right side, ball moving rightward) ---
        player_overlap = (
            (new_x + _BALL_W > jnp.float32(_PLAYER_X))
            & (new_x < jnp.float32(_PLAYER_X + _PADDLE_W))
            & (new_y + _BALL_H > player_y)
            & (new_y < player_y + _PADDLE_H)
        )
        player_hit = (ball_dx > 0) & player_overlap
        hit_rel_p = jnp.clip(
            (new_y + jnp.float32(_BALL_H / 2.0) - player_y) / jnp.float32(_PADDLE_H),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        deflect_dy_p = jnp.float32(_BALL_SPEED) * (hit_rel_p * 2.0 - 1.0)
        ball_dx = jnp.where(player_hit, -jnp.abs(ball_dx), ball_dx)
        ball_dy = jnp.where(player_hit, deflect_dy_p, ball_dy)
        new_x = jnp.where(player_hit, jnp.float32(_PLAYER_X - _BALL_W - 0.5), new_x)

        # --- CPU paddle collision (left side, ball moving leftward) ---
        cpu_overlap = (
            (new_x + _BALL_W > jnp.float32(_CPU_X))
            & (new_x < jnp.float32(_CPU_X + _PADDLE_W))
            & (new_y + _BALL_H > cpu_y)
            & (new_y < cpu_y + _PADDLE_H)
        )
        cpu_hit = (ball_dx < 0) & cpu_overlap
        hit_rel_c = jnp.clip(
            (new_y + jnp.float32(_BALL_H / 2.0) - cpu_y) / jnp.float32(_PADDLE_H),
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        deflect_dy_c = jnp.float32(_BALL_SPEED) * (hit_rel_c * 2.0 - 1.0)
        ball_dx = jnp.where(cpu_hit, jnp.abs(ball_dx), ball_dx)
        ball_dy = jnp.where(cpu_hit, deflect_dy_c, ball_dy)
        new_x = jnp.where(cpu_hit, jnp.float32(_CPU_X + _PADDLE_W + 0.5), new_x)

        # --- Scoring (checked after paddle updates so bounced balls don't score) ---
        player_scored = new_x < jnp.float32(_COURT_LEFT)
        cpu_scored = new_x + _BALL_W > jnp.float32(_COURT_RIGHT)

        new_score = state.score + jnp.where(player_scored, jnp.int32(1), jnp.int32(0))
        new_opp_score = state.opp_score + jnp.where(
            cpu_scored, jnp.int32(1), jnp.int32(0)
        )
        step_reward = (new_score - state.score).astype(jnp.float32) - (
            new_opp_score - state.opp_score
        ).astype(jnp.float32)

        # Reset ball to centre and deactivate on scoring
        any_scored = player_scored | cpu_scored
        new_x = jnp.where(any_scored, jnp.float32(_CENTRE_X), new_x)
        new_y = jnp.where(any_scored, jnp.float32(_CENTRE_Y), new_y)
        ball_active = ~any_scored

        done = (new_score >= jnp.int32(_WIN_SCORE)) | (
            new_opp_score >= jnp.int32(_WIN_SCORE)
        )

        return PongState(
            ball_x=new_x,
            ball_y=new_y,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            player_y=player_y,
            cpu_y=cpu_y,
            opp_score=new_opp_score,
            ball_active=ball_active,
            lives=jnp.int32(0),
            score=new_score,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: PongState, action: jax.Array) -> PongState:
        """
        Advance the game by one agent step (4 emulated frames).

        The reward is accumulated across all 4 frames, matching the ALE
        frame-skip convention.

        Parameters
        ----------
        state : PongState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=UP, 2=DOWN).

        Returns
        -------
        new_state : PongState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: PongState) -> PongState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, _FRAME_SKIP, body, state)

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

        # Net (dashed white line at x=_NET_X, inside court)
        net_col = _COL_IDX == _NET_X
        in_court_y = (_ROW_IDX >= _COURT_TOP) & (_ROW_IDX < _COURT_BOTTOM)
        net_dashed = (_ROW_IDX // 4) % 2 == 0
        net_mask = net_col & in_court_y & net_dashed
        frame = jnp.where(net_mask[:, :, None], jnp.uint8(180), frame)

        # CPU paddle (left, orange-tinted)
        cpu_y_int = jnp.int32(state.cpu_y)
        cpu_mask = (
            (_ROW_IDX >= cpu_y_int)
            & (_ROW_IDX < cpu_y_int + _PADDLE_H)
            & (_COL_IDX >= _CPU_X)
            & (_COL_IDX < _CPU_X + _PADDLE_W)
        )
        cpu_color = jnp.array([200, 72, 72], dtype=jnp.uint8)
        frame = jnp.where(cpu_mask[:, :, None], cpu_color, frame)

        # Player paddle (right, green-tinted)
        player_y_int = jnp.int32(state.player_y)
        player_mask = (
            (_ROW_IDX >= player_y_int)
            & (_ROW_IDX < player_y_int + _PADDLE_H)
            & (_COL_IDX >= _PLAYER_X)
            & (_COL_IDX < _PLAYER_X + _PADDLE_W)
        )
        player_color = jnp.array([72, 160, 72], dtype=jnp.uint8)
        frame = jnp.where(player_mask[:, :, None], player_color, frame)

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
            Actions: 0=NOOP, 1=UP, 2=DOWN.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_DOWN: 2,
            pygame.K_s: 2,
        }
