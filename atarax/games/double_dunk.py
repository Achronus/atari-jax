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

"""Double Dunk — JAX-native game implementation.

2-on-2 half-court basketball.  Score baskets for points; the CPU team
defends and attacks.  First to 24 points wins.

Action space (18 actions — simplified to 6):
    0 — NOOP
    1 — FIRE (shoot / steal)
    2 — UP   (jump)
    3 — RIGHT
    4 — DOWN (move to position)
    5 — LEFT

Scoring:
    2-point basket — +2
    3-point basket — +3
    Episode ends when a team reaches 24 points; lives: 0.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_COURT_LEFT: int = 10
_COURT_RIGHT: int = 150
_COURT_TOP: int = 30
_COURT_BOTTOM: int = 180
_BASKET_X: int = 130
_BASKET_Y: int = 70
_THREE_POINT_X: int = 50

_PLAYER_SPEED: float = 2.0
_BALL_SPEED: float = 3.0
_GRAVITY: float = 0.3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([200, 180, 120], dtype=jnp.uint8)  # court
_COLOR_LINES = jnp.array([160, 140, 90], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 50, 50], dtype=jnp.uint8)
_COLOR_CPU = jnp.array([50, 50, 255], dtype=jnp.uint8)
_COLOR_BALL = jnp.array([255, 140, 0], dtype=jnp.uint8)
_COLOR_BASKET = jnp.array([80, 40, 0], dtype=jnp.uint8)


@chex.dataclass
class DoubleDunkState(AtariState):
    """
    Complete Double Dunk game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Controlled player x.
    player_y : jax.Array
        float32 — Controlled player y.
    player_vy : jax.Array
        float32 — Vertical velocity.
    ball_x : jax.Array
        float32 — Ball x.
    ball_y : jax.Array
        float32 — Ball y.
    ball_vx : jax.Array
        float32 — Ball x velocity.
    ball_vy : jax.Array
        float32 — Ball y velocity.
    ball_held : jax.Array
        bool — Player holds the ball.
    cpu_x : jax.Array
        float32 — CPU defender x.
    cpu_y : jax.Array
        float32 — CPU defender y.
    cpu_score : jax.Array
        int32 — CPU team score.
    possession_timer : jax.Array
        int32 — Frames until shot clock expires.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_vx: jax.Array
    ball_vy: jax.Array
    ball_held: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    cpu_score: jax.Array
    possession_timer: jax.Array
    key: jax.Array


_GROUND_Y: float = float(_COURT_BOTTOM - 10)


class DoubleDunk(AtariEnv):
    """
    Double Dunk implemented as a pure JAX function suite.

    Score 24 points before the CPU.  Episode ends at 24 points.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=50_000)

    def _reset(self, key: jax.Array) -> DoubleDunkState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : DoubleDunkState
            Player with ball at centre, CPU defender near basket.
        """
        return DoubleDunkState(
            player_x=jnp.float32(60.0),
            player_y=jnp.float32(_GROUND_Y),
            player_vy=jnp.float32(0.0),
            ball_x=jnp.float32(60.0),
            ball_y=jnp.float32(_GROUND_Y - 10.0),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(0.0),
            ball_held=jnp.bool_(True),
            cpu_x=jnp.float32(110.0),
            cpu_y=jnp.float32(_GROUND_Y),
            cpu_score=jnp.int32(0),
            possession_timer=jnp.int32(240),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: DoubleDunkState, action: jax.Array
    ) -> DoubleDunkState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : DoubleDunkState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : DoubleDunkState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement
        on_ground = state.player_y >= jnp.float32(_GROUND_Y - 1.0)
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(
            state.player_x + dx, float(_COURT_LEFT), float(_COURT_RIGHT - 8)
        )
        jump = (action == jnp.int32(2)) & on_ground
        new_vy = jnp.where(jump, jnp.float32(-5.0), state.player_vy + _GRAVITY)
        new_py = jnp.minimum(state.player_y + new_vy, jnp.float32(_GROUND_Y))
        new_vy = jnp.where(new_py >= jnp.float32(_GROUND_Y), jnp.float32(0.0), new_vy)

        # Ball follows player when held
        new_bx = jnp.where(
            state.ball_held, new_px + jnp.float32(4.0), state.ball_x + state.ball_vx
        )
        new_by = jnp.where(
            state.ball_held, new_py - jnp.float32(10.0), state.ball_y + state.ball_vy
        )
        new_bvy = jnp.where(state.ball_held, jnp.float32(0.0), state.ball_vy + _GRAVITY)
        new_bvx = state.ball_vx

        # Bounce off floor
        at_floor = new_by >= jnp.float32(_GROUND_Y)
        new_by = jnp.where(at_floor, jnp.float32(_GROUND_Y), new_by)
        new_bvy = jnp.where(at_floor, -new_bvy * jnp.float32(0.6), new_bvy)

        # Shoot toward basket
        shoot = (action == jnp.int32(1)) & state.ball_held
        basket_dx = jnp.float32(_BASKET_X) - new_px
        basket_dy = jnp.float32(_BASKET_Y) - new_py
        shoot_dist = jnp.sqrt(basket_dx**2 + basket_dy**2 + jnp.float32(1.0))
        shoot_vx = jnp.where(shoot, basket_dx / shoot_dist * _BALL_SPEED, new_bvx)
        shoot_vy = jnp.where(shoot, jnp.float32(-4.0), new_bvy)  # arc upward
        new_ball_held = state.ball_held & ~shoot
        new_bvx2 = jnp.where(shoot, shoot_vx, new_bvx)
        new_bvy2 = jnp.where(shoot, shoot_vy, new_bvy)

        # Ball scores
        near_basket = (jnp.abs(new_bx - jnp.float32(_BASKET_X)) < jnp.float32(8.0)) & (
            jnp.abs(new_by - jnp.float32(_BASKET_Y)) < jnp.float32(8.0)
        )
        three_pointer = new_px < jnp.float32(_THREE_POINT_X)
        basket_scored = near_basket & ~state.ball_held
        step_reward = step_reward + jnp.where(
            basket_scored & three_pointer,
            jnp.float32(3.0),
            jnp.where(basket_scored, jnp.float32(2.0), jnp.float32(0.0)),
        )

        # After score, reset ball to player
        new_ball_held2 = jnp.where(basket_scored, jnp.bool_(True), new_ball_held)
        new_bx2 = jnp.where(basket_scored, new_px, new_bx)
        new_by2 = jnp.where(basket_scored, new_py - jnp.float32(10.0), new_by)
        new_bvx3 = jnp.where(basket_scored, jnp.float32(0.0), new_bvx2)
        new_bvy3 = jnp.where(basket_scored, jnp.float32(0.0), new_bvy2)

        # CPU simple AI: moves toward player if no ball, toward basket with ball
        cpu_dx = jnp.clip((new_px - state.cpu_x) * 0.04, -1.0, 1.0)
        new_cpu_x = state.cpu_x + cpu_dx
        new_cpu_x = jnp.clip(new_cpu_x, float(_COURT_LEFT), float(_COURT_RIGHT - 8))

        # CPU occasionally scores (simplified: shot clock)
        new_possession_timer = state.possession_timer - jnp.int32(1)
        cpu_scores = new_possession_timer <= jnp.int32(0)
        new_cpu_score = state.cpu_score + jnp.where(
            cpu_scores, jnp.int32(2), jnp.int32(0)
        )
        new_possession_timer = jnp.where(
            cpu_scores, jnp.int32(240), new_possession_timer
        )

        player_total = state.score + jnp.int32(step_reward)
        done = (player_total >= jnp.int32(24)) | (new_cpu_score >= jnp.int32(24))

        return DoubleDunkState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            ball_x=new_bx2,
            ball_y=new_by2,
            ball_vx=new_bvx3,
            ball_vy=new_bvy3,
            ball_held=new_ball_held2,
            cpu_x=new_cpu_x,
            cpu_y=state.cpu_y,
            cpu_score=new_cpu_score,
            possession_timer=new_possession_timer,
            key=key,
            lives=jnp.int32(0),
            score=player_total,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: DoubleDunkState, action: jax.Array) -> DoubleDunkState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : DoubleDunkState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : DoubleDunkState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: DoubleDunkState) -> DoubleDunkState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: DoubleDunkState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : DoubleDunkState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Basket
        bk = (
            (_ROW_IDX >= _BASKET_Y - 2)
            & (_ROW_IDX <= _BASKET_Y + 2)
            & (_COL_IDX >= _BASKET_X - 6)
            & (_COL_IDX <= _BASKET_X + 6)
        )
        frame = jnp.where(bk[:, :, None], _COLOR_BASKET, frame)

        # CPU player
        cx = state.cpu_x.astype(jnp.int32)
        cy = state.cpu_y.astype(jnp.int32)
        cm = (
            (_ROW_IDX >= cy - 10)
            & (_ROW_IDX < cy)
            & (_COL_IDX >= cx)
            & (_COL_IDX < cx + 8)
        )
        frame = jnp.where(cm[:, :, None], _COLOR_CPU, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py - 10)
            & (_ROW_IDX < py)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Ball
        bx = state.ball_x.astype(jnp.int32)
        by = state.ball_y.astype(jnp.int32)
        ball_mask = (
            (_ROW_IDX >= by - 4)
            & (_ROW_IDX < by + 4)
            & (_COL_IDX >= bx - 4)
            & (_COL_IDX < bx + 4)
        )
        frame = jnp.where(ball_mask[:, :, None], _COLOR_BALL, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Double Dunk action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
