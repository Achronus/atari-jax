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

"""Video Pinball — JAX-native game implementation.

Classic pinball game.  Use left/right flippers to keep the ball in play
and score points from bumpers, targets, and ramps.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (plunger — launch ball)
    2 — LEFT FLIPPER
    3 — RIGHT FLIPPER
    4 — LEFT FLIPPER + NUDGE
    5 — RIGHT FLIPPER + NUDGE

Scoring:
    Bumper hit       — +100
    Target hit       — +500
    Drain (ball lost) → lose 1 ball
    Episode ends when all balls are lost; lives = number of balls (3).
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_TABLE_LEFT: int = 15
_TABLE_RIGHT: int = 145
_TABLE_TOP: int = 20
_TABLE_BOTTOM: int = 195
_FLIPPER_Y: int = 185
_LEFT_FLIPPER_X: int = 40
_RIGHT_FLIPPER_X: int = 120
_FLIPPER_W: int = 25

_N_BUMPERS: int = 4
_N_TARGETS: int = 3

_GRAVITY: float = 0.15
_BALL_RADIUS: float = 4.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 20, 60], dtype=jnp.uint8)
_COLOR_WALLS = jnp.array([100, 80, 200], dtype=jnp.uint8)
_COLOR_BALL = jnp.array([220, 220, 220], dtype=jnp.uint8)
_COLOR_FLIPPER = jnp.array([100, 200, 255], dtype=jnp.uint8)
_COLOR_BUMPER = jnp.array([255, 100, 100], dtype=jnp.uint8)
_COLOR_TARGET = jnp.array([255, 200, 50], dtype=jnp.uint8)

_BUMPER_XS = jnp.array([50.0, 80.0, 110.0, 80.0], dtype=jnp.float32)
_BUMPER_YS = jnp.array([80.0, 60.0, 80.0, 110.0], dtype=jnp.float32)
_TARGET_XS = jnp.array([30.0, 80.0, 130.0], dtype=jnp.float32)
_TARGET_YS = jnp.array([40.0, 40.0, 40.0], dtype=jnp.float32)


@chex.dataclass
class VideoPinballState(AtariState):
    """
    Complete Video Pinball game state — a JAX pytree.

    Parameters
    ----------
    ball_x : jax.Array
        float32 — Ball x.
    ball_y : jax.Array
        float32 — Ball y.
    ball_vx : jax.Array
        float32 — Ball x velocity.
    ball_vy : jax.Array
        float32 — Ball y velocity.
    ball_in_play : jax.Array
        bool — Ball is launched.
    left_flipper_up : jax.Array
        bool — Left flipper raised.
    right_flipper_up : jax.Array
        bool — Right flipper raised.
    bumper_active : jax.Array
        bool[4] — Bumpers active (reset periodically).
    target_active : jax.Array
        bool[3] — Targets active (reset when all hit).
    balls_remaining : jax.Array
        int32 — Balls left.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    ball_x: jax.Array
    ball_y: jax.Array
    ball_vx: jax.Array
    ball_vy: jax.Array
    ball_in_play: jax.Array
    left_flipper_up: jax.Array
    right_flipper_up: jax.Array
    bumper_active: jax.Array
    target_active: jax.Array
    balls_remaining: jax.Array
    key: jax.Array


class VideoPinball(AtariEnv):
    """
    Video Pinball implemented as a pure JAX function suite.

    Keep the ball in play and score.  Lives (balls): 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> VideoPinballState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : VideoPinballState
            Ball ready to launch from plunger position.
        """
        return VideoPinballState(
            ball_x=jnp.float32(float(_RIGHT_FLIPPER_X)),
            ball_y=jnp.float32(float(_FLIPPER_Y - 10)),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(0.0),
            ball_in_play=jnp.bool_(False),
            left_flipper_up=jnp.bool_(False),
            right_flipper_up=jnp.bool_(False),
            bumper_active=jnp.ones(_N_BUMPERS, dtype=jnp.bool_),
            target_active=jnp.ones(_N_TARGETS, dtype=jnp.bool_),
            balls_remaining=jnp.int32(3),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: VideoPinballState, action: jax.Array
    ) -> VideoPinballState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : VideoPinballState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : VideoPinballState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Flipper state
        new_lf = (action == jnp.int32(2)) | (action == jnp.int32(4))
        new_rf = (action == jnp.int32(3)) | (action == jnp.int32(5))

        # Launch ball
        launch = (action == jnp.int32(1)) & ~state.ball_in_play
        new_in_play = state.ball_in_play | launch
        new_bvx = jnp.where(launch, jnp.float32(-1.0), state.ball_vx)
        new_bvy = jnp.where(launch, jnp.float32(-5.0), state.ball_vy)

        # Ball physics
        new_bvy2 = jnp.where(new_in_play, new_bvy + _GRAVITY, jnp.float32(0.0))
        new_bx = jnp.where(new_in_play, state.ball_x + new_bvx, state.ball_x)
        new_by = jnp.where(new_in_play, state.ball_y + new_bvy2, state.ball_y)

        # Wall bounces
        at_left = new_bx < float(_TABLE_LEFT + _BALL_RADIUS)
        at_right = new_bx > float(_TABLE_RIGHT - _BALL_RADIUS)
        at_top = new_by < float(_TABLE_TOP + _BALL_RADIUS)
        new_bvx2 = jnp.where(at_left | at_right, -new_bvx, new_bvx)
        new_bvy3 = jnp.where(at_top, -new_bvy2, new_bvy2)
        new_bx2 = jnp.clip(
            new_bx,
            float(_TABLE_LEFT + _BALL_RADIUS),
            float(_TABLE_RIGHT - _BALL_RADIUS),
        )
        new_by2 = jnp.where(at_top, float(_TABLE_TOP + _BALL_RADIUS), new_by)

        # Flipper deflection
        near_lflipper = (
            new_in_play
            & new_lf
            & (new_by2 >= float(_FLIPPER_Y - 8))
            & (new_bx2 >= float(_LEFT_FLIPPER_X))
            & (new_bx2 < float(_LEFT_FLIPPER_X + _FLIPPER_W))
        )
        near_rflipper = (
            new_in_play
            & new_rf
            & (new_by2 >= float(_FLIPPER_Y - 8))
            & (new_bx2 >= float(_RIGHT_FLIPPER_X - _FLIPPER_W))
            & (new_bx2 < float(_RIGHT_FLIPPER_X))
        )
        flipper_hit = near_lflipper | near_rflipper
        flip_vx = jnp.where(
            near_lflipper,
            jnp.float32(2.0),
            jnp.where(near_rflipper, jnp.float32(-2.0), jnp.float32(0.0)),
        )
        new_bvx3 = jnp.where(flipper_hit, flip_vx, new_bvx2)
        new_bvy4 = jnp.where(flipper_hit, jnp.float32(-5.0), new_bvy3)
        new_by3 = jnp.where(flipper_hit, jnp.float32(float(_FLIPPER_Y - 8)), new_by2)

        # Bumper collisions
        bumper_hit = (
            new_in_play
            & state.bumper_active
            & (jnp.abs(new_bx2 - _BUMPER_XS) < jnp.float32(8.0))
            & (jnp.abs(new_by3 - _BUMPER_YS) < jnp.float32(8.0))
        )
        step_reward = step_reward + jnp.sum(bumper_hit).astype(
            jnp.float32
        ) * jnp.float32(100.0)
        new_bumper_active = state.bumper_active & ~bumper_hit
        # Bounce ball away from bumper
        bumper_any = jnp.any(bumper_hit)
        new_bvy5 = jnp.where(
            bumper_any, -jnp.abs(new_bvy4) - jnp.float32(2.0), new_bvy4
        )

        # Target hits
        target_hit = (
            new_in_play
            & state.target_active
            & (jnp.abs(new_bx2 - _TARGET_XS) < jnp.float32(8.0))
            & (jnp.abs(new_by3 - _TARGET_YS) < jnp.float32(8.0))
        )
        step_reward = step_reward + jnp.sum(target_hit).astype(
            jnp.float32
        ) * jnp.float32(500.0)
        new_target_active = state.target_active & ~target_hit
        # Reset targets when all hit
        all_targets_hit = ~jnp.any(new_target_active)
        new_target_active2 = jnp.where(
            all_targets_hit, jnp.ones(_N_TARGETS, dtype=jnp.bool_), new_target_active
        )
        # Reset bumpers on target reset
        new_bumper_active2 = jnp.where(
            all_targets_hit, jnp.ones(_N_BUMPERS, dtype=jnp.bool_), new_bumper_active
        )

        # Ball drained
        drained = new_in_play & (new_by3 > float(_TABLE_BOTTOM))
        new_balls = state.balls_remaining - jnp.where(
            drained, jnp.int32(1), jnp.int32(0)
        )
        new_in_play2 = new_in_play & ~drained
        new_bx3 = jnp.where(drained, jnp.float32(float(_RIGHT_FLIPPER_X)), new_bx2)
        new_by4 = jnp.where(drained, jnp.float32(float(_FLIPPER_Y - 10)), new_by3)
        new_bvx4 = jnp.where(drained, jnp.float32(0.0), new_bvx3)
        new_bvy6 = jnp.where(drained, jnp.float32(0.0), new_bvy5)

        done = new_balls <= jnp.int32(0)

        return VideoPinballState(
            ball_x=new_bx3,
            ball_y=new_by4,
            ball_vx=new_bvx4,
            ball_vy=new_bvy6,
            ball_in_play=new_in_play2,
            left_flipper_up=new_lf,
            right_flipper_up=new_rf,
            bumper_active=new_bumper_active2,
            target_active=new_target_active2,
            balls_remaining=new_balls,
            key=key,
            lives=new_balls,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: VideoPinballState, action: jax.Array) -> VideoPinballState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : VideoPinballState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : VideoPinballState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: VideoPinballState) -> VideoPinballState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: VideoPinballState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : VideoPinballState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Table outline
        wall_mask = (
            (_ROW_IDX == _TABLE_TOP)
            | (_ROW_IDX == _TABLE_BOTTOM)
            | (_COL_IDX == _TABLE_LEFT)
            | (_COL_IDX == _TABLE_RIGHT)
        )
        frame = jnp.where(wall_mask[:, :, None], _COLOR_WALLS, frame)

        # Bumpers
        def draw_bumper(frm, i):
            bx = _BUMPER_XS[i].astype(jnp.int32)
            by = _BUMPER_YS[i].astype(jnp.int32)
            mask = (
                state.bumper_active[i]
                & (_ROW_IDX >= by - 5)
                & (_ROW_IDX < by + 5)
                & (_COL_IDX >= bx - 5)
                & (_COL_IDX < bx + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_BUMPER, frm), None

        frame, _ = jax.lax.scan(draw_bumper, frame, jnp.arange(_N_BUMPERS))

        # Targets
        def draw_target(frm, i):
            tx = _TARGET_XS[i].astype(jnp.int32)
            ty = _TARGET_YS[i].astype(jnp.int32)
            mask = (
                state.target_active[i]
                & (_ROW_IDX >= ty - 3)
                & (_ROW_IDX < ty + 3)
                & (_COL_IDX >= tx - 4)
                & (_COL_IDX < tx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_TARGET, frm), None

        frame, _ = jax.lax.scan(draw_target, frame, jnp.arange(_N_TARGETS))

        # Flippers
        lf_mask = (
            (_ROW_IDX >= _FLIPPER_Y - 3)
            & (_ROW_IDX < _FLIPPER_Y + 3)
            & (_COL_IDX >= _LEFT_FLIPPER_X)
            & (_COL_IDX < _LEFT_FLIPPER_X + _FLIPPER_W)
        )
        rf_mask = (
            (_ROW_IDX >= _FLIPPER_Y - 3)
            & (_ROW_IDX < _FLIPPER_Y + 3)
            & (_COL_IDX >= _RIGHT_FLIPPER_X - _FLIPPER_W)
            & (_COL_IDX < _RIGHT_FLIPPER_X)
        )
        frame = jnp.where(lf_mask[:, :, None], _COLOR_FLIPPER, frame)
        frame = jnp.where(rf_mask[:, :, None], _COLOR_FLIPPER, frame)

        # Ball
        bx = state.ball_x.astype(jnp.int32)
        by = state.ball_y.astype(jnp.int32)
        bm = (
            state.ball_in_play
            & (_ROW_IDX >= by - 3)
            & (_ROW_IDX < by + 3)
            & (_COL_IDX >= bx - 3)
            & (_COL_IDX < bx + 3)
        )
        # Ball always visible when not in play (at plunger)
        bm2 = (
            ~state.ball_in_play
            & (_ROW_IDX >= by - 3)
            & (_ROW_IDX < by + 3)
            & (_COL_IDX >= bx - 3)
            & (_COL_IDX < bx + 3)
        )
        frame = jnp.where((bm | bm2)[:, :, None], _COLOR_BALL, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Video Pinball action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_LEFT: 2,
            pygame.K_a: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
        }
