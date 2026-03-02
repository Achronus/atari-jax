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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Table left wall  : x = 15
    Table right wall : x = 145
    Table top        : y = 20
    Table bottom     : y = 195
    Left flipper     : x ∈ [40, 78], y = 185
    Right flipper    : x ∈ [82, 120], y = 185
    Bumpers (4)      : (50,80), (80,60), (110,80), (80,110)
    Targets (3)      : (30,40), (80,40), (130,40)

Action space (9 actions — ALE minimal set):
    0  NOOP
    1  FIRE (plunger)
    2  UP
    3  RIGHT (right flipper)
    4  LEFT (left flipper)
    5  DOWN
    6  UPFIRE
    7  RIGHTFIRE
    8  LEFTFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Table geometry
_TABLE_LEFT: float = 15.0
_TABLE_RIGHT: float = 145.0
_TABLE_TOP: float = 20.0
_TABLE_BOTTOM: float = 195.0

# Flippers
_FLIPPER_Y: float = 185.0
_FLIPPER_H: float = 6.0
_LEFT_FLIPPER_X0: float = 40.0
_LEFT_FLIPPER_X1: float = 78.0
_RIGHT_FLIPPER_X0: float = 82.0
_RIGHT_FLIPPER_X1: float = 120.0
_FLIPPER_KICK: float = 6.0

# Ball
_BALL_R: float = 3.0
_BALL_GRAVITY: float = 0.06
_PLUNGER_CHARGE_RATE: float = 0.5
_PLUNGER_MAX: float = 8.0
_BALL_LAUNCH_X: float = 130.0  # plunger lane x
_BALL_LAUNCH_Y: float = 180.0

# Bumpers (4)
_N_BUMPERS: int = 4
# Use Python tuples so loop bodies stay concrete (no traced JAX array indexing)
_BUMPER_CENTERS: tuple = ((50.0, 80.0), (80.0, 60.0), (110.0, 80.0), (80.0, 110.0))
_BUMPER_R: float = 8.0
_BUMPER_POINTS: int = 100
# Integer versions for render scanlines
_BUMPER_CENTERS_I: tuple = ((50, 80), (80, 60), (110, 80), (80, 110))
_BUMPER_R_I: int = int(_BUMPER_R)

# Targets (3)
_N_TARGETS: int = 3
# Use Python tuples so loop bodies stay concrete (no traced JAX array indexing)
_TARGET_ORIGINS: tuple = ((30.0, 40.0), (80.0, 40.0), (130.0, 40.0))
_TARGET_W: float = 10.0
_TARGET_H: float = 6.0
_TARGET_POINTS: int = 500

_BUMPER_CLUSTER_CX: float = 80.0    # cluster geometric centre x
_BUMPER_CLUSTER_CY: float = 82.5    # cluster geometric centre y
_BUMPER_CLUSTER_BIAS: float = 0.8   # fraction of kick direction blended toward cluster centre
_BUMPER_KICK_SPEED: float = 5.0     # fixed post-hit speed (models ROM spring-bumper impulse)

_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4

# Render
_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_TABLE_COLOR = jnp.array([20, 20, 50], dtype=jnp.uint8)
_WALL_COLOR = jnp.array([100, 100, 150], dtype=jnp.uint8)
_FLIPPER_COLOR = jnp.array([150, 200, 255], dtype=jnp.uint8)
_BALL_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
_BUMPER_COLOR = jnp.array([255, 200, 50], dtype=jnp.uint8)
_TARGET_COLOR = jnp.array([80, 220, 80], dtype=jnp.uint8)
_TARGET_HIT_COLOR = jnp.array([40, 80, 40], dtype=jnp.uint8)


@chex.dataclass
class VideoPinballState(AtariState):
    """
    Complete Video Pinball game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `lives` = balls remaining (starts at 3).

    Parameters
    ----------
    ball_x : jax.Array
        float32 — Ball centre x.
    ball_y : jax.Array
        float32 — Ball centre y.
    ball_dx : jax.Array
        float32 — Ball horizontal velocity.
    ball_dy : jax.Array
        float32 — Ball vertical velocity.
    ball_active : jax.Array
        bool — False when waiting for launch.
    left_flipper_up : jax.Array
        bool — True when left flipper is raised.
    right_flipper_up : jax.Array
        bool — True when right flipper is raised.
    bumper_active : jax.Array
        bool[4] — Active bumpers (reset when all targets cleared).
    target_hit : jax.Array
        bool[3] — Which targets have been hit.
    plunger_power : jax.Array
        float32 — Charges while FIRE is held.
    extra_ball_awarded : jax.Array
        bool — True once the one-time bonus ball has been granted
        (mirrors ROM RAM[0xA8] & 0x1 tracked in VideoPinball.cpp).
    """

    ball_x: chex.Array
    ball_y: chex.Array
    ball_dx: chex.Array
    ball_dy: chex.Array
    ball_active: chex.Array
    left_flipper_up: chex.Array
    right_flipper_up: chex.Array
    bumper_active: chex.Array
    target_hit: chex.Array
    plunger_power: chex.Array
    extra_ball_awarded: chex.Array


class VideoPinball(AtaraxGame):
    """
    Video Pinball implemented as a pure-JAX function suite.

    Use left and right flippers to keep the ball in play.  Score points
    from bumpers (+100) and targets (+500).  The ball drains past the
    flippers, losing a life.
    """

    num_actions: int = 9

    def _reset(self, key: chex.PRNGKey) -> VideoPinballState:
        """Return the canonical initial game state."""
        return VideoPinballState(
            ball_x=jnp.float32(_BALL_LAUNCH_X),
            ball_y=jnp.float32(_BALL_LAUNCH_Y),
            ball_dx=jnp.float32(0.0),
            ball_dy=jnp.float32(0.0),
            ball_active=jnp.bool_(False),
            left_flipper_up=jnp.bool_(False),
            right_flipper_up=jnp.bool_(False),
            bumper_active=jnp.ones(_N_BUMPERS, dtype=jnp.bool_),
            target_hit=jnp.zeros(_N_TARGETS, dtype=jnp.bool_),
            plunger_power=jnp.float32(0.0),
            extra_ball_awarded=jnp.bool_(False),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),
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
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : VideoPinballState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–8).

        Returns
        -------
        new_state : VideoPinballState
            State after one emulated frame.
        """
        key, _ = jax.random.split(state.key)

        # --- Action decode ---
        right_flipper = (action == 3) | (action == 7)
        left_flipper = (action == 4) | (action == 8)
        has_fire = (action == 1) | (action == 6) | (action == 7) | (action == 8)

        # --- Flipper state ---
        left_flipper_up = left_flipper
        right_flipper_up = right_flipper

        # --- Ball not active: charge plunger ---
        new_plunger = jnp.where(
            ~state.ball_active & has_fire,
            jnp.minimum(
                state.plunger_power + jnp.float32(_PLUNGER_CHARGE_RATE),
                jnp.float32(_PLUNGER_MAX),
            ),
            state.plunger_power,
        )
        # Launch ball when FIRE released (fire not held and plunger charged)
        launch = (
            ~state.ball_active & ~has_fire & (state.plunger_power > jnp.float32(0.0))
        )
        new_ball_dy = jnp.where(
            launch, -(state.plunger_power + jnp.float32(5.0)), state.ball_dy
        )
        new_ball_dx = jnp.where(launch, jnp.float32(-1.5), state.ball_dx)
        new_plunger = jnp.where(launch, jnp.float32(0.0), new_plunger)
        ball_active = state.ball_active | launch

        # --- Ball physics ---
        ball_dy = jnp.where(
            ball_active,
            new_ball_dy + jnp.float32(_BALL_GRAVITY),
            new_ball_dy,
        )
        ball_x = jnp.where(ball_active, state.ball_x + new_ball_dx, state.ball_x)
        ball_y = jnp.where(ball_active, state.ball_y + ball_dy, state.ball_y)
        ball_dx = new_ball_dx

        # --- Wall bounces ---
        hit_left_wall = ball_active & (ball_x - _BALL_R < _TABLE_LEFT)
        hit_right_wall = ball_active & (ball_x + _BALL_R > _TABLE_RIGHT)
        ball_dx = jnp.where(hit_left_wall | hit_right_wall, -ball_dx, ball_dx)
        ball_x = jnp.clip(ball_x, _TABLE_LEFT + _BALL_R, _TABLE_RIGHT - _BALL_R)

        hit_top = ball_active & (ball_y - _BALL_R < _TABLE_TOP)
        ball_dy = jnp.where(hit_top, jnp.abs(ball_dy), ball_dy)
        ball_y = jnp.where(hit_top, jnp.float32(_TABLE_TOP) + _BALL_R, ball_y)

        # --- Flipper collision ---
        # Left flipper: x ∈ [40, 65], y ∈ [185, 191]
        left_flip_hit = (
            ball_active
            & left_flipper_up
            & (ball_x + _BALL_R > _LEFT_FLIPPER_X0)
            & (ball_x - _BALL_R < _LEFT_FLIPPER_X1)
            & (ball_y + _BALL_R > _FLIPPER_Y)
            & (ball_y - _BALL_R < _FLIPPER_Y + _FLIPPER_H)
        )
        # Right flipper: x ∈ [95, 120], y ∈ [185, 191]
        right_flip_hit = (
            ball_active
            & right_flipper_up
            & (ball_x + _BALL_R > _RIGHT_FLIPPER_X0)
            & (ball_x - _BALL_R < _RIGHT_FLIPPER_X1)
            & (ball_y + _BALL_R > _FLIPPER_Y)
            & (ball_y - _BALL_R < _FLIPPER_Y + _FLIPPER_H)
        )
        any_flip = left_flip_hit | right_flip_hit
        ball_dy = jnp.where(
            any_flip, -jnp.abs(ball_dy) - jnp.float32(_FLIPPER_KICK), ball_dy
        )
        ball_y = jnp.where(any_flip, jnp.float32(_FLIPPER_Y) - _BALL_R, ball_y)

        # --- Bumper collisions ---
        step_reward = jnp.float32(0.0)
        new_bumper_active = state.bumper_active.copy()

        # Bumper collisions: use Python-literal centres to keep values concrete
        # Reflect off bumper surface then blend 50% toward cluster centre to model
        # the ROM spring-bumper resonance that keeps the ball in the bumper cluster.
        for i in range(_N_BUMPERS):
            bx_f = jnp.float32(_BUMPER_CENTERS[i][0])
            by_f = jnp.float32(_BUMPER_CENTERS[i][1])
            dist_sq = (ball_x - bx_f) ** 2 + (ball_y - by_f) ** 2
            hit = (
                ball_active
                & new_bumper_active[i]
                & (dist_sq < jnp.float32((_BALL_R + _BUMPER_R) ** 2))
            )
            nx = ball_x - bx_f
            ny = ball_y - by_f
            norm = jnp.sqrt(dist_sq + jnp.float32(1e-6))
            nx_n = nx / norm
            ny_n = ny / norm
            dot = ball_dx * nx_n + ball_dy * ny_n
            reflected_dx = ball_dx - 2.0 * dot * nx_n
            reflected_dy = ball_dy - 2.0 * dot * ny_n
            # Blend reflected direction with toward-cluster direction, then apply fixed kick
            # speed to model the ROM spring-bumper impulse (constant force, not reflection-scaled).
            refl_norm = jnp.sqrt(reflected_dx ** 2 + reflected_dy ** 2 + jnp.float32(1e-6))
            r_nx = reflected_dx / refl_norm
            r_ny = reflected_dy / refl_norm
            dcx = jnp.float32(_BUMPER_CLUSTER_CX) - ball_x
            dcy = jnp.float32(_BUMPER_CLUSTER_CY) - ball_y
            dc_norm = jnp.sqrt(dcx ** 2 + dcy ** 2 + jnp.float32(1e-6))
            dc_nx = dcx / dc_norm
            dc_ny = dcy / dc_norm
            bias = jnp.float32(_BUMPER_CLUSTER_BIAS)
            blend_nx = (jnp.float32(1.0) - bias) * r_nx + bias * dc_nx
            blend_ny = (jnp.float32(1.0) - bias) * r_ny + bias * dc_ny
            blend_norm = jnp.sqrt(blend_nx ** 2 + blend_ny ** 2 + jnp.float32(1e-6))
            kick = jnp.float32(_BUMPER_KICK_SPEED)
            ball_dx = jnp.where(hit, blend_nx / blend_norm * kick, ball_dx)
            ball_dy = jnp.where(hit, blend_ny / blend_norm * kick, ball_dy)
            step_reward = step_reward + jnp.where(
                hit, jnp.float32(_BUMPER_POINTS), jnp.float32(0.0)
            )

        # Target collisions: use Python-literal origins to keep values concrete
        new_target_hit = state.target_hit
        for i in range(_N_TARGETS):
            tx_f = jnp.float32(_TARGET_ORIGINS[i][0])
            ty_f = jnp.float32(_TARGET_ORIGINS[i][1])
            hit = (
                ball_active
                & ~new_target_hit[i]
                & (ball_x + _BALL_R > tx_f)
                & (ball_x - _BALL_R < tx_f + _TARGET_W)
                & (ball_y + _BALL_R > ty_f)
                & (ball_y - _BALL_R < ty_f + _TARGET_H)
            )
            new_target_hit = new_target_hit.at[i].set(new_target_hit[i] | hit)
            step_reward = step_reward + jnp.where(
                hit, jnp.float32(_TARGET_POINTS), jnp.float32(0.0)
            )

        # All targets hit → reset targets and bumpers
        all_targets_hit = jnp.all(new_target_hit)
        new_target_hit = jnp.where(
            all_targets_hit, jnp.zeros(_N_TARGETS, dtype=jnp.bool_), new_target_hit
        )
        new_bumper_active = jnp.where(
            all_targets_hit, jnp.ones(_N_BUMPERS, dtype=jnp.bool_), new_bumper_active
        )

        # --- Extra ball (ROM: RAM[0xA8] & 0x1) — awarded once on first target clear ---
        award_extra = all_targets_hit & ~state.extra_ball_awarded
        new_extra_ball_awarded = state.extra_ball_awarded | award_extra

        # --- Ball drain ---
        drained = ball_active & (ball_y + _BALL_R > _TABLE_BOTTOM)
        new_lives = (
            state.lives
            + jnp.where(award_extra, jnp.int32(1), jnp.int32(0))
            - jnp.where(drained, jnp.int32(1), jnp.int32(0))
        )
        ball_active = ball_active & ~drained
        ball_x = jnp.where(drained, jnp.float32(_BALL_LAUNCH_X), ball_x)
        ball_y = jnp.where(drained, jnp.float32(_BALL_LAUNCH_Y), ball_y)
        ball_dx = jnp.where(drained, jnp.float32(0.0), ball_dx)
        ball_dy = jnp.where(drained, jnp.float32(0.0), ball_dy)

        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            ball_x=ball_x,
            ball_y=ball_y,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            ball_active=ball_active,
            left_flipper_up=left_flipper_up,
            right_flipper_up=right_flipper_up,
            bumper_active=new_bumper_active,
            target_hit=new_target_hit,
            plunger_power=new_plunger,
            extra_ball_awarded=new_extra_ball_awarded,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: VideoPinballState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> VideoPinballState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: VideoPinballState) -> jax.Array:
        frame = jnp.full((210, 160, 3), 20, dtype=jnp.uint8)
        frame = frame.at[:, :, 2].set(jnp.uint8(50))

        # --- Walls ---
        left_wall = _COL_IDX == int(_TABLE_LEFT)
        right_wall = _COL_IDX == int(_TABLE_RIGHT)
        wall_mask = left_wall | right_wall
        frame = jnp.where(wall_mask[:, :, None], _WALL_COLOR[None, None, :], frame)

        # --- Flippers ---
        left_flip_mask = (
            (_ROW_IDX >= int(_FLIPPER_Y))
            & (_ROW_IDX < int(_FLIPPER_Y + _FLIPPER_H))
            & (_COL_IDX >= int(_LEFT_FLIPPER_X0))
            & (_COL_IDX < int(_LEFT_FLIPPER_X1))
        )
        right_flip_mask = (
            (_ROW_IDX >= int(_FLIPPER_Y))
            & (_ROW_IDX < int(_FLIPPER_Y + _FLIPPER_H))
            & (_COL_IDX >= int(_RIGHT_FLIPPER_X0))
            & (_COL_IDX < int(_RIGHT_FLIPPER_X1))
        )
        left_color = jnp.where(
            state.left_flipper_up,
            jnp.array([200, 240, 255], dtype=jnp.uint8),
            _FLIPPER_COLOR,
        )
        right_color = jnp.where(
            state.right_flipper_up,
            jnp.array([200, 240, 255], dtype=jnp.uint8),
            _FLIPPER_COLOR,
        )
        frame = jnp.where(left_flip_mask[:, :, None], left_color[None, None, :], frame)
        frame = jnp.where(
            right_flip_mask[:, :, None], right_color[None, None, :], frame
        )

        # --- Bumpers ---
        for i in range(_N_BUMPERS):
            bx, by = _BUMPER_CENTERS_I[i]
            br = _BUMPER_R_I
            bump_mask = (
                state.bumper_active[i]
                & (_ROW_IDX >= by - br)
                & (_ROW_IDX < by + br)
                & (_COL_IDX >= bx - br)
                & (_COL_IDX < bx + br)
            )
            frame = jnp.where(
                bump_mask[:, :, None], _BUMPER_COLOR[None, None, :], frame
            )

        # --- Targets ---
        _TW = int(_TARGET_W)
        _TH = int(_TARGET_H)
        for i in range(_N_TARGETS):
            tx = int(_TARGET_ORIGINS[i][0])
            ty = int(_TARGET_ORIGINS[i][1])
            color = jnp.where(state.target_hit[i], _TARGET_HIT_COLOR, _TARGET_COLOR)
            tgt_mask = (
                (_ROW_IDX >= ty)
                & (_ROW_IDX < ty + _TH)
                & (_COL_IDX >= tx)
                & (_COL_IDX < tx + _TW)
            )
            frame = jnp.where(tgt_mask[:, :, None], color[None, None, :], frame)

        # --- Ball ---
        bx = jnp.int32(state.ball_x)
        by = jnp.int32(state.ball_y)
        br = int(_BALL_R)
        ball_mask = (
            state.ball_active
            & (_ROW_IDX >= by - br)
            & (_ROW_IDX < by + br)
            & (_COL_IDX >= bx - br)
            & (_COL_IDX < bx + br)
        )
        frame = jnp.where(ball_mask[:, :, None], _BALL_COLOR[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,
                pygame.K_LEFT: 4,
                pygame.K_a: 4,
                pygame.K_RIGHT: 3,
                pygame.K_d: 3,
            }
        except ImportError:
            return {}
