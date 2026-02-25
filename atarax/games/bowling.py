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

"""Bowling — JAX-native game implementation.

Roll a bowling ball to knock down 10 pins at the far end of a lane.  The
player can curve the ball left or right before release.  Each set of 10 pins
is one frame; the episode ends after 10 frames (or `max_episode_steps`).

Action space (4 actions):
    0 — NOOP / hold
    1 — FIRE (release ball)
    2 — LEFT  (curve ball or step left before release)
    3 — RIGHT (curve ball or step right before release)

Scoring:
    Strike (all 10 with first ball)  — 10 + bonus (simplified: 20 pts)
    Spare  (all 10 with two balls)   — 10 + bonus (simplified: 15 pts)
    Open   (fewer than 10)           — number of pins knocked down
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_LANE_X0: int = 60  # left edge of lane
_LANE_X1: int = 100  # right edge of lane
_LANE_W: int = 40  # lane width
_LANE_MID: float = 80.0

_BOWLER_Y: float = 190.0
_PIN_Y: float = 30.0
_BALL_SPEED: float = 6.0  # px per sub-step upward
_BALL_W: int = 4
_PIN_W: int = 3
_PIN_H: int = 4

# 10-pin triangle arrangement: (x offset from lane centre, relative)
_N_PINS: int = 10
_PIN_OFFSETS_X = jnp.array(
    [-9.0, -3.0, 3.0, 9.0, -6.0, 0.0, 6.0, -3.0, 3.0, 0.0], dtype=jnp.float32
)
_PIN_OFFSETS_Y = jnp.array(
    [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 10.0, 10.0, 15.0], dtype=jnp.float32
)
_PIN_X = _LANE_MID + _PIN_OFFSETS_X  # [10]
_PIN_Y_ARR = _PIN_Y + _PIN_OFFSETS_Y  # [10]

_N_FRAMES: int = 10  # bowling frames per episode
_MAX_STEPS: int = 4000  # sub-steps per episode

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 10, 5], dtype=jnp.uint8)
_COLOR_LANE = jnp.array([200, 160, 100], dtype=jnp.uint8)
_COLOR_BALL = jnp.array([0, 0, 180], dtype=jnp.uint8)
_COLOR_PIN = jnp.array([240, 240, 240], dtype=jnp.uint8)
_COLOR_BOWLER = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_GUTTER = jnp.array([80, 50, 30], dtype=jnp.uint8)


@chex.dataclass
class BowlingState(AtariState):
    """
    Complete Bowling game state — a JAX pytree.

    Parameters
    ----------
    bowler_x : jax.Array
        float32 — Bowler x position (left edge).
    ball_x : jax.Array
        float32 — Ball x position.
    ball_y : jax.Array
        float32 — Ball y position (195=start, 30=pins).
    ball_dx : jax.Array
        float32 — Ball x velocity (curve).
    ball_active : jax.Array
        bool — Ball is rolling.
    pins : jax.Array
        bool[10] — Standing pins.
    frames_left : jax.Array
        int32 — Bowling frames remaining.
    roll_in_frame : jax.Array
        int32 — Current roll within this frame (0 or 1).
    reset_timer : jax.Array
        int32 — Sub-steps until next frame begins.
    """

    bowler_x: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_active: jax.Array
    pins: jax.Array
    frames_left: jax.Array
    roll_in_frame: jax.Array
    reset_timer: jax.Array


class Bowling(AtariEnv):
    """
    Bowling implemented as a pure JAX function suite.

    Roll a bowling ball down the lane to knock down 10 pins.  The game
    runs for 10 frames; each frame allows up to 2 rolls (1 if a strike).
    """

    num_actions: int = 4

    def __init__(self, params: EnvParams | None = None) -> None:
        super().__init__(params or EnvParams(noop_max=0, max_episode_steps=4000))

    def _reset(self, key: jax.Array) -> BowlingState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BowlingState
            Full pin set, bowler at centre, 10 frames remaining.
        """
        return BowlingState(
            bowler_x=jnp.float32(_LANE_MID - 4),
            ball_x=jnp.float32(_LANE_MID),
            ball_y=jnp.float32(_BOWLER_Y),
            ball_dx=jnp.float32(0.0),
            ball_active=jnp.bool_(False),
            pins=jnp.ones(_N_PINS, dtype=jnp.bool_),
            frames_left=jnp.int32(_N_FRAMES),
            roll_in_frame=jnp.int32(0),
            reset_timer=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: BowlingState, action: jax.Array) -> BowlingState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BowlingState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE, 2=LEFT, 3=RIGHT.

        Returns
        -------
        new_state : BowlingState
            State after one emulated frame.
        """
        step_reward = jnp.float32(0.0)

        # --- Reset timer: wait between frames ---
        in_reset = state.reset_timer > jnp.int32(0)
        new_reset_timer = jnp.where(
            in_reset, state.reset_timer - jnp.int32(1), jnp.int32(0)
        )

        # --- Pre-release: move bowler, set curve ---
        move_dx = jnp.where(
            action == jnp.int32(2),
            jnp.float32(-1.0),
            jnp.where(action == jnp.int32(3), jnp.float32(1.0), jnp.float32(0.0)),
        )
        new_bowler_x = jnp.where(
            ~state.ball_active & ~in_reset,
            jnp.clip(state.bowler_x + move_dx, float(_LANE_X0), float(_LANE_X1 - 8)),
            state.bowler_x,
        )

        # Set curve direction on release
        new_ball_dx = jnp.where(
            action == jnp.int32(2),
            jnp.float32(-0.5),
            jnp.where(action == jnp.int32(3), jnp.float32(0.5), state.ball_dx),
        )

        # --- Release ball ---
        fire = (action == jnp.int32(1)) & ~state.ball_active & ~in_reset
        new_ball_x = jnp.where(fire, new_bowler_x + jnp.float32(4), state.ball_x)
        new_ball_y = jnp.where(fire, jnp.float32(_BOWLER_Y), state.ball_y)
        new_ball_active = state.ball_active | fire

        # --- Ball movement ---
        roll_x = jnp.where(new_ball_active, new_ball_x + new_ball_dx, new_ball_x)
        roll_y = jnp.where(new_ball_active, new_ball_y - _BALL_SPEED, new_ball_y)
        # Clamp x to lane
        roll_x = jnp.clip(roll_x, float(_LANE_X0), float(_LANE_X1 - _BALL_W))

        # --- Pin collision ---
        ball_cx = roll_x + jnp.float32(_BALL_W) / 2.0
        ball_cy = roll_y
        dist_x = jnp.abs(_PIN_X - ball_cx)  # [10]
        dist_y = jnp.abs(_PIN_Y_ARR - ball_cy)  # [10]
        hit_pin = (
            (dist_x < jnp.float32(6)) & (dist_y < jnp.float32(6)) & new_ball_active
        )
        newly_knocked = hit_pin & state.pins  # [10]
        new_pins = state.pins & ~newly_knocked
        n_knocked = jnp.sum(newly_knocked).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_knocked)

        # --- Ball reaches the pins or passes them ---
        past_pins = roll_y <= _PIN_Y - jnp.float32(10)
        all_knocked = ~jnp.any(new_pins)
        frame_done = new_ball_active & (past_pins | all_knocked)

        # Strike on first roll
        is_strike = (
            (state.roll_in_frame == jnp.int32(0))
            & all_knocked
            & new_ball_active
            & frame_done
        )
        step_reward = step_reward + jnp.where(
            is_strike, jnp.float32(10.0), jnp.float32(0.0)
        )

        # Spare on second roll
        is_spare = (
            (state.roll_in_frame == jnp.int32(1))
            & all_knocked
            & new_ball_active
            & frame_done
        )
        step_reward = step_reward + jnp.where(
            is_spare, jnp.float32(5.0), jnp.float32(0.0)
        )

        new_roll_in_frame = jnp.where(
            frame_done,
            jnp.where(
                is_strike | (state.roll_in_frame == jnp.int32(1)),
                jnp.int32(0),
                jnp.int32(1),
            ),
            state.roll_in_frame,
        )
        next_frame = is_strike | (state.roll_in_frame == jnp.int32(1))
        new_frames_left = jnp.where(
            frame_done & next_frame, state.frames_left - jnp.int32(1), state.frames_left
        )
        new_pins_reset = jnp.where(
            frame_done & next_frame, jnp.ones(_N_PINS, dtype=jnp.bool_), new_pins
        )
        new_reset_timer = jnp.where(frame_done, jnp.int32(20), new_reset_timer)

        # Reset ball after frame
        new_ball_active = jnp.where(frame_done, jnp.bool_(False), new_ball_active)
        new_ball_x = jnp.where(frame_done, new_bowler_x + jnp.float32(4), roll_x)
        new_ball_y = jnp.where(frame_done, jnp.float32(_BOWLER_Y), roll_y)

        done = new_frames_left <= jnp.int32(0)

        return BowlingState(
            bowler_x=new_bowler_x,
            ball_x=new_ball_x,
            ball_y=new_ball_y,
            ball_dx=new_ball_dx,
            ball_active=new_ball_active,
            pins=new_pins_reset,
            frames_left=new_frames_left,
            roll_in_frame=new_roll_in_frame,
            reset_timer=new_reset_timer,
            lives=jnp.int32(0),
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: BowlingState, action: jax.Array) -> BowlingState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BowlingState
            Current game state.
        action : jax.Array
            int32 — Action index (0–3).

        Returns
        -------
        new_state : BowlingState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BowlingState) -> BowlingState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BowlingState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BowlingState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)
        frame = jnp.where(jnp.ones((210, 160, 1), dtype=jnp.bool_), _COLOR_BG, frame)

        # Lane
        in_lane = (_COL_IDX >= _LANE_X0) & (_COL_IDX < _LANE_X1)
        frame = jnp.where(in_lane[:, :, None], _COLOR_LANE, frame)

        # Gutters
        gutter_l = (_COL_IDX >= _LANE_X0 - 6) & (_COL_IDX < _LANE_X0)
        gutter_r = (_COL_IDX >= _LANE_X1) & (_COL_IDX < _LANE_X1 + 6)
        frame = jnp.where((gutter_l | gutter_r)[:, :, None], _COLOR_GUTTER, frame)

        # Pins
        def draw_pin(frm, i):
            px = _PIN_X[i]
            py = _PIN_Y_ARR[i]
            standing = state.pins[i]
            mask = (
                (_ROW_IDX >= jnp.int32(py))
                & (_ROW_IDX < jnp.int32(py) + _PIN_H)
                & (_COL_IDX >= jnp.int32(px))
                & (_COL_IDX < jnp.int32(px) + _PIN_W)
                & standing
            )
            return jnp.where(mask[:, :, None], _COLOR_PIN, frm), None

        frame, _ = jax.lax.scan(draw_pin, frame, jnp.arange(_N_PINS))

        # Ball
        ball_mask = (
            state.ball_active
            & (_ROW_IDX >= jnp.int32(state.ball_y))
            & (_ROW_IDX < jnp.int32(state.ball_y) + _BALL_W)
            & (_COL_IDX >= jnp.int32(state.ball_x))
            & (_COL_IDX < jnp.int32(state.ball_x) + _BALL_W)
        )
        frame = jnp.where(ball_mask[:, :, None], _COLOR_BALL, frame)

        # Bowler
        b_mask = (
            ~state.ball_active
            & (_ROW_IDX >= jnp.int32(_BOWLER_Y) - 8)
            & (_ROW_IDX < jnp.int32(_BOWLER_Y))
            & (_COL_IDX >= jnp.int32(state.bowler_x))
            & (_COL_IDX < jnp.int32(state.bowler_x) + 8)
        )
        frame = jnp.where(b_mask[:, :, None], _COLOR_BOWLER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Bowling action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_LEFT: 2,
            pygame.K_a: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
        }
