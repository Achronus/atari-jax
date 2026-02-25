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

"""Freeway — JAX-native game implementation.

Guide a chicken across 10 lanes of traffic to the opposite side of the
screen.  Each successful crossing earns +1.  Cars push the chicken back
on contact rather than costing a life.  The episode ends after a fixed
time limit handled by `AtariEnv` (`max_episode_steps = 1600`, i.e. 400
agent steps × 4 sub-steps).

Action space (3 actions):
    0 — NOOP
    1 — UP   (move toward the goal)
    2 — DOWN (move back toward start)
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_LANE_H: int = 18  # pixels per traffic lane
_N_LANES: int = 10  # number of traffic lanes
_ROAD_TOP: float = 15.0  # y of topmost lane top edge
_ROAD_BOT: float = 195.0  # y of bottommost lane bottom edge
_GOAL_Y: float = 15.0  # chicken crosses when chicken_y <= this
_START_Y: float = 192.0  # starting y (bottom lane centre)
_CHICKEN_X: int = 76  # left edge of chicken (centred at x=80)
_CHICKEN_W: int = 8  # chicken sprite width / height
_CAR_W: int = 16  # car sprite width
_CAR_H: int = 8  # car sprite height

# Per-lane car speeds (px / sub-step); positive = rightward
_CAR_SPEEDS = jnp.array(
    [2.0, -2.5, 1.5, -3.0, 2.5, -1.5, 3.0, -2.0, 1.5, -2.5],
    dtype=jnp.float32,
)

# Initial car x positions (left edge)
_CAR_STARTS = jnp.array(
    [10.0, 90.0, 40.0, 120.0, 5.0, 70.0, 100.0, 20.0, 60.0, 140.0],
    dtype=jnp.float32,
)

# Car y-positions (top edge) per lane
_CAR_LANE_Y = jnp.array(
    [int(_ROAD_TOP) + i * _LANE_H + 5 for i in range(_N_LANES)],
    dtype=jnp.int32,
)

# Lane divider y-positions (rows between lanes)
_LANE_DIV_YS = jnp.array(
    [int(_ROAD_TOP) + i * _LANE_H for i in range(1, _N_LANES)],
    dtype=jnp.int32,
)

# Precomputed index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]

# Colours
_COLOR_BG = jnp.array([0, 100, 0], dtype=jnp.uint8)  # green grass
_COLOR_ROAD = jnp.array([80, 80, 80], dtype=jnp.uint8)  # grey road
_COLOR_DIVIDER = jnp.array([200, 200, 200], dtype=jnp.uint8)  # lane dividers
_COLOR_CAR = jnp.array([180, 40, 40], dtype=jnp.uint8)  # red cars
_COLOR_CHICKEN = jnp.array([255, 255, 80], dtype=jnp.uint8)  # yellow chicken


@chex.dataclass
class FreewayState(AtariState):
    """
    Complete Freeway game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score` from `AtariState`.

    Parameters
    ----------
    chicken_y : jax.Array
        float32 — Vertical position of the chicken (195=start, ≤15=crossed).
    car_x : jax.Array
        float32[10] — Left-edge x position of each lane's car.
    """

    chicken_y: jax.Array  # float32
    car_x: jax.Array  # float32[10]


class Freeway(AtariEnv):
    """
    Freeway implemented as a pure JAX function suite.

    Guide the chicken across 10 lanes of traffic.  +1 reward per crossing.
    Cars push the chicken back on contact; no lives.  Episode ends after the
    time limit (`max_episode_steps = 1600` emulated frames = 400 agent steps).
    """

    num_actions: int = 3

    def __init__(self, params: EnvParams | None = None) -> None:
        super().__init__(params or EnvParams(noop_max=0, max_episode_steps=1600))

    def _reset(self, key: jax.Array) -> FreewayState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : FreewayState
            Chicken at starting row, cars at initial positions.
        """
        return FreewayState(
            chicken_y=jnp.float32(_START_Y),
            car_x=_CAR_STARTS,
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: FreewayState, action: jax.Array) -> FreewayState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : FreewayState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=UP, 2=DOWN.

        Returns
        -------
        new_state : FreewayState
            State after one emulated frame.
        """
        # Chicken vertical movement: 4.5 px per sub-step = 18 px per agent step
        dy = jnp.where(
            action == jnp.int32(1),
            jnp.float32(-4.5),
            jnp.where(action == jnp.int32(2), jnp.float32(4.5), jnp.float32(0.0)),
        )
        new_y = jnp.clip(state.chicken_y + dy, _GOAL_Y, _ROAD_BOT)

        # Cars advance, wrapping at x=160
        new_car_x = (state.car_x + _CAR_SPEEDS) % jnp.float32(160.0)

        # Collision: which lane is the chicken in?
        in_road = (new_y >= _ROAD_TOP) & (new_y < _ROAD_BOT)
        lane_idx = jnp.clip(jnp.int32((new_y - _ROAD_TOP) / _LANE_H), 0, _N_LANES - 1)
        chicken_cx = jnp.float32(_CHICKEN_X) + jnp.float32(_CHICKEN_W) / 2.0
        lanes = jnp.arange(_N_LANES)
        in_lane = lanes == lane_idx  # bool[10]
        car_r = new_car_x + jnp.float32(_CAR_W)  # float32[10]
        overlap_x = (chicken_cx >= new_car_x) & (chicken_cx <= car_r)
        hit = in_lane & overlap_x & in_road
        any_hit = jnp.any(hit)

        # Push chicken back one lane on collision
        new_y = jnp.where(
            any_hit,
            jnp.clip(new_y + jnp.float32(_LANE_H), _ROAD_TOP, _ROAD_BOT),
            new_y,
        )

        # Crossing: chicken reaches the top goal
        crossed = new_y <= _GOAL_Y + jnp.float32(0.5)
        step_reward = jnp.where(crossed, jnp.float32(1.0), jnp.float32(0.0))
        new_y = jnp.where(crossed, jnp.float32(_START_Y), new_y)

        return FreewayState(
            chicken_y=new_y,
            car_x=new_car_x,
            lives=jnp.int32(0),
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=jnp.bool_(False),
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: FreewayState, action: jax.Array) -> FreewayState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : FreewayState
            Current game state.
        action : jax.Array
            int32 — Action index (0–2).

        Returns
        -------
        new_state : FreewayState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: FreewayState) -> FreewayState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: FreewayState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : FreewayState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)

        # Grass (top and bottom)
        in_grass = (_ROW_IDX < jnp.int32(_ROAD_TOP)) | (
            _ROW_IDX >= jnp.int32(_ROAD_BOT)
        )
        frame = jnp.where(in_grass[:, :, None], _COLOR_BG, frame)

        # Road
        in_road = (_ROW_IDX >= jnp.int32(_ROAD_TOP)) & (_ROW_IDX < jnp.int32(_ROAD_BOT))
        frame = jnp.where(in_road[:, :, None], _COLOR_ROAD, frame)

        # Lane dividers — broadcast [9, 1, 1] against _ROW_IDX [210, 1]
        div_mask = jnp.any(
            _ROW_IDX[None, :, :] == _LANE_DIV_YS[:, None, None], axis=0
        )  # [210, 1]
        frame = jnp.where(
            div_mask[:, :, None] & in_road[:, :, None], _COLOR_DIVIDER, frame
        )

        # Cars — draw one per lane using lax.scan
        def draw_car(frm, i):
            cx = state.car_x[i]
            cy = _CAR_LANE_Y[i]
            mask = (
                (_ROW_IDX >= cy)
                & (_ROW_IDX < cy + _CAR_H)
                & (_COL_IDX >= jnp.int32(cx))
                & (_COL_IDX < jnp.int32(cx) + _CAR_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_CAR, frm), None

        frame, _ = jax.lax.scan(draw_car, frame, jnp.arange(_N_LANES))

        # Chicken
        cy_i = jnp.int32(state.chicken_y) - 4
        chicken_mask = (
            (_ROW_IDX >= cy_i)
            & (_ROW_IDX < cy_i + _CHICKEN_W)
            & (_COL_IDX >= _CHICKEN_X)
            & (_COL_IDX < _CHICKEN_X + _CHICKEN_W)
        )
        frame = jnp.where(chicken_mask[:, :, None], _COLOR_CHICKEN, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Freeway action indices.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_DOWN: 2,
            pygame.K_s: 2,
        }
