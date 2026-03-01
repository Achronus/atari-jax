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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Road       : y ∈ [15, 195],  10 lanes each 18 px tall
    Chicken    : 8×8 px, fixed x=76, starts y=192
    Goal       : y <= 15 (top of road)
    Cars       : 16×8 px, one per lane, wrap horizontally

Action space (3 actions):
    0 — NOOP
    1 — UP   (move chicken toward goal, y decreases)
    2 — DOWN (move chicken toward start, y increases)
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry constants
_PLAY_LEFT: int = 0
_PLAY_RIGHT: int = 160
_ROAD_Y0: int = 15       # top edge of lane 0 (goal boundary)
_N_LANES: int = 10
_LANE_H: int = 18
_GOAL_Y: int = 15        # chicken reaches goal when chicken_y <= this
_CHICKEN_X: int = 76
_CHICKEN_W: int = 8
_CHICKEN_H: int = 8
_CHICKEN_START_Y: int = 192
_CAR_W: int = 16
_CAR_H: int = 8
_MAX_STEPS: int = 400    # agent steps (1600 emulated frames / 4-frame skip)
_FRAME_SKIP: int = 4

# Fixed alternating speeds per lane (positive = right, negative = left)
_CAR_SPEEDS = jnp.array([2., -2., 3., -3., 2., -2., 3., -3., 2., -2.], dtype=jnp.float32)

# Car y positions — top edge of car, centred in each lane
_CAR_LANE_Y = jnp.array(
    [_ROAD_Y0 + i * _LANE_H + (_LANE_H - _CAR_H) // 2 for i in range(_N_LANES)],
    dtype=jnp.float32,
)

# Car lane colours (alternating yellow / red)
_CAR_COLORS = jnp.array(
    [
        [210, 210, 0],
        [200, 72, 72],
        [210, 210, 0],
        [200, 72, 72],
        [210, 210, 0],
        [200, 72, 72],
        [210, 210, 0],
        [200, 72, 72],
        [210, 210, 0],
        [200, 72, 72],
    ],
    dtype=jnp.uint8,
)

_CHICKEN_COLOR = jnp.array([252, 252, 84], dtype=jnp.uint8)  # bright yellow
_ROAD_COLOR = jnp.array([80, 80, 80], dtype=jnp.uint8)
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
    """Blit a 3×5 digit glyph onto *frame* at pixel (x0, y0)."""
    glyph = _DIGIT_FONT[digit]  # bool[5, 3]
    dr = jnp.clip(_ROW_IDX - y0, 0, 4)
    dc = jnp.clip(_COL_IDX - x0, 0, 2)
    in_box = (
        (_ROW_IDX >= y0) & (_ROW_IDX < y0 + 5) & (_COL_IDX >= x0) & (_COL_IDX < x0 + 3)
    )
    lit = glyph[dr, dc]
    return jnp.where((in_box & lit)[:, :, None], _SCORE_COLOR, frame)


@chex.dataclass
class FreewayState(AtariState):
    """
    Complete Freeway game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    Parameters
    ----------
    chicken_y : jax.Array
        float32 — Top-edge y coordinate of the chicken.
        Starts at 192; decreases toward goal (y <= 15).
    car_x : jax.Array
        float32[10] — Left-edge x coordinate of each car (one per lane).
        Cars wrap horizontally as they move.
    """

    chicken_y: jax.Array
    car_x: jax.Array


class Freeway(AtaraxGame):
    """
    Freeway implemented as a pure-JAX function suite.

    Physics: chicken moves one pixel per emulated frame up or down. Cars move
    at fixed speeds and wrap at screen edges. A car collision pushes the chicken
    back by one lane. Reaching the goal (y <= 15) scores +1 and resets the
    chicken. Episode ends after 400 agent steps.
    """

    num_actions: int = 3

    def _reset(self, key: chex.PRNGKey) -> FreewayState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : chex.PRNGKey
            JAX PRNG key (used to randomise initial car positions).

        Returns
        -------
        state : FreewayState
            Chicken at start position, cars spread across the road.
        """
        car_x = jax.random.uniform(
            key, shape=(_N_LANES,), minval=_PLAY_LEFT, maxval=_PLAY_RIGHT, dtype=jnp.float32
        )
        return FreewayState(
            chicken_y=jnp.float32(_CHICKEN_START_Y),
            car_x=car_x,
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: FreewayState, action: jax.Array) -> FreewayState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : FreewayState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=UP, 2=DOWN.

        Returns
        -------
        new_state : FreewayState
            State after one emulated frame. `episode_step` is NOT incremented
            here — it is incremented once per agent step in `_step`.
        """
        # Move cars (wrap at screen edges)
        new_car_x = jnp.mod(
            state.car_x + _CAR_SPEEDS + jnp.float32(_PLAY_RIGHT),
            jnp.float32(_PLAY_RIGHT),
        )

        # Chicken movement
        dy = jnp.where(action == 1, jnp.float32(-1.0), jnp.where(action == 2, jnp.float32(1.0), jnp.float32(0.0)))
        new_chicken_y = jnp.clip(
            state.chicken_y + dy,
            jnp.float32(_GOAL_Y),
            jnp.float32(_CHICKEN_START_Y),
        )

        # Collision detection: chicken vs each car (vectorised over 10 lanes)
        x_overlap = (
            (jnp.float32(_CHICKEN_X) < new_car_x + jnp.float32(_CAR_W))
            & (jnp.float32(_CHICKEN_X + _CHICKEN_W) > new_car_x)
        )
        y_overlap = (
            (new_chicken_y < _CAR_LANE_Y + jnp.float32(_CAR_H))
            & (new_chicken_y + jnp.float32(_CHICKEN_H) > _CAR_LANE_Y)
        )
        hit_mask = x_overlap & y_overlap  # bool[10]
        any_hit = jnp.any(hit_mask)

        # Collision pushes chicken back one lane (clamped to start)
        new_chicken_y = jnp.where(
            any_hit,
            jnp.minimum(new_chicken_y + jnp.float32(_LANE_H), jnp.float32(_CHICKEN_START_Y)),
            new_chicken_y,
        )

        # Goal check: chicken reached the far side
        scored = new_chicken_y <= jnp.float32(_GOAL_Y)
        step_reward = jnp.where(scored, jnp.float32(1.0), jnp.float32(0.0))
        new_score = state.score + jnp.where(scored, jnp.int32(1), jnp.int32(0))
        new_chicken_y = jnp.where(scored, jnp.float32(_CHICKEN_START_Y), new_chicken_y)

        return FreewayState(
            chicken_y=new_chicken_y,
            car_x=new_car_x,
            lives=jnp.int32(0),
            score=new_score,
            level=state.level,
            reward=state.reward + step_reward,
            done=state.done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step,  # incremented once per agent step in _step
            key=state.key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: FreewayState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> FreewayState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : FreewayState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=UP, 2=DOWN).
        params : AtaraxParams
            Static environment parameters.

        Returns
        -------
        new_state : FreewayState
            State after 4 emulated frames with `episode_step` incremented once.
        """
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        new_episode_step = state.episode_step + jnp.int32(1)
        done = new_episode_step >= jnp.int32(_MAX_STEPS)
        return new_state.__replace__(episode_step=new_episode_step, done=done)

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
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Road lanes (gray stripes)
        road_mask = (_ROW_IDX >= _ROAD_Y0) & (_ROW_IDX < _ROAD_Y0 + _N_LANES * _LANE_H)
        frame = jnp.where(road_mask[:, :, None], _ROAD_COLOR, frame)

        # Lane dividers (slightly darker stripe at every lane boundary)
        def _draw_cars(frame: jax.Array, i: int) -> jax.Array:
            cx = jnp.int32(state.car_x[i])
            car_y = jnp.int32(_CAR_LANE_Y[i])
            car_mask = (
                (_ROW_IDX >= car_y)
                & (_ROW_IDX < car_y + _CAR_H)
                & (_COL_IDX >= cx)
                & (_COL_IDX < cx + _CAR_W)
            )
            return jnp.where(car_mask[:, :, None], _CAR_COLORS[i], frame)

        for i in range(_N_LANES):
            frame = _draw_cars(frame, i)

        # Chicken
        cy = jnp.int32(state.chicken_y)
        chicken_mask = (
            (_ROW_IDX >= cy)
            & (_ROW_IDX < cy + _CHICKEN_H)
            & (_COL_IDX >= _CHICKEN_X)
            & (_COL_IDX < _CHICKEN_X + _CHICKEN_W)
        )
        frame = jnp.where(chicken_mask[:, :, None], _CHICKEN_COLOR, frame)

        # Score (2-digit)
        score = state.score
        frame = _blit_digit(frame, (score // 10) % 10, x0=8, y0=4)
        frame = _blit_digit(frame, score % 10, x0=12, y0=4)

        return frame

    def _key_map(self) -> dict:
        """Return the key-to-action mapping for interactive play."""
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_DOWN: 2,
            pygame.K_s: 2,
        }
