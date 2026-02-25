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

"""Enduro — JAX-native game implementation.

Race a car through traffic across multiple days.  Pass a required number
of opponent cars each day to advance.  Colliding with opponents slows
you down.

Action space (9 actions):
    0 — NOOP
    1 — FIRE  (accelerate)
    2 — RIGHT
    3 — LEFT
    4 — DOWN  (brake)
    5 — DOWN + RIGHT
    6 — DOWN + LEFT
    7 — RIGHT + FIRE
    8 — LEFT + FIRE

Scoring:
    +1 for every opponent car passed.
    Day advance bonus: +100 per day completed.
    Episode ends when the allotted time runs out; no lives system.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_CARS: int = 8  # concurrent opponent cars
_PLAYER_X_MIN: float = 30.0
_PLAYER_X_MAX: float = 130.0
_PLAYER_Y: int = 150  # player car y (fixed)
_PLAYER_SPEED_BASE: float = 2.0
_PLAYER_SPEED_MAX: float = 6.0
_OPPONENT_SPEED: float = 1.5  # opponent cars move up relative to player

_ROAD_LEFT: int = 20
_ROAD_RIGHT: int = 140
_SPAWN_Y: float = 30.0  # opponents appear here
_DESPAWN_Y: float = 190.0

_CARS_PER_DAY: int = 200
_DAY_FRAMES: int = 6000  # frames per day

_SPAWN_INTERVAL: int = 30
_INIT_LIVES: int = 0  # no lives system

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([80, 80, 80], dtype=jnp.uint8)  # road surface
_COLOR_GRASS = jnp.array([60, 100, 40], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_OPPONENT = jnp.array([220, 80, 80], dtype=jnp.uint8)
_COLOR_LINE = jnp.array([255, 255, 255], dtype=jnp.uint8)


@chex.dataclass
class EnduroState(AtariState):
    """
    Complete Enduro game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player car x (centre).
    player_speed : jax.Array
        float32 — Current forward speed.
    car_x : jax.Array
        float32[8] — Opponent car x positions.
    car_y : jax.Array
        float32[8] — Opponent car y positions (scroll space).
    car_active : jax.Array
        bool[8] — Opponent cars on track.
    cars_passed : jax.Array
        int32 — Cars passed today.
    day : jax.Array
        int32 — Current day (0-indexed).
    day_timer : jax.Array
        int32 — Frames remaining today.
    spawn_timer : jax.Array
        int32 — Frames until next opponent spawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_speed: jax.Array
    car_x: jax.Array
    car_y: jax.Array
    car_active: jax.Array
    cars_passed: jax.Array
    day: jax.Array
    day_timer: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class Enduro(AtariEnv):
    """
    Enduro implemented as a pure JAX function suite.

    Race through traffic; pass enough cars each day to continue.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> EnduroState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : EnduroState
            Player at centre, day 0, no opponents.
        """
        return EnduroState(
            player_x=jnp.float32(80.0),
            player_speed=jnp.float32(_PLAYER_SPEED_BASE),
            car_x=jnp.full(_N_CARS, 80.0, dtype=jnp.float32),
            car_y=jnp.full(_N_CARS, -20.0, dtype=jnp.float32),
            car_active=jnp.zeros(_N_CARS, dtype=jnp.bool_),
            cars_passed=jnp.int32(0),
            day=jnp.int32(0),
            day_timer=jnp.int32(_DAY_FRAMES),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            key=key,
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: EnduroState, action: jax.Array) -> EnduroState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : EnduroState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : EnduroState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Speed control
        accel = (
            (action == jnp.int32(1))
            | (action == jnp.int32(7))
            | (action == jnp.int32(8))
        )
        brake = (
            (action == jnp.int32(4))
            | (action == jnp.int32(5))
            | (action == jnp.int32(6))
        )
        new_speed = jnp.clip(
            state.player_speed
            + jnp.where(accel, 0.1, 0.0)
            + jnp.where(brake, -0.2, 0.0)
            - 0.02,  # natural deceleration
            _PLAYER_SPEED_BASE,
            _PLAYER_SPEED_MAX,
        )

        # Lateral movement
        steer_right = (
            (action == jnp.int32(2))
            | (action == jnp.int32(5))
            | (action == jnp.int32(7))
        )
        steer_left = (
            (action == jnp.int32(3))
            | (action == jnp.int32(6))
            | (action == jnp.int32(8))
        )
        new_px = jnp.clip(
            state.player_x
            + jnp.where(steer_right, 2.0, 0.0)
            + jnp.where(steer_left, -2.0, 0.0),
            _PLAYER_X_MIN,
            _PLAYER_X_MAX,
        )

        # Scroll opponents down (relative motion: player moving forward)
        scroll = new_speed - _OPPONENT_SPEED
        new_car_y = state.car_y + scroll
        # Opponent passes player (scrolled past bottom): score
        passed = state.car_active & (new_car_y > _DESPAWN_Y)
        n_passed = jnp.sum(passed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_passed)
        new_car_active = state.car_active & ~passed
        new_car_y = jnp.where(passed, jnp.float32(-20.0), new_car_y)

        # Spawn new opponent
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(
            do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )
        spawn_x = (
            jax.random.uniform(sk) * (float(_ROAD_RIGHT - _ROAD_LEFT)) + _ROAD_LEFT
        )
        free_slot = jnp.argmin(new_car_active.astype(jnp.int32))
        new_car_x = jnp.where(
            do_spawn,
            state.car_x.at[free_slot].set(spawn_x),
            state.car_x,
        )
        new_car_y = jnp.where(
            do_spawn,
            new_car_y.at[free_slot].set(_SPAWN_Y),
            new_car_y,
        )
        new_car_active = jnp.where(
            do_spawn,
            new_car_active.at[free_slot].set(True),
            new_car_active,
        )

        # Collision with opponent
        collision = (
            new_car_active
            & (jnp.abs(new_car_x - new_px) < 10.0)
            & (jnp.abs(new_car_y - jnp.float32(_PLAYER_Y)) < 12.0)
        )
        hit = jnp.any(collision)
        # Collision slows player down, no life loss
        new_speed = jnp.where(hit, _PLAYER_SPEED_BASE, new_speed)

        # Day progression
        new_cars_passed = state.cars_passed + n_passed
        new_day_timer = state.day_timer - jnp.int32(1)
        day_over = new_day_timer <= jnp.int32(0)
        # Advance day if enough cars passed
        day_success = day_over & (new_cars_passed >= jnp.int32(_CARS_PER_DAY))
        step_reward = step_reward + jnp.where(
            day_success, jnp.float32(100.0), jnp.float32(0.0)
        )
        new_day = state.day + jnp.where(day_success, jnp.int32(1), jnp.int32(0))
        new_cars_passed = jnp.where(day_success, jnp.int32(0), new_cars_passed)
        new_day_timer = jnp.where(day_over, jnp.int32(_DAY_FRAMES), new_day_timer)

        # Episode ends when day ends without enough cars
        done = day_over & ~day_success

        return EnduroState(
            player_x=new_px,
            player_speed=new_speed,
            car_x=new_car_x,
            car_y=new_car_y,
            car_active=new_car_active,
            cars_passed=new_cars_passed,
            day=new_day,
            day_timer=new_day_timer,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=jnp.int32(0),
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: EnduroState, action: jax.Array) -> EnduroState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : EnduroState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : EnduroState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: EnduroState) -> EnduroState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: EnduroState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : EnduroState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_GRASS, dtype=jnp.uint8)

        # Road strip
        road_mask = (_COL_IDX >= _ROAD_LEFT) & (_COL_IDX <= _ROAD_RIGHT)
        frame = jnp.where(road_mask[:, :, None], _COLOR_BG, frame)

        # Centre dashes
        dash_mask = road_mask & (_COL_IDX == 80) & ((_ROW_IDX % 20) < 10)
        frame = jnp.where(dash_mask[:, :, None], _COLOR_LINE, frame)

        # Opponent cars
        def draw_car(frm, i):
            cx = state.car_x[i].astype(jnp.int32)
            cy = state.car_y[i].astype(jnp.int32)
            mask = (
                state.car_active[i]
                & (_ROW_IDX >= cy - 8)
                & (_ROW_IDX <= cy + 8)
                & (_COL_IDX >= cx - 6)
                & (_COL_IDX <= cx + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_OPPONENT, frm), None

        frame, _ = jax.lax.scan(draw_car, frame, jnp.arange(_N_CARS))

        # Player car
        px = state.player_x.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= _PLAYER_Y - 8)
            & (_ROW_IDX <= _PLAYER_Y + 8)
            & (_COL_IDX >= px - 6)
            & (_COL_IDX <= px + 6)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Enduro action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
        }
