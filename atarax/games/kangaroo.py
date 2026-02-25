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

"""Kangaroo — JAX-native game implementation.

A mother kangaroo must climb four platform levels to rescue her joey,
punching monkeys that throw apples along the way.

Action space (6 actions):
    0 — NOOP
    1 — FIRE  (punch)
    2 — UP    (jump / climb ladder)
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Monkey punched — +200
    Apple dodged   — +100
    Joey rescued   — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Platform layout (4 platforms)
# ---------------------------------------------------------------------------
_N_PLATFORMS: int = 4
_N_MONKEYS: int = 3
_N_APPLES: int = 6

# Platform y-positions (bottom edge of each floor)
_PLATFORM_Y = jnp.array([185, 140, 95, 50], dtype=jnp.int32)
_PLATFORM_LEFT = jnp.array([10, 10, 10, 10], dtype=jnp.int32)
_PLATFORM_RIGHT = jnp.array([150, 150, 150, 150], dtype=jnp.int32)

# Ladder positions (x, connects floor i to floor i+1)
_LADDER_X = jnp.array([80, 50, 110], dtype=jnp.int32)  # [3 ladders]
_LADDER_FROM = jnp.array([0, 1, 2], dtype=jnp.int32)  # connects floor

_PLAYER_SPEED: float = 2.0
_JUMP_VELOCITY: float = -4.0
_GRAVITY: float = 0.3
_PUNCH_RANGE: float = 20.0
_APPLE_SPEED: float = 2.5

_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([30, 30, 60], dtype=jnp.uint8)
_COLOR_PLATFORM = jnp.array([140, 90, 40], dtype=jnp.uint8)
_COLOR_LADDER = jnp.array([120, 80, 40], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 220, 80], dtype=jnp.uint8)
_COLOR_MONKEY = jnp.array([150, 100, 40], dtype=jnp.uint8)
_COLOR_APPLE = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_JOEY = jnp.array([200, 180, 100], dtype=jnp.uint8)


@chex.dataclass
class KangarooState(AtariState):
    """
    Complete Kangaroo game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    player_vy : jax.Array
        float32 — Vertical velocity (negative = upward).
    player_floor : jax.Array
        int32 — Current floor (0=ground, 3=top).
    monkey_x : jax.Array
        float32[3] — Monkey x positions.
    monkey_floor : jax.Array
        int32[3] — Monkey floors.
    monkey_dir : jax.Array
        int32[3] — Monkey directions.
    apple_x : jax.Array
        float32[6] — Apple x positions.
    apple_y : jax.Array
        float32[6] — Apple y positions.
    apple_active : jax.Array
        bool[6] — Apple in-flight.
    punch_active : jax.Array
        bool — Punch animation active.
    punch_timer : jax.Array
        int32 — Punch duration.
    spawn_timer : jax.Array
        int32 — Frames until next apple throw.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    player_floor: jax.Array
    monkey_x: jax.Array
    monkey_floor: jax.Array
    monkey_dir: jax.Array
    apple_x: jax.Array
    apple_y: jax.Array
    apple_active: jax.Array
    punch_active: jax.Array
    punch_timer: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class Kangaroo(AtariEnv):
    """
    Kangaroo implemented as a pure JAX function suite.

    Climb to rescue the joey while punching monkeys.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=50_000)

    def _reset(self, key: jax.Array) -> KangarooState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : KangarooState
            Player at ground level, 3 monkeys, 3 lives.
        """
        return KangarooState(
            player_x=jnp.float32(20.0),
            player_y=jnp.float32(float(_PLATFORM_Y[0]) - 10.0),
            player_vy=jnp.float32(0.0),
            player_floor=jnp.int32(0),
            monkey_x=jnp.array([120.0, 80.0, 120.0], dtype=jnp.float32),
            monkey_floor=jnp.array([1, 2, 3], dtype=jnp.int32),
            monkey_dir=jnp.array([-1, 1, -1], dtype=jnp.int32),
            apple_x=jnp.full(_N_APPLES, -10.0, dtype=jnp.float32),
            apple_y=jnp.full(_N_APPLES, -10.0, dtype=jnp.float32),
            apple_active=jnp.zeros(_N_APPLES, dtype=jnp.bool_),
            punch_active=jnp.bool_(False),
            punch_timer=jnp.int32(0),
            spawn_timer=jnp.int32(60),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: KangarooState, action: jax.Array) -> KangarooState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : KangarooState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : KangarooState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Horizontal movement
        move_r = (action == jnp.int32(3)) | (action == jnp.int32(2))
        move_l = action == jnp.int32(5)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l, -_PLAYER_SPEED, 0.0),
            10.0,
            150.0,
        )

        # Jump
        on_platform = state.player_vy == 0.0
        do_jump = (action == jnp.int32(2)) & on_platform
        new_vy = jnp.where(
            do_jump, jnp.float32(_JUMP_VELOCITY), state.player_vy + _GRAVITY
        )

        new_py = state.player_y + new_vy

        # Floor collision
        floor_y = jnp.float32(_PLATFORM_Y[state.player_floor]) - 10.0
        landed = new_py >= floor_y
        new_py = jnp.where(landed, floor_y, new_py)
        new_vy = jnp.where(landed, jnp.float32(0.0), new_vy)

        # Climbing: if near ladder and pressing UP, advance floor
        near_ladder = jnp.any(jnp.abs(jnp.float32(_LADDER_X) - new_px) < 8.0)
        on_same_floor = _LADDER_FROM == state.player_floor
        can_climb = near_ladder & jnp.any(on_same_floor) & (action == jnp.int32(2))
        new_floor = jnp.where(
            can_climb & (state.player_floor < 3),
            state.player_floor + jnp.int32(1),
            state.player_floor,
        )
        new_py = jnp.where(
            can_climb & (state.player_floor < 3),
            jnp.float32(_PLATFORM_Y[new_floor]) - 10.0,
            new_py,
        )
        new_vy = jnp.where(can_climb, jnp.float32(0.0), new_vy)

        # Joey rescued (reach top floor)
        rescued = new_floor >= jnp.int32(3)
        step_reward = step_reward + jnp.where(
            rescued, jnp.float32(1000.0), jnp.float32(0.0)
        )
        # Reset to ground on rescue
        new_floor = jnp.where(rescued, jnp.int32(0), new_floor)
        new_py = jnp.where(rescued, jnp.float32(float(_PLATFORM_Y[0]) - 10.0), new_py)
        new_px = jnp.where(rescued, jnp.float32(20.0), new_px)

        # Punch
        do_punch = action == jnp.int32(1)
        new_punch_active = jnp.where(do_punch, jnp.bool_(True), state.punch_active)
        new_punch_timer = jnp.where(
            do_punch, jnp.int32(8), state.punch_timer - jnp.int32(1)
        )
        new_punch_timer = jnp.maximum(new_punch_timer, jnp.int32(0))
        new_punch_active = new_punch_active & (new_punch_timer > jnp.int32(0))

        # Punch hits monkey
        punch_x = new_px + 15.0  # punch slightly ahead
        monkey_punched = (
            new_punch_active
            & (jnp.abs(state.monkey_x - punch_x) < _PUNCH_RANGE)
            & (state.monkey_floor == new_floor)
        )
        n_punched = jnp.sum(monkey_punched).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_punched * 200)
        # Respawn punched monkeys at far side
        new_monkey_x = jnp.where(monkey_punched, jnp.float32(150.0), state.monkey_x)

        # Monkey movement
        new_monkey_x = new_monkey_x + state.monkey_dir.astype(jnp.float32) * 1.2
        at_wall = (new_monkey_x < 10.0) | (new_monkey_x > 150.0)
        new_monkey_dir = jnp.where(at_wall, -state.monkey_dir, state.monkey_dir)
        new_monkey_x = jnp.clip(new_monkey_x, 10.0, 150.0)

        # Apples
        new_apple_y = state.apple_y + jnp.where(state.apple_active, _APPLE_SPEED, 0.0)
        apple_off = state.apple_active & (new_apple_y > 200.0)
        step_reward = step_reward + jnp.float32(
            jnp.sum(apple_off).astype(jnp.int32) * 100
        )
        new_apple_active = state.apple_active & ~apple_off

        # Spawn apple from random monkey
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(60), new_spawn_timer)
        thrower = jax.random.randint(sk, (), 0, _N_MONKEYS)
        free_apple = jnp.argmin(new_apple_active.astype(jnp.int32))
        throw_y = jnp.float32(_PLATFORM_Y[state.monkey_floor[thrower]]) - 10.0
        new_apple_x = jnp.where(
            do_spawn,
            state.apple_x.at[free_apple].set(state.monkey_x[thrower]),
            state.apple_x,
        )
        new_apple_y = jnp.where(
            do_spawn, new_apple_y.at[free_apple].set(throw_y), new_apple_y
        )
        new_apple_active = jnp.where(
            do_spawn, new_apple_active.at[free_apple].set(True), new_apple_active
        )

        # Apple hits player
        apple_hits = (
            new_apple_active
            & (jnp.abs(new_apple_x - new_px) < 10.0)
            & (jnp.abs(new_apple_y - new_py) < 10.0)
        )
        hit = jnp.any(apple_hits)
        new_apple_active = new_apple_active & ~apple_hits

        new_lives = state.lives - jnp.where(hit, jnp.int32(1), jnp.int32(0))
        new_py = jnp.where(hit, jnp.float32(float(_PLATFORM_Y[0]) - 10.0), new_py)
        new_px = jnp.where(hit, jnp.float32(20.0), new_px)
        new_floor = jnp.where(hit, jnp.int32(0), new_floor)

        done = new_lives <= jnp.int32(0)

        return KangarooState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            player_floor=new_floor,
            monkey_x=new_monkey_x,
            monkey_floor=state.monkey_floor,
            monkey_dir=new_monkey_dir,
            apple_x=new_apple_x,
            apple_y=new_apple_y,
            apple_active=new_apple_active,
            punch_active=new_punch_active,
            punch_timer=new_punch_timer,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: KangarooState, action: jax.Array) -> KangarooState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : KangarooState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : KangarooState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: KangarooState) -> KangarooState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: KangarooState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : KangarooState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Platforms
        def draw_platform(frm, i):
            py = _PLATFORM_Y[i]
            mask = (
                (_ROW_IDX >= py - 4)
                & (_ROW_IDX <= py)
                & (_COL_IDX >= 10)
                & (_COL_IDX <= 150)
            )
            return jnp.where(mask[:, :, None], _COLOR_PLATFORM, frm), None

        frame, _ = jax.lax.scan(draw_platform, frame, jnp.arange(_N_PLATFORMS))

        # Ladders
        def draw_ladder(frm, i):
            lx = _LADDER_X[i]
            f = _LADDER_FROM[i]
            y0 = _PLATFORM_Y[f + 1]
            y1 = _PLATFORM_Y[f]
            mask = (
                (_ROW_IDX >= y0)
                & (_ROW_IDX <= y1)
                & (_COL_IDX >= lx - 3)
                & (_COL_IDX <= lx + 3)
            )
            return jnp.where(mask[:, :, None], _COLOR_LADDER, frm), None

        frame, _ = jax.lax.scan(draw_ladder, frame, jnp.arange(3))

        # Joey at top
        joey_mask = (
            (_ROW_IDX >= _PLATFORM_Y[3] - 20)
            & (_ROW_IDX <= _PLATFORM_Y[3] - 5)
            & (_COL_IDX >= 130)
            & (_COL_IDX <= 148)
        )
        frame = jnp.where(joey_mask[:, :, None], _COLOR_JOEY, frame)

        # Monkeys
        def draw_monkey(frm, i):
            mx = state.monkey_x[i].astype(jnp.int32)
            my = _PLATFORM_Y[state.monkey_floor[i]] - 14
            mask = (
                (_ROW_IDX >= my - 6)
                & (_ROW_IDX <= my + 6)
                & (_COL_IDX >= mx - 6)
                & (_COL_IDX <= mx + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_MONKEY, frm), None

        frame, _ = jax.lax.scan(draw_monkey, frame, jnp.arange(_N_MONKEYS))

        # Apples
        def draw_apple(frm, i):
            ax = state.apple_x[i].astype(jnp.int32)
            ay = state.apple_y[i].astype(jnp.int32)
            mask = (
                state.apple_active[i]
                & (_ROW_IDX >= ay - 4)
                & (_ROW_IDX <= ay + 4)
                & (_COL_IDX >= ax - 4)
                & (_COL_IDX <= ax + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_APPLE, frm), None

        frame, _ = jax.lax.scan(draw_apple, frame, jnp.arange(_N_APPLES))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 8)
            & (_ROW_IDX <= py + 8)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX <= px + 5)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Kangaroo action indices.
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
