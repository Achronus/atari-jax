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

"""Up 'n Down — JAX-native game implementation.

Drive a jeep on a winding mountain road, collecting flags and running over
or jumping over enemy vehicles.  Collect all flags in a level to advance.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (unused / honk)
    2 — UP   (jump)
    3 — RIGHT (accelerate)
    4 — DOWN  (unused)
    5 — LEFT  (brake)

Scoring:
    Flag collected — +100
    Enemy squashed — +200
    Level complete — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_FLAGS: int = 8
_N_ENEMIES: int = 4
_ROAD_Y: int = 140

_PLAYER_SPEED_MIN: float = 1.0
_PLAYER_SPEED_MAX: float = 4.0
_JUMP_VY: float = -5.0
_GRAVITY: float = 0.4
_ENEMY_SPEED: float = 2.5

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([100, 160, 80], dtype=jnp.uint8)
_COLOR_ROAD = jnp.array([180, 160, 120], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([200, 100, 50], dtype=jnp.uint8)
_COLOR_FLAG = jnp.array([255, 50, 50], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([80, 80, 200], dtype=jnp.uint8)


@chex.dataclass
class UpNDownState(AtariState):
    """
    Complete Up 'n Down game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x (screen, road scrolls).
    player_y : jax.Array
        float32 — Player y.
    player_vy : jax.Array
        float32 — Vertical velocity.
    speed : jax.Array
        float32 — Forward speed (road scroll rate).
    scroll_x : jax.Array
        float32 — World scroll.
    flag_x : jax.Array
        float32[8] — Flag world x.
    flag_active : jax.Array
        bool[8] — Flag not collected.
    enemy_x : jax.Array
        float32[4] — Enemy world x.
    enemy_active : jax.Array
        bool[4] — Enemy alive.
    level : jax.Array
        int32 — Current level.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    speed: jax.Array
    scroll_x: jax.Array
    flag_x: jax.Array
    flag_active: jax.Array
    enemy_x: jax.Array
    enemy_active: jax.Array
    level: jax.Array
    key: jax.Array


_GROUND_Y: float = float(_ROAD_Y - 12)


class UpNDown(AtariEnv):
    """
    Up 'n Down implemented as a pure JAX function suite.

    Collect all flags on winding roads.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> UpNDownState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : UpNDownState
            Player at left, flags and enemies spread on road.
        """
        return UpNDownState(
            player_x=jnp.float32(30.0),
            player_y=jnp.float32(_GROUND_Y),
            player_vy=jnp.float32(0.0),
            speed=jnp.float32(2.0),
            scroll_x=jnp.float32(0.0),
            flag_x=jnp.linspace(200.0, 1400.0, _N_FLAGS, dtype=jnp.float32),
            flag_active=jnp.ones(_N_FLAGS, dtype=jnp.bool_),
            enemy_x=jnp.array([300.0, 600.0, 900.0, 1200.0], dtype=jnp.float32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            level=jnp.int32(1),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: UpNDownState, action: jax.Array) -> UpNDownState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : UpNDownState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : UpNDownState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Speed
        on_ground = state.player_y >= jnp.float32(_GROUND_Y - 1.0)
        accel = action == jnp.int32(3)
        brake = action == jnp.int32(5)
        new_speed = jnp.clip(
            state.speed
            + jnp.where(
                accel,
                jnp.float32(0.2),
                jnp.where(brake, jnp.float32(-0.2), jnp.float32(0.0)),
            ),
            _PLAYER_SPEED_MIN,
            _PLAYER_SPEED_MAX,
        )

        # Jump
        jump = (action == jnp.int32(2)) & on_ground
        new_vy = jnp.where(jump, jnp.float32(_JUMP_VY), state.player_vy + _GRAVITY)
        new_py = jnp.minimum(state.player_y + new_vy, jnp.float32(_GROUND_Y))
        new_vy = jnp.where(new_py >= jnp.float32(_GROUND_Y), jnp.float32(0.0), new_vy)

        # Scroll world
        new_scroll_x = state.scroll_x + new_speed

        # Flag collection
        screen_flag_x = state.flag_x - new_scroll_x
        player_screen_x = state.player_x
        flag_hit = (
            state.flag_active
            & (jnp.abs(screen_flag_x - player_screen_x) < jnp.float32(12.0))
            & on_ground
        )
        step_reward = step_reward + jnp.sum(flag_hit).astype(jnp.float32) * jnp.float32(
            100.0
        )
        new_flag_active = state.flag_active & ~flag_hit

        # Enemy collision
        screen_enemy_x = state.enemy_x - new_scroll_x
        airborne = ~on_ground
        # Jump over enemy: player is in air → no collision
        enemy_hit = (
            state.enemy_active
            & (jnp.abs(screen_enemy_x - player_screen_x) < jnp.float32(12.0))
            & on_ground
        )
        # Squash enemy (land on top while jumping)
        squash = (
            state.enemy_active
            & (jnp.abs(screen_enemy_x - player_screen_x) < jnp.float32(12.0))
            & airborne
            & (state.player_vy > jnp.float32(0.0))
        )
        step_reward = step_reward + jnp.sum(squash).astype(jnp.float32) * jnp.float32(
            200.0
        )
        new_enemy_active = state.enemy_active & ~squash
        hit_by_enemy = jnp.any(enemy_hit)

        # Level clear
        level_clear = ~jnp.any(new_flag_active)
        step_reward = step_reward + jnp.where(
            level_clear, jnp.float32(1000.0), jnp.float32(0.0)
        )
        new_level = state.level + jnp.where(level_clear, jnp.int32(1), jnp.int32(0))
        new_flag_active2 = jnp.where(
            level_clear, jnp.ones(_N_FLAGS, dtype=jnp.bool_), new_flag_active
        )
        new_enemy_active2 = jnp.where(
            level_clear, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        new_scroll_x2 = jnp.where(level_clear, jnp.float32(0.0), new_scroll_x)

        life_lost = hit_by_enemy
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return UpNDownState(
            player_x=state.player_x,
            player_y=new_py,
            player_vy=new_vy,
            speed=new_speed,
            scroll_x=new_scroll_x2,
            flag_x=state.flag_x,
            flag_active=new_flag_active2,
            enemy_x=state.enemy_x,
            enemy_active=new_enemy_active2,
            level=new_level,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: UpNDownState, action: jax.Array) -> UpNDownState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : UpNDownState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : UpNDownState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: UpNDownState) -> UpNDownState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: UpNDownState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : UpNDownState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Road
        road_mask = (_ROW_IDX >= _ROAD_Y - 20) & (_ROW_IDX < _ROAD_Y + 5)
        frame = jnp.where(road_mask[:, :, None], _COLOR_ROAD, frame)

        # Flags
        def draw_flag(frm, i):
            fx = (state.flag_x[i] - state.scroll_x).astype(jnp.int32)
            mask = (
                state.flag_active[i]
                & (_ROW_IDX >= _ROAD_Y - 20)
                & (_ROW_IDX < _ROAD_Y - 8)
                & (_COL_IDX >= fx - 4)
                & (_COL_IDX < fx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_FLAG, frm), None

        frame, _ = jax.lax.scan(draw_flag, frame, jnp.arange(_N_FLAGS))

        # Enemies
        def draw_enemy(frm, i):
            ex = (state.enemy_x[i] - state.scroll_x).astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= _ROAD_Y - 16)
                & (_ROW_IDX < _ROAD_Y)
                & (_COL_IDX >= ex - 6)
                & (_COL_IDX < ex + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py)
            & (_ROW_IDX < py + 12)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX < px + 5)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Up 'n Down action indices.
        """
        import pygame

        return {
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
