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

"""Solaris — JAX-native game implementation.

Defend the solar system from Zylon squadrons.  Navigate a star map to reach
sectors under attack; then engage in close-range combat.  Manage fuel and
protect friendly planets.

Action space (8 actions):
    0 — NOOP
    1 — FIRE
    2 — UP   (thrust forward)
    3 — RIGHT
    4 — DOWN (thrust back)
    5 — LEFT
    6 — WARP (hyperspace jump)
    7 — FIRE+UP

Scoring:
    Zylon fighter — +250
    Zylon base    — +500
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMIES: int = 6
_PLAYER_Y: int = 160
_BULLET_SPEED: float = 6.0
_ENEMY_SPEED: float = 1.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 20], dtype=jnp.uint8)
_COLOR_STAR = jnp.array([200, 200, 220], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 200, 255], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 80, 80], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 100, 0], dtype=jnp.uint8)
_COLOR_PLANET = jnp.array([80, 200, 80], dtype=jnp.uint8)


@chex.dataclass
class SolarisState(AtariState):
    """
    Complete Solaris game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player ship x.
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    enemy_x : jax.Array
        float32[6] — Enemy x.
    enemy_y : jax.Array
        float32[6] — Enemy y (scroll down).
    enemy_active : jax.Array
        bool[6] — Enemy alive.
    enemy_bx : jax.Array
        float32 — Enemy bullet x.
    enemy_by : jax.Array
        float32 — Enemy bullet y.
    enemy_bactive : jax.Array
        bool — Enemy bullet active.
    fuel : jax.Array
        int32 — Remaining fuel.
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    fire_timer : jax.Array
        int32 — Frames until enemy fires.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    enemy_bx: jax.Array
    enemy_by: jax.Array
    enemy_bactive: jax.Array
    fuel: jax.Array
    spawn_timer: jax.Array
    fire_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class Solaris(AtariEnv):
    """
    Solaris implemented as a pure JAX function suite.

    Defend solar systems from Zylon attacks.  Lives: 5.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> SolarisState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : SolarisState
            Player at centre, no enemies, full fuel.
        """
        return SolarisState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_active=jnp.bool_(False),
            enemy_x=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_y=jnp.linspace(20.0, 120.0, _N_ENEMIES, dtype=jnp.float32),
            enemy_active=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            enemy_bx=jnp.float32(0.0),
            enemy_by=jnp.float32(0.0),
            enemy_bactive=jnp.bool_(False),
            fuel=jnp.int32(2000),
            spawn_timer=jnp.int32(60),
            fire_timer=jnp.int32(90),
            wave=jnp.int32(1),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: SolarisState, action: jax.Array) -> SolarisState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : SolarisState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : SolarisState
            State after one emulated frame.
        """
        key, k_spawn, k_ex = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Player movement
        move_r = action == jnp.int32(3)
        move_l = action == jnp.int32(5)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(
                move_r,
                jnp.float32(2.0),
                jnp.where(move_l, jnp.float32(-2.0), jnp.float32(0.0)),
            ),
            jnp.float32(5.0),
            jnp.float32(147.0),
        )

        # Fuel decreases with thrust
        thrust = (action == jnp.int32(2)) | (action == jnp.int32(7))
        new_fuel = (
            state.fuel - jnp.int32(1) - jnp.where(thrust, jnp.int32(1), jnp.int32(0))
        )

        # Fire
        fire = (
            (action == jnp.int32(1)) | (action == jnp.int32(7))
        ) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(float(_PLAYER_Y - 8)), state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(10.0))

        # Enemies scroll toward player
        new_ey = state.enemy_y + jnp.where(
            state.enemy_active, jnp.float32(_ENEMY_SPEED), jnp.float32(0.0)
        )
        new_enemy_active = state.enemy_active & (
            new_ey < jnp.float32(float(_PLAYER_Y - 5))
        )

        # Bullet hits enemy
        bullet_hit = (
            new_bactive
            & new_enemy_active
            & (jnp.abs(new_bx - state.enemy_x) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_ey) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(bullet_hit).astype(
            jnp.float32
        ) * jnp.float32(250.0)
        new_enemy_active = new_enemy_active & ~bullet_hit
        new_bactive = new_bactive & ~jnp.any(bullet_hit)

        # Enemy reaches player
        enemy_reaches = new_enemy_active & (new_ey >= jnp.float32(float(_PLAYER_Y - 5)))
        hit_by_enemy = jnp.any(enemy_reaches)

        # Spawn
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_spawn = new_spawn_timer <= jnp.int32(0)
        free_e = jnp.argmin(new_enemy_active.astype(jnp.int32))
        spawn_x = jax.random.uniform(k_ex, minval=10.0, maxval=150.0)
        new_enemy_active2 = jnp.where(
            can_spawn, new_enemy_active.at[free_e].set(True), new_enemy_active
        )
        new_ex = jnp.where(
            can_spawn, state.enemy_x.at[free_e].set(spawn_x), state.enemy_x
        )
        new_ey2 = jnp.where(can_spawn, new_ey.at[free_e].set(jnp.float32(10.0)), new_ey)
        new_spawn_timer = jnp.where(can_spawn, jnp.int32(50), new_spawn_timer)

        # Enemy fires
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & jnp.any(new_enemy_active2)
        rand_e = jax.random.uniform(k_spawn, (_N_ENEMIES,))
        alive_e = jnp.where(new_enemy_active2, rand_e, jnp.float32(-1.0))
        shooter = jnp.argmax(alive_e)
        new_ebx = jnp.where(
            can_fire, new_ex[shooter] + jnp.float32(5.0), state.enemy_bx
        )
        new_eby = jnp.where(
            can_fire, new_ey2[shooter] + jnp.float32(5.0), state.enemy_by
        )
        new_ebactive = jnp.where(can_fire, jnp.bool_(True), state.enemy_bactive)
        new_fire_timer = jnp.where(can_fire, jnp.int32(60), new_fire_timer)
        new_eby = jnp.where(new_ebactive, new_eby + jnp.float32(3.0), new_eby)
        new_ebactive = new_ebactive & (new_eby < jnp.float32(float(_PLAYER_Y + 10)))

        enemy_bullet_hits = (
            new_ebactive
            & (jnp.abs(new_ebx - new_px) < jnp.float32(8.0))
            & (new_eby >= jnp.float32(float(_PLAYER_Y - 6)))
        )

        fuel_empty = new_fuel <= jnp.int32(0)
        life_lost = hit_by_enemy | enemy_bullet_hits | fuel_empty
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_fuel = jnp.where(fuel_empty, jnp.int32(2000), new_fuel)
        done = new_lives <= jnp.int32(0)

        return SolarisState(
            player_x=new_px,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey2,
            enemy_active=new_enemy_active2,
            enemy_bx=new_ebx,
            enemy_by=new_eby,
            enemy_bactive=new_ebactive,
            fuel=new_fuel,
            spawn_timer=new_spawn_timer,
            fire_timer=new_fire_timer,
            wave=state.wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: SolarisState, action: jax.Array) -> SolarisState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : SolarisState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : SolarisState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: SolarisState) -> SolarisState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: SolarisState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : SolarisState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey)
                & (_ROW_IDX < ey + 10)
                & (_COL_IDX >= ex)
                & (_COL_IDX < ex + 10)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Enemy bullet
        ebm = (
            state.enemy_bactive
            & (_ROW_IDX >= state.enemy_by.astype(jnp.int32))
            & (_ROW_IDX < state.enemy_by.astype(jnp.int32) + 5)
            & (_COL_IDX >= state.enemy_bx.astype(jnp.int32))
            & (_COL_IDX < state.enemy_bx.astype(jnp.int32) + 2)
        )
        frame = jnp.where(ebm[:, :, None], _COLOR_ENEMY_BULLET, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32))
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 6)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32))
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= _PLAYER_Y - 6)
            & (_ROW_IDX < _PLAYER_Y + 6)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Solaris action indices.
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
