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

"""Zaxxon — JAX-native game implementation.

Isometric scrolling space fortress assault.  Fly your spaceship through a
fortress, navigating walls, shooting gun emplacements and jet fighters,
then fight the robot boss Zaxxon.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP   (gain altitude)
    3 — RIGHT
    4 — DOWN (lose altitude)
    5 — LEFT
    6 — FIRE+UP
    7 — FIRE+RIGHT
    8 — FIRE+LEFT

Scoring:
    Gun emplacement — +150
    Fuel tank       — +200
    Jet fighter     — +1000
    Zaxxon boss     — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMIES: int = 8
_PLAYER_Y: int = 105  # fixed horizontal position
_PLAYER_X: int = 60

_BULLET_SPEED: float = 5.0
_ENEMY_SCROLL: float = 2.0
_ALTITUDE_SPEED: float = 2.0
_ALTITUDE_MIN: float = 30.0
_ALTITUDE_MAX: float = 180.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 20, 60], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([50, 120, 50], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([200, 200, 100], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_SHADOW = jnp.array([30, 80, 30], dtype=jnp.uint8)
_COLOR_WALL = jnp.array([100, 80, 40], dtype=jnp.uint8)

_ENEMY_SCORES = jnp.array([150, 200, 1000, 1000, 150, 200, 1000, 1000], dtype=jnp.int32)


@chex.dataclass
class ZaxxonState(AtariState):
    """
    Complete Zaxxon game state — a JAX pytree.

    Parameters
    ----------
    altitude : jax.Array
        float32 — Player altitude (screen y; higher = lower altitude).
    scroll_x : jax.Array
        float32 — World scroll position.
    bullet_x : jax.Array
        float32 — Bullet x (scrolling world coord).
    bullet_y : jax.Array
        float32 — Bullet altitude.
    bullet_active : jax.Array
        bool — Bullet in flight.
    enemy_x : jax.Array
        float32[8] — Enemy x (world).
    enemy_y : jax.Array
        float32[8] — Enemy altitude.
    enemy_active : jax.Array
        bool[8] — Enemy alive.
    fuel : jax.Array
        int32 — Fuel remaining.
    wave : jax.Array
        int32 — Wave (fortress section).
    key : jax.Array
        uint32[2] — PRNG key.
    """

    altitude: jax.Array
    scroll_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    fuel: jax.Array
    wave: jax.Array
    key: jax.Array


class Zaxxon(AtariEnv):
    """
    Zaxxon implemented as a pure JAX function suite.

    Navigate fortress, shoot enemies.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> ZaxxonState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : ZaxxonState
            Player at mid-altitude, enemies scattered ahead.
        """
        enemy_xs = jnp.linspace(200.0, 1200.0, _N_ENEMIES, dtype=jnp.float32)
        enemy_ys = jnp.array(
            [80.0, 100.0, 120.0, 80.0, 100.0, 60.0, 140.0, 90.0], dtype=jnp.float32
        )
        return ZaxxonState(
            altitude=jnp.float32(105.0),
            scroll_x=jnp.float32(0.0),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(105.0),
            bullet_active=jnp.bool_(False),
            enemy_x=enemy_xs,
            enemy_y=enemy_ys,
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            fuel=jnp.int32(2000),
            wave=jnp.int32(1),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: ZaxxonState, action: jax.Array) -> ZaxxonState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : ZaxxonState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : ZaxxonState
            State after one emulated frame.
        """
        key, k_spawn = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Altitude control
        up = (action == 2) | (action == 6)
        dn = action == 4
        new_alt = jnp.clip(
            state.altitude
            + jnp.where(up, -_ALTITUDE_SPEED, jnp.where(dn, _ALTITUDE_SPEED, 0.0)),
            _ALTITUDE_MIN,
            _ALTITUDE_MAX,
        )

        # Scroll world forward
        new_scroll = state.scroll_x + _ENEMY_SCROLL
        new_fuel = state.fuel - jnp.int32(1)

        # Enemy screen x = enemy_world_x - scroll_x + PLAYER_X
        enemy_screen_x = state.enemy_x - new_scroll + jnp.float32(_PLAYER_X)

        # Fire bullet (moves forward in world)
        fire = ((action == 1) | (action >= 6)) & ~state.bullet_active
        new_bx = jnp.where(
            fire, state.scroll_x + jnp.float32(_PLAYER_X + 10), state.bullet_x
        )
        new_by = jnp.where(fire, new_alt, state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_bx = jnp.where(new_bactive, new_bx + _BULLET_SPEED, new_bx)
        bullet_screen_x = new_bx - new_scroll + jnp.float32(_PLAYER_X)
        new_bactive = new_bactive & (bullet_screen_x < jnp.float32(155.0))

        # Bullet hits enemy
        b_hits_e = (
            new_bactive
            & state.enemy_active
            & (jnp.abs(bullet_screen_x - enemy_screen_x) < jnp.float32(12.0))
            & (jnp.abs(new_by - state.enemy_y) < jnp.float32(12.0))
        )
        step_reward = step_reward + jnp.sum(
            jnp.where(b_hits_e, _ENEMY_SCORES, jnp.zeros(_N_ENEMIES, dtype=jnp.int32))
        ).astype(jnp.float32)
        new_enemy_active = state.enemy_active & ~b_hits_e
        new_bactive = new_bactive & ~jnp.any(b_hits_e)

        # Player collides with enemy
        player_hits_e = (
            new_enemy_active
            & (jnp.abs(enemy_screen_x - jnp.float32(_PLAYER_X)) < jnp.float32(10.0))
            & (jnp.abs(state.enemy_y - new_alt) < jnp.float32(10.0))
        )
        hit_by_enemy = jnp.any(player_hits_e)

        # Enemies scroll off-screen: respawn ahead
        respawn = new_enemy_active & (enemy_screen_x < jnp.float32(-20.0))
        new_ex = jnp.where(respawn, state.enemy_x + jnp.float32(1000.0), state.enemy_x)

        # Wave clear when all enemies passed
        wave_clear = ~jnp.any(new_enemy_active)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_enemy_active2 = jnp.where(
            wave_clear, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        new_ex2 = jnp.where(
            wave_clear,
            jnp.linspace(200.0, 1200.0, _N_ENEMIES, dtype=jnp.float32) + new_scroll,
            new_ex,
        )

        fuel_empty = new_fuel <= jnp.int32(0)
        life_lost = hit_by_enemy | fuel_empty
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_fuel = jnp.where(fuel_empty, jnp.int32(2000), new_fuel)
        done = new_lives <= jnp.int32(0)

        return ZaxxonState(
            altitude=new_alt,
            scroll_x=new_scroll,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            enemy_x=new_ex2,
            enemy_y=state.enemy_y,
            enemy_active=new_enemy_active2,
            fuel=new_fuel,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: ZaxxonState, action: jax.Array) -> ZaxxonState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : ZaxxonState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : ZaxxonState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: ZaxxonState) -> ZaxxonState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: ZaxxonState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : ZaxxonState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground (isometric horizon)
        ground_mask = _ROW_IDX >= 170
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Enemies
        def draw_enemy(frm, i):
            ex_screen = (
                state.enemy_x[i] - state.scroll_x + jnp.float32(_PLAYER_X)
            ).astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            # Shadow on ground
            shadow = (
                state.enemy_active[i]
                & (_ROW_IDX >= 165)
                & (_ROW_IDX < 170)
                & (_COL_IDX >= ex_screen - 6)
                & (_COL_IDX < ex_screen + 6)
            )
            body_mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 6)
                & (_ROW_IDX < ey + 6)
                & (_COL_IDX >= ex_screen - 6)
                & (_COL_IDX < ex_screen + 6)
            )
            frm = jnp.where(shadow[:, :, None], _COLOR_SHADOW, frm)
            frm = jnp.where(body_mask[:, :, None], _COLOR_ENEMY, frm)
            return frm, None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Bullet
        bx_screen = (state.bullet_x - state.scroll_x + jnp.float32(_PLAYER_X)).astype(
            jnp.int32
        )
        by = state.bullet_y.astype(jnp.int32)
        bm = (
            state.bullet_active
            & (_ROW_IDX >= by - 2)
            & (_ROW_IDX < by + 2)
            & (_COL_IDX >= bx_screen)
            & (_COL_IDX < bx_screen + 6)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        alt = state.altitude.astype(jnp.int32)
        # Player shadow
        pshadow = (
            (_ROW_IDX >= 165)
            & (_ROW_IDX < 170)
            & (_COL_IDX >= _PLAYER_X - 6)
            & (_COL_IDX < _PLAYER_X + 6)
        )
        frame = jnp.where(pshadow[:, :, None], _COLOR_SHADOW, frame)
        pm = (
            (_ROW_IDX >= alt - 6)
            & (_ROW_IDX < alt + 6)
            & (_COL_IDX >= _PLAYER_X - 6)
            & (_COL_IDX < _PLAYER_X + 6)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Zaxxon action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
