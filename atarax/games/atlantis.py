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

"""Atlantis — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    City sections : 6 sections at y = 175
    Left cannon   : x = 14, y = 170
    Centre cannon : x = 80, y = 170
    Right cannon  : x = 146, y = 170
    Aliens        : drift horizontally across the screen

Action space (4 actions — ALE minimal set):
    0  NOOP
    1  FIRE CENTRE
    2  FIRE RIGHT
    3  FIRE LEFT
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Cannon positions
_CANNON_LEFT_X: int = 14
_CANNON_CENTRE_X: int = 80
_CANNON_RIGHT_X: int = 146
_CANNON_Y: int = 170
_CANNON_W: int = 8
_CANNON_H: int = 6

# Bullets
_BULLET_W: int = 2
_BULLET_H: int = 4
_BULLET_SPEED_Y: float = 5.0  # vertical speed
_BULLET_SPEED_X: float = 1.5  # diagonal drift (left/right cannons)

# City sections
_CITY_Y: int = 175
_CITY_W: int = 16
_CITY_H: int = 10
_N_CITY: int = 6
_CITY_XS = jnp.array([15.0, 35.0, 55.0, 85.0, 105.0, 125.0], dtype=jnp.float32)

# Aliens
_N_ALIENS: int = 6
_ALIEN_W: int = 12
_ALIEN_H: int = 8
_ALIEN_SPEED: float = 0.8
_ALIEN_DESCENT: float = 0.05  # px/frame; scales with level
_ALIEN_POINTS: int = 250
_SCREEN_LEFT: float = 0.0
_SCREEN_RIGHT: float = 148.0  # 160 - ALIEN_W

# Initial alien positions (spread across top portion of screen)
_ALIEN_INIT_X = jnp.array([10.0, 40.0, 70.0, 90.0, 120.0, 140.0], dtype=jnp.float32)
_ALIEN_INIT_Y = jnp.array([40.0, 60.0, 80.0, 50.0, 70.0, 90.0], dtype=jnp.float32)
_ALIEN_INIT_DX = jnp.array([0.8, -0.8, 0.8, -0.8, 0.8, -0.8], dtype=jnp.float32)

_FRAME_SKIP: int = 4

# Render
_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_BG_COLOR = jnp.array([0, 0, 30], dtype=jnp.uint8)
_CITY_COLOR = jnp.array([80, 120, 200], dtype=jnp.uint8)
_CANNON_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 100], dtype=jnp.uint8)
_ALIEN_COLOR = jnp.array([200, 60, 200], dtype=jnp.uint8)


@chex.dataclass
class AtlantisState(AtariState):
    """
    Complete Atlantis game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `lives` = sum(city_alive).

    Parameters
    ----------
    alien_x : jax.Array
        float32[6] — Alien x positions.
    alien_y : jax.Array
        float32[6] — Alien y positions.
    alien_dx : jax.Array
        float32[6] — Alien horizontal velocities.
    alien_alive : jax.Array
        bool[6] — Alive aliens.
    bullet_left_x : jax.Array
        float32 — Left cannon bullet x.
    bullet_left_y : jax.Array
        float32 — Left cannon bullet y.
    bullet_left_active : jax.Array
        bool — Left bullet in flight.
    bullet_centre_x : jax.Array
        float32 — Centre cannon bullet x.
    bullet_centre_y : jax.Array
        float32 — Centre cannon bullet y.
    bullet_centre_active : jax.Array
        bool — Centre bullet in flight.
    bullet_right_x : jax.Array
        float32 — Right cannon bullet x.
    bullet_right_y : jax.Array
        float32 — Right cannon bullet y.
    bullet_right_active : jax.Array
        bool — Right bullet in flight.
    city_alive : jax.Array
        bool[6] — Surviving city sections.
    """

    alien_x: chex.Array
    alien_y: chex.Array
    alien_dx: chex.Array
    alien_alive: chex.Array
    bullet_left_x: chex.Array
    bullet_left_y: chex.Array
    bullet_left_active: chex.Array
    bullet_centre_x: chex.Array
    bullet_centre_y: chex.Array
    bullet_centre_active: chex.Array
    bullet_right_x: chex.Array
    bullet_right_y: chex.Array
    bullet_right_active: chex.Array
    city_alive: chex.Array


class Atlantis(AtaraxGame):
    """
    Atlantis implemented as a pure-JAX function suite.

    Defend six city sections using three cannon emplacements.  Alien ships
    drift across the screen; shoot them before they reach the city.
    """

    num_actions: int = 4

    def _reset(self, key: chex.PRNGKey) -> AtlantisState:
        """Return the canonical initial game state."""
        return AtlantisState(
            alien_x=_ALIEN_INIT_X.copy(),
            alien_y=_ALIEN_INIT_Y.copy(),
            alien_dx=_ALIEN_INIT_DX.copy(),
            alien_alive=jnp.ones(_N_ALIENS, dtype=jnp.bool_),
            bullet_left_x=jnp.float32(0.0),
            bullet_left_y=jnp.float32(0.0),
            bullet_left_active=jnp.bool_(False),
            bullet_centre_x=jnp.float32(0.0),
            bullet_centre_y=jnp.float32(0.0),
            bullet_centre_active=jnp.bool_(False),
            bullet_right_x=jnp.float32(0.0),
            bullet_right_y=jnp.float32(0.0),
            bullet_right_active=jnp.bool_(False),
            city_alive=jnp.ones(_N_CITY, dtype=jnp.bool_),
            lives=jnp.int32(_N_CITY),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: AtlantisState, action: jax.Array) -> AtlantisState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : AtlantisState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–3).

        Returns
        -------
        new_state : AtlantisState
            State after one emulated frame.
        """
        key, _ = jax.random.split(state.key)

        # --- Fire actions ---
        fire_centre = (action == 1) & ~state.bullet_centre_active
        fire_right = (action == 2) & ~state.bullet_right_active
        fire_left = (action == 3) & ~state.bullet_left_active

        # Spawn bullets at cannon positions
        bcx = jnp.where(
            fire_centre, jnp.float32(_CANNON_CENTRE_X), state.bullet_centre_x
        )
        bcy = jnp.where(
            fire_centre, jnp.float32(_CANNON_Y - _BULLET_H), state.bullet_centre_y
        )
        centre_active = state.bullet_centre_active | fire_centre

        brx = jnp.where(fire_right, jnp.float32(_CANNON_RIGHT_X), state.bullet_right_x)
        bry = jnp.where(
            fire_right, jnp.float32(_CANNON_Y - _BULLET_H), state.bullet_right_y
        )
        right_active = state.bullet_right_active | fire_right

        blx = jnp.where(fire_left, jnp.float32(_CANNON_LEFT_X), state.bullet_left_x)
        bly = jnp.where(
            fire_left, jnp.float32(_CANNON_Y - _BULLET_H), state.bullet_left_y
        )
        left_active = state.bullet_left_active | fire_left

        # --- Move bullets ---
        # Centre: straight up
        bcy = jnp.where(centre_active, bcy - jnp.float32(_BULLET_SPEED_Y), bcy)
        # Right: diagonally right and up
        brx = jnp.where(right_active, brx + jnp.float32(_BULLET_SPEED_X), brx)
        bry = jnp.where(right_active, bry - jnp.float32(_BULLET_SPEED_Y), bry)
        # Left: diagonally left and up
        blx = jnp.where(left_active, blx - jnp.float32(_BULLET_SPEED_X), blx)
        bly = jnp.where(left_active, bly - jnp.float32(_BULLET_SPEED_Y), bly)

        # Deactivate if out of screen
        centre_active = centre_active & (bcy > 0.0) & (bcx >= 0.0) & (bcx < 160.0)
        right_active = right_active & (bry > 0.0) & (brx < 160.0)
        left_active = left_active & (bly > 0.0) & (blx >= 0.0)

        # --- Move aliens ---
        new_ax = state.alien_x + state.alien_dx
        hit_left = new_ax < jnp.float32(_SCREEN_LEFT)
        hit_right = new_ax > jnp.float32(_SCREEN_RIGHT)
        new_adx = jnp.where(hit_left | hit_right, -state.alien_dx, state.alien_dx)
        new_ax = jnp.clip(new_ax, _SCREEN_LEFT, _SCREEN_RIGHT)

        # Aliens descend toward city; speed scales with current level (wave)
        alien_descent = jnp.float32(_ALIEN_DESCENT) * (
            jnp.float32(1.0) + jnp.float32(0.15) * state.level.astype(jnp.float32)
        )
        new_ay = state.alien_y + alien_descent

        # --- Bullet vs alien collisions ---
        def _check_bullet_alien(bx, by, b_active, aliens_alive):
            hit = (
                b_active
                & aliens_alive
                & (bx + jnp.float32(_BULLET_W) > new_ax)
                & (bx < new_ax + jnp.float32(_ALIEN_W))
                & (by + jnp.float32(_BULLET_H) > state.alien_y)
                & (by < state.alien_y + jnp.float32(_ALIEN_H))
            )
            return hit

        centre_hits = _check_bullet_alien(bcx, bcy, centre_active, state.alien_alive)
        right_hits = _check_bullet_alien(brx, bry, right_active, state.alien_alive)
        left_hits = _check_bullet_alien(blx, bly, left_active, state.alien_alive)

        all_hits = centre_hits | right_hits | left_hits
        new_alien_alive = state.alien_alive & ~all_hits
        kills = jnp.sum(all_hits.astype(jnp.int32))
        step_reward = jnp.float32(kills * _ALIEN_POINTS)

        # Deactivate bullet if it hit something
        centre_active = centre_active & ~jnp.any(centre_hits)
        right_active = right_active & ~jnp.any(right_hits)
        left_active = left_active & ~jnp.any(left_hits)

        # --- Alien reaches city ---
        alien_at_city = new_alien_alive & (
            new_ay + jnp.float32(_ALIEN_H) >= jnp.float32(_CITY_Y)
        )

        # For each city section, check if any alien overlaps it horizontally
        def _city_hit(city_idx):
            cx = _CITY_XS[city_idx]
            return jnp.any(
                alien_at_city
                & (new_ax + jnp.float32(_ALIEN_W) > cx)
                & (new_ax < cx + jnp.float32(_CITY_W))
            )

        city_hit_mask = jnp.array([_city_hit(i) for i in range(_N_CITY)])
        new_city_alive = state.city_alive & ~city_hit_mask

        # Reset aliens that reached city back to top of screen
        new_ay = jnp.where(alien_at_city, jnp.float32(20.0), new_ay)

        # --- Wave clear: all aliens killed → advance level, reset formation ---
        all_killed = ~jnp.any(new_alien_alive)
        new_level = state.level + jnp.where(all_killed, jnp.int32(1), jnp.int32(0))
        wave_speed_scale = jnp.float32(1.0) + jnp.float32(0.15) * new_level.astype(
            jnp.float32
        )
        new_ax = jnp.where(all_killed, _ALIEN_INIT_X.copy(), new_ax)
        new_ay = jnp.where(all_killed, _ALIEN_INIT_Y.copy(), new_ay)
        new_adx = jnp.where(
            all_killed, _ALIEN_INIT_DX.copy() * wave_speed_scale, new_adx
        )
        new_alien_alive = jnp.where(
            all_killed, jnp.ones(_N_ALIENS, dtype=jnp.bool_), new_alien_alive
        )

        new_lives = jnp.sum(new_city_alive.astype(jnp.int32))
        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            alien_x=new_ax,
            alien_y=new_ay,
            alien_dx=new_adx,
            alien_alive=new_alien_alive,
            bullet_left_x=blx,
            bullet_left_y=bly,
            bullet_left_active=left_active,
            bullet_centre_x=bcx,
            bullet_centre_y=bcy,
            bullet_centre_active=centre_active,
            bullet_right_x=brx,
            bullet_right_y=bry,
            bullet_right_active=right_active,
            city_alive=new_city_alive,
            lives=new_lives,
            level=new_level,
            score=state.score + jnp.int32(kills * _ALIEN_POINTS),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: AtlantisState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> AtlantisState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: AtlantisState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
        frame = frame.at[:, :, 2].set(jnp.uint8(30))

        # --- City sections ---
        for i in range(_N_CITY):
            cx = jnp.int32(_CITY_XS[i])
            city_mask = (
                state.city_alive[i]
                & (_ROW_IDX >= _CITY_Y)
                & (_ROW_IDX < _CITY_Y + _CITY_H)
                & (_COL_IDX >= cx)
                & (_COL_IDX < cx + _CITY_W)
            )
            frame = jnp.where(city_mask[:, :, None], _CITY_COLOR[None, None, :], frame)

        # --- Cannons ---
        for cx in [_CANNON_LEFT_X, _CANNON_CENTRE_X, _CANNON_RIGHT_X]:
            cannon_mask = (
                (_ROW_IDX >= _CANNON_Y)
                & (_ROW_IDX < _CANNON_Y + _CANNON_H)
                & (_COL_IDX >= cx)
                & (_COL_IDX < cx + _CANNON_W)
            )
            frame = jnp.where(
                cannon_mask[:, :, None], _CANNON_COLOR[None, None, :], frame
            )

        # --- Bullets ---
        for bx, by, ba in [
            (
                jnp.int32(state.bullet_left_x),
                jnp.int32(state.bullet_left_y),
                state.bullet_left_active,
            ),
            (
                jnp.int32(state.bullet_centre_x),
                jnp.int32(state.bullet_centre_y),
                state.bullet_centre_active,
            ),
            (
                jnp.int32(state.bullet_right_x),
                jnp.int32(state.bullet_right_y),
                state.bullet_right_active,
            ),
        ]:
            b_mask = (
                ba
                & (_ROW_IDX >= by)
                & (_ROW_IDX < by + _BULLET_H)
                & (_COL_IDX >= bx)
                & (_COL_IDX < bx + _BULLET_W)
            )
            frame = jnp.where(b_mask[:, :, None], _BULLET_COLOR[None, None, :], frame)

        # --- Aliens ---
        for i in range(_N_ALIENS):
            ax = jnp.int32(state.alien_x[i])
            ay = jnp.int32(state.alien_y[i])
            a_mask = (
                state.alien_alive[i]
                & (_ROW_IDX >= ay)
                & (_ROW_IDX < ay + _ALIEN_H)
                & (_COL_IDX >= ax)
                & (_COL_IDX < ax + _ALIEN_W)
            )
            frame = jnp.where(a_mask[:, :, None], _ALIEN_COLOR[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,
                pygame.K_RIGHT: 2,
                pygame.K_d: 2,
                pygame.K_LEFT: 3,
                pygame.K_a: 3,
            }
        except ImportError:
            return {}
