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

Defend the underwater city of Atlantis from waves of alien ships using
three gun emplacements.  Ships descend in passes; each one that reaches
the city destroys a section.  The episode ends when all six city sections
are destroyed.

Action space (4 actions):
    0 — NOOP
    1 — FIRE LEFT cannon  (diagonal left shot)
    2 — FIRE CENTRE cannon (vertical shot)
    3 — FIRE RIGHT cannon (diagonal right shot)

Scoring:
    Alien destroyed — +250
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_SECTIONS: int = 6  # city sections (= lives)
_N_ALIENS: int = 5  # aliens per wave
_ALIEN_W: int = 12
_ALIEN_H: int = 8
_ALIEN_SPEED: float = 0.6  # px per sub-step horizontal

# City base at the bottom
_CITY_Y: int = 175
_SECTION_W: int = 22
_SECTION_H: int = 10
_SECTION_XS = jnp.array(
    [8.0, 32.0, 56.0, 80.0, 104.0, 128.0], dtype=jnp.float32
)  # left edges of city sections

# Cannon positions (x centre)
_CANNON_LEFT_X: int = 14
_CANNON_CENTRE_X: int = 80
_CANNON_RIGHT_X: int = 146
_CANNON_Y: int = 170

_BULLET_SPEED: float = 5.0
_BULLET_W: int = 2
_BULLET_H: int = 6
_BULLET_MAX: int = 3  # up to 3 simultaneous bullets (one per cannon)

_ALIEN_SPAWN_Y: float = 20.0
_ALIEN_PASS_Y: float = float(_CITY_Y)  # aliens that reach this damage the city
_SPAWN_INTERVAL: int = 60  # sub-steps between wave spawns
_POINTS: int = 250

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 50], dtype=jnp.uint8)
_COLOR_CITY = jnp.array([0, 180, 200], dtype=jnp.uint8)
_COLOR_CITY_DMG = jnp.array([40, 40, 40], dtype=jnp.uint8)
_COLOR_ALIEN = jnp.array([200, 80, 200], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 220, 0], dtype=jnp.uint8)
_COLOR_CANNON = jnp.array([180, 200, 255], dtype=jnp.uint8)


@chex.dataclass
class AtlantisState(AtariState):
    """
    Complete Atlantis game state — a JAX pytree.

    Parameters
    ----------
    sections : jax.Array
        bool[6] — Surviving city sections.
    alien_x : jax.Array
        float32[5] — Alien x positions.
    alien_y : jax.Array
        float32[5] — Alien y positions (descend each pass).
    alien_active : jax.Array
        bool[5] — Alien on-screen.
    alien_dx : jax.Array
        float32[5] — Alien horizontal velocities.
    bullet_x : jax.Array
        float32[3] — Bullet x positions (one per cannon).
    bullet_y : jax.Array
        float32[3] — Bullet y positions.
    bullet_dx : jax.Array
        float32[3] — Bullet diagonal x velocity.
    bullet_active : jax.Array
        bool[3] — Bullet in-flight flags.
    spawn_timer : jax.Array
        int32 — Sub-steps until next alien spawns.
    wave : jax.Array
        int32 — Wave counter.
    """

    sections: jax.Array
    alien_x: jax.Array
    alien_y: jax.Array
    alien_active: jax.Array
    alien_dx: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dx: jax.Array
    bullet_active: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array


class Atlantis(AtariEnv):
    """
    Atlantis implemented as a pure JAX function suite.

    Defend the city using three gun emplacements.  Episode ends when all
    six city sections are destroyed.
    """

    num_actions: int = 4

    def _reset(self, key: jax.Array) -> AtlantisState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AtlantisState
            All city sections intact, no aliens, 6 lives.
        """
        return AtlantisState(
            sections=jnp.ones(_N_SECTIONS, dtype=jnp.bool_),
            alien_x=jnp.zeros(_N_ALIENS, dtype=jnp.float32),
            alien_y=jnp.full(_N_ALIENS, _ALIEN_SPAWN_Y, dtype=jnp.float32),
            alien_active=jnp.zeros(_N_ALIENS, dtype=jnp.bool_),
            alien_dx=jnp.array([1.0, -1.0, 1.0, -1.0, 1.0], dtype=jnp.float32),
            bullet_x=jnp.zeros(3, dtype=jnp.float32),
            bullet_y=jnp.zeros(3, dtype=jnp.float32),
            bullet_dx=jnp.zeros(3, dtype=jnp.float32),
            bullet_active=jnp.zeros(3, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            wave=jnp.int32(0),
            lives=jnp.int32(_N_SECTIONS),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: AtlantisState, action: jax.Array) -> AtlantisState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AtlantisState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE_LEFT, 2=FIRE_CENTRE, 3=FIRE_RIGHT.

        Returns
        -------
        new_state : AtlantisState
            State after one emulated frame.
        """
        step_reward = jnp.float32(0.0)

        # Fire action: which cannon (0=left, 1=centre, 2=right)
        fire_left = action == jnp.int32(1)
        fire_ctr = action == jnp.int32(2)
        fire_right = action == jnp.int32(3)
        cannon_x = jnp.array(
            [float(_CANNON_LEFT_X), float(_CANNON_CENTRE_X), float(_CANNON_RIGHT_X)],
            dtype=jnp.float32,
        )
        # Diagonal dx: left cannon → −2, centre → 0, right → +2
        cannon_dx = jnp.array([-2.0, 0.0, 2.0], dtype=jnp.float32)
        fire_flags = jnp.array([fire_left, fire_ctr, fire_right])

        new_bullet_x = jnp.where(
            fire_flags & ~state.bullet_active, cannon_x, state.bullet_x
        )
        new_bullet_y = jnp.where(
            fire_flags & ~state.bullet_active,
            jnp.float32(float(_CANNON_Y)),
            state.bullet_y,
        )
        new_bullet_dx = jnp.where(
            fire_flags & ~state.bullet_active, cannon_dx, state.bullet_dx
        )
        new_bullet_active = state.bullet_active | (fire_flags & ~state.bullet_active)

        # Advance bullets
        new_bullet_x = jnp.where(
            new_bullet_active, new_bullet_x + new_bullet_dx, new_bullet_x
        )
        new_bullet_y = jnp.where(
            new_bullet_active, new_bullet_y - _BULLET_SPEED, new_bullet_y
        )
        new_bullet_active = new_bullet_active & (new_bullet_y > jnp.float32(10.0))

        # Alien movement
        new_alien_x = jnp.where(
            state.alien_active,
            state.alien_x + state.alien_dx * _ALIEN_SPEED,
            state.alien_x,
        )
        hit_right = new_alien_x + _ALIEN_W >= jnp.float32(152.0)
        hit_left = new_alien_x <= jnp.float32(8.0)
        new_alien_dx = jnp.where(
            state.alien_active & (hit_right | hit_left),
            -state.alien_dx,
            state.alien_dx,
        )
        new_alien_x = jnp.clip(
            new_alien_x, jnp.float32(8.0), jnp.float32(152.0 - _ALIEN_W)
        )

        # Aliens descend one pass when they hit an edge
        passed_edge = state.alien_active & (hit_right | hit_left)
        new_alien_y = jnp.where(
            passed_edge, state.alien_y + jnp.float32(20.0), state.alien_y
        )

        # Bullet–alien collision: check each bullet×alien
        # Simplify: for each alien, check if any bullet overlaps
        ax = new_alien_x  # [5]
        ay = new_alien_y  # [5]
        bx = new_bullet_x  # [3]
        by = new_bullet_y  # [3]

        # Broadcast [5,1] vs [1,3] → [5,3]
        hit_x = (ax[:, None] <= bx[None, :] + _BULLET_W) & (
            ax[:, None] + _ALIEN_W >= bx[None, :]
        )
        hit_y = (ay[:, None] <= by[None, :] + _BULLET_H) & (
            ay[:, None] + _ALIEN_H >= by[None, :]
        )
        hit_mat = (
            hit_x & hit_y & state.alien_active[:, None] & new_bullet_active[None, :]
        )

        alien_hit = jnp.any(hit_mat, axis=1)  # [5]
        bullet_used = jnp.any(hit_mat, axis=0)  # [3]

        n_killed = jnp.sum(alien_hit).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_killed * _POINTS)
        new_alien_active = state.alien_active & ~alien_hit
        new_bullet_active = new_bullet_active & ~bullet_used

        # Aliens reaching city level damage sections
        alien_at_city = new_alien_active & (
            new_alien_y + _ALIEN_H >= jnp.float32(_CITY_Y)
        )
        n_dmg = jnp.sum(alien_at_city).astype(jnp.int32)

        # Find nearest section for each alien that reached city (simplified: lose 1 section per alien)
        new_sections = state.sections
        for i in range(_N_ALIENS):
            dmg = alien_at_city[i]
            # Find first remaining section
            first_alive = jnp.argmax(new_sections)
            new_sections = jnp.where(
                dmg, new_sections.at[first_alive].set(False), new_sections
            )

        new_alien_active = jnp.where(alien_at_city, jnp.bool_(False), new_alien_active)
        new_alien_y = jnp.where(alien_at_city, jnp.float32(_ALIEN_SPAWN_Y), new_alien_y)
        new_alien_x = jnp.where(alien_at_city, jnp.float32(8.0), new_alien_x)

        # Spawn new alien
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_spawn = (new_spawn_timer <= jnp.int32(0)) & jnp.any(~new_alien_active)
        new_spawn_timer = jnp.where(
            can_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )

        # Find first inactive slot
        inactive_idx = jnp.argmax(~new_alien_active)
        new_alien_active = jnp.where(
            can_spawn,
            new_alien_active.at[inactive_idx].set(True),
            new_alien_active,
        )
        new_alien_x = jnp.where(
            can_spawn,
            new_alien_x.at[inactive_idx].set(jnp.float32(8.0)),
            new_alien_x,
        )
        new_alien_y = jnp.where(
            can_spawn,
            new_alien_y.at[inactive_idx].set(jnp.float32(_ALIEN_SPAWN_Y)),
            new_alien_y,
        )

        n_alive_sections = jnp.sum(new_sections).astype(jnp.int32)
        done = n_alive_sections <= jnp.int32(0)

        return AtlantisState(
            sections=new_sections,
            alien_x=new_alien_x,
            alien_y=new_alien_y,
            alien_active=new_alien_active,
            alien_dx=new_alien_dx,
            bullet_x=new_bullet_x,
            bullet_y=new_bullet_y,
            bullet_dx=new_bullet_dx,
            bullet_active=new_bullet_active,
            spawn_timer=new_spawn_timer,
            wave=state.wave,
            lives=n_alive_sections,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: AtlantisState, action: jax.Array) -> AtlantisState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AtlantisState
            Current game state.
        action : jax.Array
            int32 — Action index (0–3).

        Returns
        -------
        new_state : AtlantisState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AtlantisState) -> AtlantisState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AtlantisState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AtlantisState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)
        frame = jnp.where(jnp.ones((210, 160, 1), dtype=jnp.bool_), _COLOR_BG, frame)

        # City sections
        def draw_section(frm, i):
            alive = state.sections[i]
            sx = _SECTION_XS[i]
            mask = (
                (_ROW_IDX >= _CITY_Y)
                & (_ROW_IDX < _CITY_Y + _SECTION_H)
                & (_COL_IDX >= jnp.int32(sx))
                & (_COL_IDX < jnp.int32(sx) + _SECTION_W)
            )
            color = jnp.where(alive, _COLOR_CITY, _COLOR_CITY_DMG)
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_section, frame, jnp.arange(_N_SECTIONS))

        # Cannons
        for cx in [_CANNON_LEFT_X, _CANNON_CENTRE_X, _CANNON_RIGHT_X]:
            cm = (
                (_ROW_IDX >= _CANNON_Y)
                & (_ROW_IDX < _CANNON_Y + 6)
                & (_COL_IDX >= cx - 3)
                & (_COL_IDX < cx + 3)
            )
            frame = jnp.where(cm[:, :, None], _COLOR_CANNON, frame)

        # Aliens
        def draw_alien(frm, i):
            alive = state.alien_active[i]
            ax = jnp.int32(state.alien_x[i])
            ay = jnp.int32(state.alien_y[i])
            mask = (
                alive
                & (_ROW_IDX >= ay)
                & (_ROW_IDX < ay + _ALIEN_H)
                & (_COL_IDX >= ax)
                & (_COL_IDX < ax + _ALIEN_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_ALIEN, frm), None

        frame, _ = jax.lax.scan(draw_alien, frame, jnp.arange(_N_ALIENS))

        # Bullets
        def draw_bullet(frm, i):
            active = state.bullet_active[i]
            bx = jnp.int32(state.bullet_x[i])
            by = jnp.int32(state.bullet_y[i])
            mask = (
                active
                & (_ROW_IDX >= by)
                & (_ROW_IDX < by + _BULLET_H)
                & (_COL_IDX >= bx)
                & (_COL_IDX < bx + _BULLET_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(3))

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Atlantis action indices.
        """
        import pygame

        return {
            pygame.K_LEFT: 1,
            pygame.K_a: 1,
            pygame.K_SPACE: 2,
            pygame.K_UP: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
        }
