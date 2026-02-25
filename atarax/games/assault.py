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

"""Assault — JAX-native game implementation.

Pilot a rotating cannon at the bottom of the screen, shooting wave after
wave of descending enemy formations.  Enemies fire back; dodge or be
destroyed.

Action space (7 actions):
    0 — NOOP
    1 — FIRE
    2 — UP    (thrust forward)
    3 — RIGHT (rotate right)
    4 — DOWN  (unused)
    5 — LEFT  (rotate left)
    6 — UP + FIRE

Scoring:
    Enemy destroyed — +10 × wave
    Wave cleared    — +100 bonus
    Episode ends when all lives are lost; lives: 4.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_ENEMIES: int = 12
_N_BULLETS: int = 4
_N_ENEMY_BULLETS: int = 6

_CANNON_Y: int = 185  # player turret y position (fixed)
_CANNON_X: int = 80  # player x (can move slightly)
_CANNON_X_MIN: float = 20.0
_CANNON_X_MAX: float = 140.0
_CANNON_SPEED: float = 2.0

_BULLET_SPEED: float = 4.5
_ENEMY_SPEED: float = 0.6
_ENEMY_BULLET_SPEED: float = 2.0
_ENEMY_Y0: float = 30.0
_ENEMY_Y_LIMIT: float = 175.0

_SPAWN_INTERVAL: int = 50
_FIRE_INTERVAL: int = 20
_INIT_LIVES: int = 4

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([60, 200, 255], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([220, 80, 40], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 80], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 80, 80], dtype=jnp.uint8)


@chex.dataclass
class AssaultState(AtariState):
    """
    Complete Assault game state — a JAX pytree.

    Parameters
    ----------
    cannon_x : jax.Array
        float32 — Player cannon x (fixed y at 185).
    bullet_x : jax.Array
        float32[4] — Player bullet x positions.
    bullet_y : jax.Array
        float32[4] — Player bullet y positions.
    bullet_active : jax.Array
        bool[4] — Bullets in-flight.
    enemy_x : jax.Array
        float32[12] — Enemy x positions.
    enemy_y : jax.Array
        float32[12] — Enemy y positions.
    enemy_dx : jax.Array
        float32[12] — Enemy x velocities.
    enemy_active : jax.Array
        bool[12] — Enemy alive.
    enemy_bullet_x : jax.Array
        float32[6] — Enemy bullet x positions.
    enemy_bullet_y : jax.Array
        float32[6] — Enemy bullet y positions.
    enemy_bullet_active : jax.Array
        bool[6] — Enemy bullets in-flight.
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    fire_timer : jax.Array
        int32 — Frames until next enemy fires.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    cannon_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_dx: jax.Array
    enemy_active: jax.Array
    enemy_bullet_x: jax.Array
    enemy_bullet_y: jax.Array
    enemy_bullet_active: jax.Array
    spawn_timer: jax.Array
    fire_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class Assault(AtariEnv):
    """
    Assault implemented as a pure JAX function suite.

    Destroy incoming enemy waves.  Lives: 4.
    """

    num_actions: int = 7

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=150_000)

    def _reset(self, key: jax.Array) -> AssaultState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AssaultState
            Cannon at centre, no enemies, 4 lives.
        """
        # Initial enemy formation: 3 rows × 4 columns
        ex = jnp.array([20.0, 50.0, 80.0, 110.0] * 3, dtype=jnp.float32)
        ey = jnp.array([30.0] * 4 + [50.0] * 4 + [70.0] * 4, dtype=jnp.float32)
        return AssaultState(
            cannon_x=jnp.float32(_CANNON_X),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            enemy_x=ex,
            enemy_y=ey,
            enemy_dx=jnp.full(_N_ENEMIES, _ENEMY_SPEED, dtype=jnp.float32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            enemy_bullet_x=jnp.full(_N_ENEMY_BULLETS, -10.0, dtype=jnp.float32),
            enemy_bullet_y=jnp.full(_N_ENEMY_BULLETS, -10.0, dtype=jnp.float32),
            enemy_bullet_active=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            fire_timer=jnp.int32(_FIRE_INTERVAL),
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: AssaultState, action: jax.Array) -> AssaultState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AssaultState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : AssaultState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Cannon movement
        dx = jnp.where(action == jnp.int32(3), _CANNON_SPEED, 0.0) + jnp.where(
            action == jnp.int32(5), -_CANNON_SPEED, 0.0
        )
        new_cx = jnp.clip(state.cannon_x + dx, _CANNON_X_MIN, _CANNON_X_MAX)

        # Fire player bullet (straight up)
        do_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(2))
            | (action == jnp.int32(6))
        )
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_cx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(jnp.float32(_CANNON_Y - 10)),
            state.bullet_y,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )

        # Move bullets up
        new_by = new_by - _BULLET_SPEED
        new_bactive = new_bactive & (new_by > 0.0)

        # Move enemies (side to side, descending slowly)
        new_ex = state.enemy_x + state.enemy_dx
        # Bounce off walls
        at_wall = (new_ex < 10.0) | (new_ex > 150.0)
        new_edx = jnp.where(at_wall, -state.enemy_dx, state.enemy_dx)
        new_ex = jnp.clip(new_ex, 10.0, 150.0)
        # Descend slowly
        new_ey = state.enemy_y + jnp.float32(0.1)
        new_enemy_active = state.enemy_active

        # Bullet–enemy collision
        bul_hits_enemy = (
            new_bactive[:, None]
            & new_enemy_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_ex[None, :]) < 8.0)
            & (jnp.abs(new_by[:, None] - new_ey[None, :]) < 8.0)
        )
        enemy_killed = jnp.any(bul_hits_enemy, axis=0)
        bul_used = jnp.any(bul_hits_enemy, axis=1)
        n_killed = jnp.sum(enemy_killed).astype(jnp.int32)
        pts = (state.wave + jnp.int32(1)) * jnp.int32(10)
        step_reward = step_reward + jnp.float32(n_killed) * pts.astype(jnp.float32)
        new_enemy_active = new_enemy_active & ~enemy_killed
        new_bactive = new_bactive & ~bul_used

        # Enemy reaches ground: deactivate (player survives but threat clears)
        new_enemy_active = new_enemy_active & (new_ey < jnp.float32(_ENEMY_Y_LIMIT))

        # Wave complete
        wave_clear = ~jnp.any(new_enemy_active)
        step_reward = step_reward + jnp.where(
            wave_clear, jnp.float32(100.0), jnp.float32(0.0)
        )
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        # Reset formation on wave clear
        ex_reset = jnp.array([20.0, 50.0, 80.0, 110.0] * 3, dtype=jnp.float32)
        ey_reset = jnp.array([30.0] * 4 + [50.0] * 4 + [70.0] * 4, dtype=jnp.float32)
        new_ex = jnp.where(wave_clear, ex_reset, new_ex)
        new_ey = jnp.where(wave_clear, ey_reset, new_ey)
        new_enemy_active = jnp.where(
            wave_clear, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )

        # Enemy fires
        new_fire_timer = state.fire_timer - jnp.int32(1)
        do_enemy_fire = new_fire_timer <= jnp.int32(0)
        new_fire_timer = jnp.where(
            do_enemy_fire, jnp.int32(_FIRE_INTERVAL), new_fire_timer
        )
        shooter_idx = jax.random.randint(sk1, (), 0, _N_ENEMIES)
        free_eb = jnp.argmin(state.enemy_bullet_active.astype(jnp.int32))
        new_ebx = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_x.at[free_eb].set(new_ex[shooter_idx]),
            state.enemy_bullet_x,
        )
        new_eby = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_y.at[free_eb].set(new_ey[shooter_idx]),
            state.enemy_bullet_y,
        )
        new_ebactive = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_active.at[free_eb].set(new_enemy_active[shooter_idx]),
            state.enemy_bullet_active,
        )

        # Move enemy bullets down
        new_eby = new_eby + _ENEMY_BULLET_SPEED
        new_ebactive = new_ebactive & (new_eby < 210.0)

        # Enemy bullet hits cannon
        eb_hits = (
            new_ebactive
            & (jnp.abs(new_ebx - new_cx) < 10.0)
            & (jnp.abs(new_eby - jnp.float32(_CANNON_Y)) < 10.0)
        )
        hit = jnp.any(eb_hits)
        new_ebactive = new_ebactive & ~eb_hits

        # Enemy reaches cannon
        enemy_at_cannon = new_enemy_active & (new_ey >= jnp.float32(_ENEMY_Y_LIMIT))
        hit = hit | jnp.any(enemy_at_cannon)

        new_lives = state.lives - jnp.where(hit, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return AssaultState(
            cannon_x=new_cx,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_dx=new_edx,
            enemy_active=new_enemy_active,
            enemy_bullet_x=new_ebx,
            enemy_bullet_y=new_eby,
            enemy_bullet_active=new_ebactive,
            spawn_timer=state.spawn_timer,
            fire_timer=new_fire_timer,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: AssaultState, action: jax.Array) -> AssaultState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AssaultState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : AssaultState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AssaultState) -> AssaultState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AssaultState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AssaultState
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
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX <= ey + 5)
                & (_COL_IDX >= ex - 6)
                & (_COL_IDX <= ex + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Player bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = state.bullet_y[i].astype(jnp.int32)
            mask = (
                state.bullet_active[i]
                & (_ROW_IDX >= by - 3)
                & (_ROW_IDX <= by)
                & (_COL_IDX == bx)
            )
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Enemy bullets
        def draw_eb(frm, i):
            ebx = state.enemy_bullet_x[i].astype(jnp.int32)
            eby = state.enemy_bullet_y[i].astype(jnp.int32)
            mask = (
                state.enemy_bullet_active[i]
                & (_ROW_IDX >= eby)
                & (_ROW_IDX <= eby + 3)
                & (_COL_IDX == ebx)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_eb, frame, jnp.arange(_N_ENEMY_BULLETS))

        # Cannon
        cx = state.cannon_x.astype(jnp.int32)
        cannon_mask = (
            (_ROW_IDX >= _CANNON_Y - 6)
            & (_ROW_IDX <= _CANNON_Y + 6)
            & (_COL_IDX >= cx - 8)
            & (_COL_IDX <= cx + 8)
        )
        frame = jnp.where(cannon_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Assault action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
