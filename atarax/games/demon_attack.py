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

"""Demon Attack — JAX-native game implementation.

Destroy waves of demons descending from the sky using a cannon on an ice
planet.  Demons fire back; getting hit costs a life.  Waves grow more
difficult as enemies split into two when their row is shot out.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — RIGHT + FIRE
    5 — LEFT  + FIRE

Scoring:
    Demon killed — +30 (base; higher waves award more)
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_COLS: int = 6  # demons per row
_N_ROWS: int = 3  # demon rows
_DEMON_W: int = 10
_DEMON_H: int = 8
_COL_STEP: int = 22  # horizontal spacing between demons
_ROW_STEP: int = 20  # vertical spacing between rows
_DEMON_X0: float = 8.0
_DEMON_Y0: float = 30.0

_CANNON_W: int = 8
_CANNON_H: int = 6
_CANNON_Y: int = 185
_CANNON_X0: float = 76.0
_CANNON_SPEED: float = 2.0

_BULLET_W: int = 2
_BULLET_H: int = 6
_BULLET_SPEED: float = 5.0

_ENEMY_BULLET_W: int = 2
_ENEMY_BULLET_H: int = 5
_ENEMY_BULLET_SPEED: float = 2.5

_FORMATION_SPEED: float = 0.4  # px per sub-step horizontal
_DESCENT_SPEED: float = 0.008  # px per sub-step downward
_FIRE_INTERVAL: int = 80  # sub-steps between enemy shots
_INIT_LIVES: int = 3
_BASE_POINTS: int = 30

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

# Demon column local offsets from formation left
_COL_OFFSETS = jnp.arange(_N_COLS, dtype=jnp.float32) * _COL_STEP  # [6]
_ROW_OFFSETS = jnp.arange(_N_ROWS, dtype=jnp.float32) * _ROW_STEP  # [3]

_COLOR_BG = jnp.array([0, 0, 30], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([60, 90, 120], dtype=jnp.uint8)
_COLOR_CANNON = jnp.array([160, 220, 255], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 50, 50], dtype=jnp.uint8)

# Row colours (top→bottom)
_DEMON_COLORS = jnp.array(
    [[255, 80, 0], [200, 0, 200], [255, 160, 0]],
    dtype=jnp.uint8,
)  # [3, 3]


@chex.dataclass
class DemonAttackState(AtariState):
    """
    Complete Demon Attack game state — a JAX pytree.

    Parameters
    ----------
    demons : jax.Array
        bool[3, 6] — Active demon grid.
    demon_x : jax.Array
        float32 — Formation left-edge x.
    demon_y : jax.Array
        float32 — Formation top-edge y.
    demon_dx : jax.Array
        float32 — Horizontal direction (+1 or −1).
    cannon_x : jax.Array
        float32 — Cannon left-edge x.
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_active : jax.Array
        bool — Player bullet in flight.
    enemy_bullet_x : jax.Array
        float32 — Enemy bullet x.
    enemy_bullet_y : jax.Array
        float32 — Enemy bullet y.
    enemy_bullet_active : jax.Array
        bool — Enemy bullet in flight.
    fire_timer : jax.Array
        int32 — Sub-steps until enemy fires.
    wave : jax.Array
        int32 — Current wave number (affects points and speed).
    key : jax.Array
        uint32[2] — PRNG for enemy shot selection.
    """

    demons: jax.Array
    demon_x: jax.Array
    demon_y: jax.Array
    demon_dx: jax.Array
    cannon_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    enemy_bullet_x: jax.Array
    enemy_bullet_y: jax.Array
    enemy_bullet_active: jax.Array
    fire_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class DemonAttack(AtariEnv):
    """
    Demon Attack implemented as a pure JAX function suite.

    Destroy all demons in each wave to advance.  Lives: 3.
    """

    num_actions: int = 6

    def _reset(self, key: jax.Array) -> DemonAttackState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : DemonAttackState
            Full demon grid, cannon at centre, 3 lives.
        """
        return DemonAttackState(
            demons=jnp.ones((_N_ROWS, _N_COLS), dtype=jnp.bool_),
            demon_x=jnp.float32(_DEMON_X0),
            demon_y=jnp.float32(_DEMON_Y0),
            demon_dx=jnp.float32(1.0),
            cannon_x=jnp.float32(_CANNON_X0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_CANNON_Y)),
            bullet_active=jnp.bool_(False),
            enemy_bullet_x=jnp.float32(80.0),
            enemy_bullet_y=jnp.float32(50.0),
            enemy_bullet_active=jnp.bool_(False),
            fire_timer=jnp.int32(_FIRE_INTERVAL),
            wave=jnp.int32(0),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: DemonAttackState, action: jax.Array
    ) -> DemonAttackState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : DemonAttackState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : DemonAttackState
            State after one emulated frame.
        """
        key, k_fire = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Cannon movement
        move_right = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_left = (action == jnp.int32(3)) | (action == jnp.int32(5))
        cdx = jnp.where(
            move_right,
            _CANNON_SPEED,
            jnp.where(move_left, -_CANNON_SPEED, jnp.float32(0.0)),
        )
        new_cx = jnp.clip(state.cannon_x + cdx, jnp.float32(8.0), jnp.float32(144.0))

        # Fire player bullet
        fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(4))
            | (action == jnp.int32(5))
        ) & ~state.bullet_active
        new_bx = jnp.where(fire, new_cx + jnp.float32(_CANNON_W // 2), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(_CANNON_Y), state.bullet_y)
        new_bactive = state.bullet_active | fire

        # Advance bullet
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(10.0))

        # Demon positions: formation_x + col_offset, formation_y + row_offset
        # Shape: demons[row, col] with x = demon_x + col*_COL_STEP, y = demon_y + row*_ROW_STEP
        demon_xs = (
            state.demon_x + _COL_OFFSETS[None, :]
        )  # [1, 6] → broadcast with [3,6]
        demon_ys = state.demon_y + _ROW_OFFSETS[:, None]  # [3, 1]

        # Bullet–demon collision
        bx = new_bx
        by = new_by
        hit_x = (bx >= demon_xs) & (bx < demon_xs + _DEMON_W)
        hit_y = (by >= demon_ys) & (by < demon_ys + _DEMON_H)
        hit = hit_x & hit_y & state.demons & new_bactive  # [3, 6]
        any_hit = jnp.any(hit)
        new_demons = state.demons & ~hit
        n_killed = jnp.sum(hit).astype(jnp.int32)
        pts = n_killed * (state.wave + jnp.int32(1)) * jnp.int32(_BASE_POINTS)
        step_reward = step_reward + jnp.float32(pts)

        # Deactivate bullet on hit
        new_bactive = new_bactive & ~any_hit

        # Demon formation movement
        n_alive = jnp.sum(new_demons).astype(jnp.int32)
        speed = _FORMATION_SPEED * (
            jnp.float32(1.0) + jnp.float32(0.5) * jnp.float32(state.wave)
        )
        new_dx_x = state.demon_x + speed * state.demon_dx
        right_edge = state.demon_x + jnp.float32(_N_COLS * _COL_STEP)
        hit_right = right_edge >= jnp.float32(152.0)
        hit_left = state.demon_x <= jnp.float32(8.0)
        new_demon_dx = jnp.where(hit_right | hit_left, -state.demon_dx, state.demon_dx)
        new_demon_x = jnp.where(
            hit_right,
            jnp.float32(152.0 - _N_COLS * _COL_STEP),
            jnp.where(hit_left, jnp.float32(8.0), new_dx_x),
        )
        new_demon_y = state.demon_y + _DESCENT_SPEED

        # Wave complete: reset
        wave_done = n_alive <= jnp.int32(0)
        new_wave = jnp.where(wave_done, state.wave + jnp.int32(1), state.wave)
        new_demons = jnp.where(
            wave_done, jnp.ones((_N_ROWS, _N_COLS), dtype=jnp.bool_), new_demons
        )
        new_demon_y = jnp.where(wave_done, jnp.float32(_DEMON_Y0), new_demon_y)
        new_demon_x = jnp.where(wave_done, jnp.float32(_DEMON_X0), new_demon_x)
        new_demon_dx = jnp.where(wave_done, jnp.float32(1.0), new_demon_dx)

        # Enemy fire
        new_fire_timer = state.fire_timer - jnp.int32(1)
        enemy_fires = (
            (new_fire_timer <= jnp.int32(0))
            & ~state.enemy_bullet_active
            & (n_alive > jnp.int32(0))
        )
        new_fire_timer = jnp.where(
            enemy_fires, jnp.int32(_FIRE_INTERVAL), new_fire_timer
        )

        # Pick a random alive demon to fire from
        alive_flat = new_demons.ravel()  # [18]
        rand_scores = jnp.where(
            alive_flat, jax.random.uniform(k_fire, (18,)), jnp.float32(-1.0)
        )
        fire_idx = jnp.argmax(rand_scores)
        fire_row = fire_idx // _N_COLS
        fire_col = fire_idx % _N_COLS
        eby = jnp.where(
            enemy_fires,
            new_demon_y + jnp.float32(fire_row * _ROW_STEP + _DEMON_H),
            state.enemy_bullet_y,
        )
        ebx = jnp.where(
            enemy_fires,
            new_demon_x + jnp.float32(fire_col * _COL_STEP + _DEMON_W // 2),
            state.enemy_bullet_x,
        )
        new_eb_active = state.enemy_bullet_active | enemy_fires

        # Advance enemy bullet
        new_eby = jnp.where(new_eb_active, eby + _ENEMY_BULLET_SPEED, eby)
        new_eb_active = new_eb_active & (new_eby < jnp.float32(_CANNON_Y))

        # Enemy bullet hits cannon
        cannon_cx = new_cx + jnp.float32(_CANNON_W // 2)
        cannon_cy = jnp.float32(_CANNON_Y)
        eb_hit_cannon = (
            new_eb_active
            & (jnp.abs(ebx - cannon_cx) < jnp.float32(8))
            & (jnp.abs(new_eby - cannon_cy) < jnp.float32(8))
        )
        new_lives = state.lives - jnp.where(eb_hit_cannon, jnp.int32(1), jnp.int32(0))
        new_eb_active = new_eb_active & ~eb_hit_cannon

        # Demons reaching ground
        reached_ground = new_demon_y + jnp.float32(_N_ROWS * _ROW_STEP) >= jnp.float32(
            _CANNON_Y
        )
        new_lives = jnp.where(reached_ground, jnp.int32(0), new_lives)

        done = new_lives <= jnp.int32(0)

        return DemonAttackState(
            demons=new_demons,
            demon_x=new_demon_x,
            demon_y=new_demon_y,
            demon_dx=new_demon_dx,
            cannon_x=new_cx,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            enemy_bullet_x=ebx,
            enemy_bullet_y=new_eby,
            enemy_bullet_active=new_eb_active,
            fire_timer=new_fire_timer,
            wave=new_wave,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: DemonAttackState, action: jax.Array) -> DemonAttackState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : DemonAttackState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : DemonAttackState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: DemonAttackState) -> DemonAttackState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: DemonAttackState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : DemonAttackState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)
        frame = jnp.where(jnp.ones((210, 160, 1), dtype=jnp.bool_), _COLOR_BG, frame)

        # Ground
        ground = _ROW_IDX >= _CANNON_Y + _CANNON_H
        frame = jnp.where(ground[:, :, None], _COLOR_GROUND, frame)

        # Demons — scan over 18 (3×6)
        def draw_demon(frm, idx):
            r = idx // _N_COLS
            c = idx % _N_COLS
            alive = state.demons[r, c]
            dx = jnp.int32(state.demon_x) + c * _COL_STEP
            dy = jnp.int32(state.demon_y) + r * _ROW_STEP
            color = _DEMON_COLORS[r]
            mask = (
                alive
                & (_ROW_IDX >= dy)
                & (_ROW_IDX < dy + _DEMON_H)
                & (_COL_IDX >= dx)
                & (_COL_IDX < dx + _DEMON_W)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_demon, frame, jnp.arange(_N_ROWS * _N_COLS))

        # Player bullet
        if True:
            bm = (
                state.bullet_active
                & (_ROW_IDX >= jnp.int32(state.bullet_y))
                & (_ROW_IDX < jnp.int32(state.bullet_y) + _BULLET_H)
                & (_COL_IDX >= jnp.int32(state.bullet_x))
                & (_COL_IDX < jnp.int32(state.bullet_x) + _BULLET_W)
            )
            frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Enemy bullet
        ebm = (
            state.enemy_bullet_active
            & (_ROW_IDX >= jnp.int32(state.enemy_bullet_y))
            & (_ROW_IDX < jnp.int32(state.enemy_bullet_y) + _ENEMY_BULLET_H)
            & (_COL_IDX >= jnp.int32(state.enemy_bullet_x))
            & (_COL_IDX < jnp.int32(state.enemy_bullet_x) + _ENEMY_BULLET_W)
        )
        frame = jnp.where(ebm[:, :, None], _COLOR_ENEMY_BULLET, frame)

        # Cannon
        cm = (
            (_ROW_IDX >= _CANNON_Y)
            & (_ROW_IDX < _CANNON_Y + _CANNON_H)
            & (_COL_IDX >= jnp.int32(state.cannon_x))
            & (_COL_IDX < jnp.int32(state.cannon_x) + _CANNON_W)
        )
        frame = jnp.where(cm[:, :, None], _COLOR_CANNON, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Demon Attack action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
        }
