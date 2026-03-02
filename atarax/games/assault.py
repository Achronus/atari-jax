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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Player cannon  : y = 177–185, x ∈ [10, 150]
    Enemy formation: 3 rows × 5 cols, 12×8 px each
    Enemies start  : y = 30; descend each frame

Action space (7 actions — ALE minimal set):
    0  NOOP
    1  FIRE
    2  UP
    3  RIGHT
    4  LEFT
    5  RIGHTFIRE
    6  LEFTFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry
_CANNON_Y: int = 177
_CANNON_H: int = 8
_CANNON_W: int = 13
_PLAYER_LEFT: float = 10.0
_PLAYER_RIGHT: float = 150.0
_PLAYER_SPEED: float = 2.0

_BULLET_W: int = 1
_BULLET_H: int = 4
_BULLET_SPEED: float = 4.0
_ENEMY_BULLET_SPEED: float = 2.0

# Enemies
_N_ROWS: int = 3
_N_COLS: int = 5
_N_ENEMIES: int = _N_ROWS * _N_COLS  # 15
_ENEMY_W: int = 12
_ENEMY_H: int = 8
_ENEMY_COL_SPACING: float = 20.0
_ENEMY_ROW_SPACING: float = 20.0
_ENEMY_X0: float = 20.0
_ENEMY_Y0: float = 30.0
_ENEMY_DESCENT: float = 0.1  # px/frame

_ENEMY_POINTS: int = 10
_N_ENEMY_BULLETS: int = 3
_FIRE_INTERVAL: int = 24  # frames between enemy shots

_INIT_LIVES: int = 4
_FRAME_SKIP: int = 4
_PLAY_TOP: int = 30

# Precomputed initial positions for the 3×5 enemy grid
_ENEMY_INIT_X = (
    _ENEMY_X0
    + jnp.tile(jnp.arange(_N_COLS), _N_ROWS).astype(jnp.float32) * _ENEMY_COL_SPACING
)
_ENEMY_INIT_Y = (
    _ENEMY_Y0
    + jnp.repeat(jnp.arange(_N_ROWS), _N_COLS).astype(jnp.float32) * _ENEMY_ROW_SPACING
)

# Render
_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_BG_COLOR = jnp.array([10, 10, 40], dtype=jnp.uint8)
_PLAYER_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
_ENEMY_COLOR = jnp.array([200, 60, 60], dtype=jnp.uint8)
_ENEMY_BULLET_COLOR = jnp.array([255, 160, 0], dtype=jnp.uint8)


@chex.dataclass
class AssaultState(AtariState):
    """
    Complete Assault game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Cannon left-edge x ∈ [10, 150].
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_active : jax.Array
        bool — True while player bullet in flight.
    enemy_x : jax.Array
        float32[15] — Enemy left-edge x positions.
    enemy_y : jax.Array
        float32[15] — Enemy top-edge y positions.
    enemy_alive : jax.Array
        bool[15] — Active enemies.
    enemy_bullet_x : jax.Array
        float32[3] — Enemy bullet x slots.
    enemy_bullet_y : jax.Array
        float32[3] — Enemy bullet y slots.
    enemy_bullet_active : jax.Array
        bool[3] — Active enemy bullets.
    wave : jax.Array
        int32 — Current wave (0-based).
    fire_timer : jax.Array
        int32 — Countdown to next enemy volley.
    """

    player_x: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
    enemy_x: chex.Array
    enemy_y: chex.Array
    enemy_alive: chex.Array
    enemy_bullet_x: chex.Array
    enemy_bullet_y: chex.Array
    enemy_bullet_active: chex.Array
    wave: chex.Array
    fire_timer: chex.Array


class Assault(AtaraxGame):
    """
    Assault implemented as a pure-JAX function suite.

    Rotate a turret at the bottom of the screen, shooting waves of descending
    enemy ships before they reach your position.  Enemies fire back.
    """

    num_actions: int = 7

    def _reset(self, key: chex.PRNGKey) -> AssaultState:
        """Return the canonical initial game state."""
        return AssaultState(
            player_x=jnp.float32(80.0),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            enemy_x=_ENEMY_INIT_X.copy(),
            enemy_y=_ENEMY_INIT_Y.copy(),
            enemy_alive=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            enemy_bullet_x=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.float32),
            enemy_bullet_y=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.float32),
            enemy_bullet_active=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.bool_),
            wave=jnp.int32(0),
            fire_timer=jnp.int32(_FIRE_INTERVAL),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: AssaultState, action: jax.Array) -> AssaultState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : AssaultState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–6).

        Returns
        -------
        new_state : AssaultState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Action decode ---
        move_right = (action == 3) | (action == 5)
        move_left = (action == 4) | (action == 6)
        has_fire = (action == 1) | (action == 5) | (action == 6)

        # --- Player movement ---
        dx = jnp.where(
            move_right,
            jnp.float32(_PLAYER_SPEED),
            jnp.where(move_left, jnp.float32(-_PLAYER_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(state.player_x + dx, _PLAYER_LEFT, _PLAYER_RIGHT)

        # --- Player fires ---
        fire = has_fire & ~state.bullet_active
        new_bx = jnp.where(fire, player_x + jnp.float32(_CANNON_W / 2), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(_CANNON_Y - _BULLET_H), state.bullet_y)
        bullet_active = state.bullet_active | fire

        # --- Move player bullet up ---
        new_by = jnp.where(bullet_active, new_by - jnp.float32(_BULLET_SPEED), new_by)
        pb_oob = bullet_active & (new_by < jnp.float32(_PLAY_TOP))
        bullet_active = bullet_active & ~pb_oob

        # --- Move enemy bullets down ---
        new_eby = jnp.where(
            state.enemy_bullet_active,
            state.enemy_bullet_y + jnp.float32(_ENEMY_BULLET_SPEED),
            state.enemy_bullet_y,
        )
        eb_oob = state.enemy_bullet_active & (
            new_eby + jnp.float32(_BULLET_H) > jnp.float32(_CANNON_Y + _CANNON_H)
        )
        enemy_bullet_active = state.enemy_bullet_active & ~eb_oob

        # --- Enemy bullet hits player ---
        ebx = state.enemy_bullet_x
        eb_hit = (
            enemy_bullet_active
            & (ebx + jnp.float32(_BULLET_W) > player_x)
            & (ebx < player_x + jnp.float32(_CANNON_W))
            & (new_eby + jnp.float32(_BULLET_H) > jnp.float32(_CANNON_Y))
            & (new_eby < jnp.float32(_CANNON_Y + _CANNON_H))
        )
        lives_lost_by_bullet = jnp.sum(eb_hit.astype(jnp.int32))
        enemy_bullet_active = enemy_bullet_active & ~eb_hit

        # --- Player bullet vs enemy collision ---
        hit_mask = (
            state.enemy_alive
            & bullet_active
            & (new_bx + jnp.float32(_BULLET_W) > state.enemy_x)
            & (new_bx < state.enemy_x + jnp.float32(_ENEMY_W))
            & (new_by + jnp.float32(_BULLET_H) > state.enemy_y)
            & (new_by < state.enemy_y + jnp.float32(_ENEMY_H))
        )
        any_hit = jnp.any(hit_mask)
        new_enemy_alive = state.enemy_alive & ~hit_mask
        kills = jnp.sum(hit_mask.astype(jnp.int32))
        step_reward = jnp.float32(kills * _ENEMY_POINTS)
        bullet_active = bullet_active & ~any_hit

        # --- Enemies descend ---
        new_enemy_y = state.enemy_y + jnp.float32(_ENEMY_DESCENT)

        # --- Enemy reaching player level ---
        at_ground = new_enemy_alive & (
            new_enemy_y + jnp.float32(_ENEMY_H) >= jnp.float32(_CANNON_Y)
        )
        lives_lost_by_descent = jnp.sum(at_ground.astype(jnp.int32))
        # Remove enemies that reached ground
        new_enemy_alive = new_enemy_alive & ~at_ground

        # --- Wave clear ---
        all_clear = ~jnp.any(new_enemy_alive)
        new_wave = state.wave + jnp.where(all_clear, jnp.int32(1), jnp.int32(0))
        new_enemy_alive = jnp.where(
            all_clear,
            jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            new_enemy_alive,
        )
        new_enemy_y = jnp.where(
            all_clear,
            _ENEMY_INIT_Y.copy(),
            new_enemy_y,
        )

        # --- Enemy fires ---
        n_alive = jnp.sum(new_enemy_alive.astype(jnp.int32))
        # Pick one alive enemy to fire from
        rand_idx = jax.random.randint(subkey, (), 0, _N_ENEMIES)
        fire_enemy_alive = new_enemy_alive[rand_idx]
        spawn_bx = state.enemy_x[rand_idx] + jnp.float32(_ENEMY_W / 2)
        spawn_by = state.enemy_y[rand_idx] + jnp.float32(_ENEMY_H)

        # Use fire_timer; pick first inactive slot
        should_fire = (
            (state.fire_timer <= jnp.int32(0))
            & fire_enemy_alive
            & (n_alive > jnp.int32(0))
        )
        # Write to first available slot (slot 0 rotates using wave mod)
        slot = state.step % jnp.int32(_N_ENEMY_BULLETS)
        slot_free = ~enemy_bullet_active[slot]
        do_fire = should_fire & slot_free

        new_ebx = state.enemy_bullet_x.at[slot].set(
            jnp.where(do_fire, spawn_bx, state.enemy_bullet_x[slot])
        )
        new_eby_launch = new_eby.at[slot].set(
            jnp.where(do_fire, spawn_by, new_eby[slot])
        )
        enemy_bullet_active = enemy_bullet_active.at[slot].set(
            enemy_bullet_active[slot] | do_fire
        )

        new_fire_timer = jnp.where(
            state.fire_timer <= jnp.int32(0),
            jnp.int32(_FIRE_INTERVAL),
            state.fire_timer - jnp.int32(1),
        )

        # --- Lives ---
        new_lives = state.lives - lives_lost_by_bullet - lives_lost_by_descent
        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            player_x=player_x,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=bullet_active,
            enemy_x=state.enemy_x,
            enemy_y=new_enemy_y,
            enemy_alive=new_enemy_alive,
            enemy_bullet_x=new_ebx,
            enemy_bullet_y=new_eby_launch,
            enemy_bullet_active=enemy_bullet_active,
            wave=new_wave,
            fire_timer=new_fire_timer,
            lives=new_lives,
            score=state.score + jnp.int32(kills * _ENEMY_POINTS),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: AssaultState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> AssaultState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: AssaultState) -> jax.Array:
        frame = jnp.full((210, 160, 3), 10, dtype=jnp.uint8)
        frame = frame.at[:, :, 2].set(jnp.uint8(40))

        # --- Player cannon ---
        px = jnp.int32(state.player_x)
        cannon_mask = (
            (_ROW_IDX >= _CANNON_Y)
            & (_ROW_IDX < _CANNON_Y + _CANNON_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _CANNON_W)
        )
        frame = jnp.where(cannon_mask[:, :, None], _PLAYER_COLOR[None, None, :], frame)

        # --- Player bullet ---
        bx = jnp.int32(state.bullet_x)
        by = jnp.int32(state.bullet_y)
        pb_mask = (
            state.bullet_active
            & (_ROW_IDX >= by)
            & (_ROW_IDX < by + _BULLET_H)
            & (_COL_IDX >= bx)
            & (_COL_IDX < bx + _BULLET_W)
        )
        frame = jnp.where(pb_mask[:, :, None], _BULLET_COLOR[None, None, :], frame)

        # --- Enemies ---
        for i in range(_N_ENEMIES):
            ex = jnp.int32(state.enemy_x[i])
            ey = jnp.int32(state.enemy_y[i])
            em_mask = (
                state.enemy_alive[i]
                & (_ROW_IDX >= ey)
                & (_ROW_IDX < ey + _ENEMY_H)
                & (_COL_IDX >= ex)
                & (_COL_IDX < ex + _ENEMY_W)
            )
            frame = jnp.where(em_mask[:, :, None], _ENEMY_COLOR[None, None, :], frame)

        # --- Enemy bullets ---
        for i in range(_N_ENEMY_BULLETS):
            ebx = jnp.int32(state.enemy_bullet_x[i])
            eby = jnp.int32(state.enemy_bullet_y[i])
            eb_mask = (
                state.enemy_bullet_active[i]
                & (_ROW_IDX >= eby)
                & (_ROW_IDX < eby + _BULLET_H)
                & (_COL_IDX >= ebx)
                & (_COL_IDX < ebx + _BULLET_W)
            )
            frame = jnp.where(
                eb_mask[:, :, None], _ENEMY_BULLET_COLOR[None, None, :], frame
            )

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,
                pygame.K_UP: 2,
                pygame.K_w: 2,
                pygame.K_RIGHT: 3,
                pygame.K_d: 3,
                pygame.K_LEFT: 4,
                pygame.K_a: 4,
            }
        except ImportError:
            return {}
