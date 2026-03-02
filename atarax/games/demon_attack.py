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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Cannon         : y = 185, x ∈ [8, 144]
    Demon formation: 3 rows × 6 cols, 12×8 px each
    Demon top      : y = 30 (initial)
    Row spacing    : 20 px
    Column spacing : 22 px

Action space (6 actions — ALE minimal set):
    0  NOOP
    1  FIRE
    2  RIGHT
    3  LEFT
    4  RIGHTFIRE
    5  LEFTFIRE
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
_PLAYER_LEFT: float = 8.0
_PLAYER_RIGHT: float = 144.0
_PLAYER_SPEED: float = 2.0

_BULLET_W: int = 1
_BULLET_H: int = 4
_BULLET_SPEED: float = 4.0
_ENEMY_BULLET_SPEED: float = 2.0

# Demons
_N_ROWS: int = 3
_N_COLS: int = 6
_N_DEMONS: int = _N_ROWS * _N_COLS  # 18
_DEMON_W: int = 12
_DEMON_H: int = 8
_COL_SPACING: float = 22.0
_ROW_SPACING: float = 20.0
_DEMON_X0: float = 10.0
_DEMON_Y0: float = 30.0
_DEMON_DESCENT: float = 0.08  # px/frame

_N_ENEMY_BULLETS: int = 3
_FIRE_INTERVAL: int = 12
_FORMATION_SPEED_INIT: float = 0.5  # formation drift px/frame at wave 0
_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4
_PLAY_TOP: int = 20

# Precomputed initial demon positions
_DEMON_INIT_X = (
    _DEMON_X0
    + jnp.tile(jnp.arange(_N_COLS), _N_ROWS).astype(jnp.float32) * _COL_SPACING
)
_DEMON_INIT_Y = (
    _DEMON_Y0
    + jnp.repeat(jnp.arange(_N_ROWS), _N_COLS).astype(jnp.float32) * _ROW_SPACING
)

# Render
_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_PLAYER_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
_DEMON_COLOR = jnp.array([220, 80, 80], dtype=jnp.uint8)
_ENEMY_BULLET_COLOR = jnp.array([255, 120, 0], dtype=jnp.uint8)


@chex.dataclass
class DemonAttackState(AtariState):
    """
    Complete Demon Attack game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `level` mirrors `wave` for API compatibility.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Cannon left-edge x ∈ [8, 144].
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_active : jax.Array
        bool — True while bullet in flight.
    demon_x : jax.Array
        float32[18] — Demon x positions.
    demon_y : jax.Array
        float32[18] — Demon y positions.
    demon_alive : jax.Array
        bool[18] — Active demons.
    enemy_bullet_x : jax.Array
        float32[3] — Enemy bullet x slots.
    enemy_bullet_y : jax.Array
        float32[3] — Enemy bullet y slots.
    enemy_bullet_active : jax.Array
        bool[3] — Active enemy bullets.
    wave : jax.Array
        int32 — Current wave (0-based); reward = (wave+1)*30 per kill.
    fire_timer : jax.Array
        int32 — Countdown to next enemy shot.
    """

    player_x: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
    demon_x: chex.Array
    demon_y: chex.Array
    demon_alive: chex.Array
    enemy_bullet_x: chex.Array
    enemy_bullet_y: chex.Array
    enemy_bullet_active: chex.Array
    wave: chex.Array
    fire_timer: chex.Array


class DemonAttack(AtaraxGame):
    """
    Demon Attack implemented as a pure-JAX function suite.

    Destroy waves of descending demons using a cannon.  Points per kill scale
    by wave: ``(wave + 1) * 30``.
    """

    num_actions: int = 6

    def _reset(self, key: chex.PRNGKey) -> DemonAttackState:
        """Return the canonical initial game state."""
        return DemonAttackState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            demon_x=_DEMON_INIT_X.copy(),
            demon_y=_DEMON_INIT_Y.copy(),
            demon_alive=jnp.ones(_N_DEMONS, dtype=jnp.bool_),
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

    def _step_physics(
        self, state: DemonAttackState, action: jax.Array
    ) -> DemonAttackState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : DemonAttackState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–5).

        Returns
        -------
        new_state : DemonAttackState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Action decode ---
        move_right = (action == 2) | (action == 4)
        move_left = (action == 3) | (action == 5)
        has_fire = (action == 1) | (action == 4) | (action == 5)

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

        # --- Demons descend ---
        # Descent speed increases slightly each wave
        wave_speed = jnp.float32(_DEMON_DESCENT) * (
            jnp.float32(1.0) + jnp.float32(0.1) * state.wave.astype(jnp.float32)
        )
        new_demon_y = state.demon_y + wave_speed

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

        # --- Player bullet vs demon collision ---
        hit_mask = (
            state.demon_alive
            & bullet_active
            & (new_bx + jnp.float32(_BULLET_W) > state.demon_x)
            & (new_bx < state.demon_x + jnp.float32(_DEMON_W))
            & (new_by + jnp.float32(_BULLET_H) > new_demon_y)
            & (new_by < new_demon_y + jnp.float32(_DEMON_H))
        )
        any_hit = jnp.any(hit_mask)
        new_demon_alive = state.demon_alive & ~hit_mask
        kills = jnp.sum(hit_mask.astype(jnp.int32))
        pts_per_kill = (state.wave + jnp.int32(1)) * jnp.int32(30)
        step_reward = jnp.float32(kills) * pts_per_kill.astype(jnp.float32)
        bullet_active = bullet_active & ~any_hit

        # --- Demon reaching player level → lose a life ---
        at_ground = new_demon_alive & (
            new_demon_y + jnp.float32(_DEMON_H) >= jnp.float32(_CANNON_Y)
        )
        lives_lost_by_descent = jnp.minimum(
            jnp.sum(at_ground.astype(jnp.int32)), jnp.int32(1)
        )
        new_demon_alive = new_demon_alive & ~at_ground

        # --- Wave clear → advance wave ---
        all_clear = ~jnp.any(new_demon_alive)
        new_wave = state.wave + jnp.where(all_clear, jnp.int32(1), jnp.int32(0))
        new_demon_alive = jnp.where(
            all_clear, jnp.ones(_N_DEMONS, dtype=jnp.bool_), new_demon_alive
        )
        new_demon_y = jnp.where(all_clear, _DEMON_INIT_Y.copy(), new_demon_y)

        # --- Enemy fires ---
        # Pick the alive demon whose x-centre is closest to the player (aimed fire).
        # This makes enemy bullets significantly more dangerous, increasing difficulty.
        n_alive = jnp.sum(new_demon_alive.astype(jnp.int32))
        player_centre = player_x + jnp.float32(_CANNON_W / 2)
        demon_centres = state.demon_x + jnp.float32(_DEMON_W / 2)
        dist_to_player = jnp.where(
            new_demon_alive,
            jnp.abs(demon_centres - player_centre),
            jnp.full(_N_DEMONS, jnp.float32(1000.0)),
        )
        rand_idx = jnp.argmin(dist_to_player)
        fire_demon_alive = new_demon_alive[rand_idx]
        spawn_bx = state.demon_x[rand_idx] + jnp.float32(_DEMON_W / 2)
        spawn_by = state.demon_y[rand_idx] + jnp.float32(_DEMON_H)

        should_fire = (
            (state.fire_timer <= jnp.int32(0))
            & fire_demon_alive
            & (n_alive > jnp.int32(0))
        )
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

        # Fire interval shrinks with wave (min 5 frames)
        fire_interval = jnp.maximum(
            jnp.int32(5),
            jnp.int32(_FIRE_INTERVAL) - state.wave * jnp.int32(2),
        )
        new_fire_timer = jnp.where(
            state.fire_timer <= jnp.int32(0),
            fire_interval,
            state.fire_timer - jnp.int32(1),
        )

        # --- Lives ---
        new_lives = state.lives - lives_lost_by_bullet - lives_lost_by_descent
        done = new_lives <= jnp.int32(0)
        new_level = new_wave

        return state.__replace__(
            player_x=player_x,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=bullet_active,
            demon_x=state.demon_x,
            demon_y=new_demon_y,
            demon_alive=new_demon_alive,
            enemy_bullet_x=new_ebx,
            enemy_bullet_y=new_eby_launch,
            enemy_bullet_active=enemy_bullet_active,
            wave=new_wave,
            fire_timer=new_fire_timer,
            lives=new_lives,
            score=state.score + jnp.int32(kills) * pts_per_kill,
            level=new_level,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: DemonAttackState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> DemonAttackState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: DemonAttackState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

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

        # --- Demons ---
        for i in range(_N_DEMONS):
            dx = jnp.int32(state.demon_x[i])
            dy = jnp.int32(state.demon_y[i])
            dm_mask = (
                state.demon_alive[i]
                & (_ROW_IDX >= dy)
                & (_ROW_IDX < dy + _DEMON_H)
                & (_COL_IDX >= dx)
                & (_COL_IDX < dx + _DEMON_W)
            )
            frame = jnp.where(dm_mask[:, :, None], _DEMON_COLOR[None, None, :], frame)

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
                pygame.K_RIGHT: 2,
                pygame.K_d: 2,
                pygame.K_LEFT: 3,
                pygame.K_a: 3,
            }
        except ImportError:
            return {}
