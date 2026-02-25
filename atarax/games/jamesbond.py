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

"""James Bond 007 — JAX-native game implementation.

James Bond must navigate between three zones — air, sea, and land —
shooting enemies in each.  Enemies approach from the right; Bond moves
left and right within his zone.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE
    2 — UP    (switch to air zone)
    3 — RIGHT
    4 — DOWN  (switch to sea zone)
    5 — LEFT
    6 — UP + FIRE
    7 — DOWN + FIRE

Scoring:
    Air enemy shot  — +100
    Sea enemy shot  — +200
    Land enemy shot — +300
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_ENEMIES: int = 6
_N_BULLETS: int = 3

_SCREEN_W: int = 160
_ZONE_Y = jnp.array([50, 110, 160], dtype=jnp.int32)  # air, sea, land y-centres
_N_ZONES: int = 3
_PLAYER_X: int = 20  # Bond stays near left
_PLAYER_SPEED: float = 1.5
_ENEMY_SPEED: float = 1.0
_BULLET_SPEED: float = 5.0
_ZONE_SCORES = jnp.array([100, 200, 300], dtype=jnp.int32)

_INIT_LIVES: int = 5
_SPAWN_INTERVAL: int = 40

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_SKY = jnp.array([60, 100, 180], dtype=jnp.uint8)
_COLOR_SEA = jnp.array([20, 60, 120], dtype=jnp.uint8)
_COLOR_LAND = jnp.array([60, 100, 30], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([220, 220, 80], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class JamesBondState(AtariState):
    """
    Complete James Bond game state — a JAX pytree.

    Parameters
    ----------
    player_zone : jax.Array
        int32 — Current zone (0=air, 1=sea, 2=land).
    player_x : jax.Array
        float32 — Player x within zone.
    enemy_x : jax.Array
        float32[6] — Enemy x positions.
    enemy_zone : jax.Array
        int32[6] — Enemy zones.
    enemy_active : jax.Array
        bool[6] — Enemy alive.
    bullet_x : jax.Array
        float32[3] — Bullet x.
    bullet_zone : jax.Array
        int32[3] — Bullet zones.
    bullet_active : jax.Array
        bool[3] — Bullets in-flight.
    spawn_timer : jax.Array
        int32 — Frames until next spawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_zone: jax.Array
    player_x: jax.Array
    enemy_x: jax.Array
    enemy_zone: jax.Array
    enemy_active: jax.Array
    bullet_x: jax.Array
    bullet_zone: jax.Array
    bullet_active: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class JamesBond(AtariEnv):
    """
    James Bond 007 implemented as a pure JAX function suite.

    Shoot enemies in three zones.  Lives: 5.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> JamesBondState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : JamesBondState
            Bond in sea zone, no enemies, 5 lives.
        """
        return JamesBondState(
            player_zone=jnp.int32(1),
            player_x=jnp.float32(float(_PLAYER_X)),
            enemy_x=jnp.full(_N_ENEMIES, 160.0, dtype=jnp.float32),
            enemy_zone=jnp.zeros(_N_ENEMIES, dtype=jnp.int32),
            enemy_active=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_zone=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: JamesBondState, action: jax.Array) -> JamesBondState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : JamesBondState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : JamesBondState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Zone switch
        zone_up = action == jnp.int32(2)
        zone_dn = action == jnp.int32(4)
        new_zone = jnp.clip(
            state.player_zone
            - jnp.where(zone_up, jnp.int32(1), jnp.int32(0))
            + jnp.where(zone_dn, jnp.int32(1), jnp.int32(0)),
            0,
            _N_ZONES - 1,
        )

        # Player horizontal movement
        move_r = action == jnp.int32(3)
        move_l = action == jnp.int32(5)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l, -_PLAYER_SPEED, 0.0),
            5.0,
            60.0,
        )

        # Fire bullet
        do_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(6))
            | (action == jnp.int32(7))
        )
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_px + 10.0),
            state.bullet_x,
        )
        new_bzone = jnp.where(
            do_fire & has_free,
            state.bullet_zone.at[free_slot].set(new_zone),
            state.bullet_zone,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )

        # Move bullets right
        new_bx = new_bx + jnp.where(new_bactive, _BULLET_SPEED, 0.0)
        new_bactive = new_bactive & (new_bx < 165.0)

        # Move enemies left
        new_ex = state.enemy_x - jnp.where(state.enemy_active, _ENEMY_SPEED, 0.0)
        # Enemies reaching left edge: catch player if same zone
        enemy_at_player = (
            state.enemy_active
            & (new_ex < new_px + 10.0)
            & (state.enemy_zone == new_zone)
        )
        any_caught = jnp.any(enemy_at_player)
        new_enemy_active = state.enemy_active & ~(new_ex < 5.0)

        # Bullet–enemy collision
        bul_hits = (
            new_bactive[:, None]
            & new_enemy_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_ex[None, :]) < 8.0)
            & (new_bzone[:, None] == state.enemy_zone[None, :])
        )
        enemy_killed = jnp.any(bul_hits, axis=0)
        bul_used = jnp.any(bul_hits, axis=1)
        scores = jnp.sum(
            jnp.where(enemy_killed, _ZONE_SCORES[state.enemy_zone], jnp.int32(0))
        )
        step_reward = step_reward + scores.astype(jnp.float32)
        new_enemy_active = new_enemy_active & ~enemy_killed
        new_bactive = new_bactive & ~bul_used

        # Spawn enemy
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(
            do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )
        spawn_zone = jax.random.randint(sk, (), 0, _N_ZONES)
        free_e = jnp.argmin(new_enemy_active.astype(jnp.int32))
        new_ex = jnp.where(do_spawn, new_ex.at[free_e].set(jnp.float32(165.0)), new_ex)
        new_ezone = jnp.where(
            do_spawn, state.enemy_zone.at[free_e].set(spawn_zone), state.enemy_zone
        )
        new_enemy_active = jnp.where(
            do_spawn, new_enemy_active.at[free_e].set(True), new_enemy_active
        )

        new_lives = state.lives - jnp.where(any_caught, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return JamesBondState(
            player_zone=new_zone,
            player_x=new_px,
            enemy_x=new_ex,
            enemy_zone=new_ezone,
            enemy_active=new_enemy_active,
            bullet_x=new_bx,
            bullet_zone=new_bzone,
            bullet_active=new_bactive,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: JamesBondState, action: jax.Array) -> JamesBondState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : JamesBondState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : JamesBondState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: JamesBondState) -> JamesBondState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: JamesBondState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : JamesBondState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_SKY, dtype=jnp.uint8)

        # Zone backgrounds
        frame = jnp.where((_ROW_IDX >= 85)[:, :, None], _COLOR_SEA, frame)
        frame = jnp.where((_ROW_IDX >= 135)[:, :, None], _COLOR_LAND, frame)

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = _ZONE_Y[state.enemy_zone[i]]
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 8)
                & (_ROW_IDX <= ey + 8)
                & (_COL_IDX >= ex - 8)
                & (_COL_IDX <= ex + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = _ZONE_Y[state.bullet_zone[i]]
            mask = (
                state.bullet_active[i]
                & (_ROW_IDX == by)
                & (_COL_IDX >= bx - 2)
                & (_COL_IDX <= bx + 2)
            )
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = _ZONE_Y[state.player_zone]
        player_mask = (
            (_ROW_IDX >= py - 8)
            & (_ROW_IDX <= py + 8)
            & (_COL_IDX >= px - 6)
            & (_COL_IDX <= px + 6)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to James Bond action indices.
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
