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

"""Krull — JAX-native game implementation.

Prince Colwyn must navigate a series of challenges to rescue Princess
Lyssa, using the magical Glaive as a weapon.  This implementation models
the core arena segment: throw the Glaive at enemies and collect it on
its return arc.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE  (throw Glaive)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6–9 — diagonal + combinations

Scoring:
    Enemy hit by Glaive — +50
    Slayer (boss) hit   — +200
    Episode ends when all lives are lost; lives: 3.
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
_PLAYER_SPEED: float = 2.0
_GLAIVE_SPEED: float = 4.0
_ENEMY_SPEED: float = 1.0

_SCREEN_W: int = 160
_SCREEN_H: int = 210
_INIT_LIVES: int = 3
_SPAWN_INTERVAL: int = 50

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([30, 20, 40], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([220, 180, 60], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([160, 60, 200], dtype=jnp.uint8)
_COLOR_GLAIVE = jnp.array([255, 230, 80], dtype=jnp.uint8)


@chex.dataclass
class KrullState(AtariState):
    """
    Complete Krull game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    glaive_x : jax.Array
        float32 — Glaive x.
    glaive_y : jax.Array
        float32 — Glaive y.
    glaive_vx : jax.Array
        float32 — Glaive x velocity.
    glaive_vy : jax.Array
        float32 — Glaive y velocity.
    glaive_active : jax.Array
        bool — Glaive in-flight.
    glaive_returning : jax.Array
        bool — Glaive returning to player.
    enemy_x : jax.Array
        float32[6] — Enemy x positions.
    enemy_y : jax.Array
        float32[6] — Enemy y positions.
    enemy_active : jax.Array
        bool[6] — Enemy alive.
    spawn_timer : jax.Array
        int32 — Frames until next spawn.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    glaive_x: jax.Array
    glaive_y: jax.Array
    glaive_vx: jax.Array
    glaive_vy: jax.Array
    glaive_active: jax.Array
    glaive_returning: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class Krull(AtariEnv):
    """
    Krull implemented as a pure JAX function suite.

    Throw the Glaive at enemies.  Lives: 3.
    """

    num_actions: int = 10

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=200_000)

    def _reset(self, key: jax.Array) -> KrullState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : KrullState
            Player at centre, Glaive in hand, 3 lives.
        """
        return KrullState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(140.0),
            glaive_x=jnp.float32(80.0),
            glaive_y=jnp.float32(140.0),
            glaive_vx=jnp.float32(0.0),
            glaive_vy=jnp.float32(0.0),
            glaive_active=jnp.bool_(False),
            glaive_returning=jnp.bool_(False),
            enemy_x=jnp.array(
                [20.0, 60.0, 100.0, 140.0, 40.0, 120.0], dtype=jnp.float32
            ),
            enemy_y=jnp.array([40.0, 40.0, 40.0, 40.0, 80.0, 80.0], dtype=jnp.float32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: KrullState, action: jax.Array) -> KrullState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : KrullState
            Current game state.
        action : jax.Array
            int32 — Action index (0–9).

        Returns
        -------
        new_state : KrullState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        _dy = (
            jnp.array([0, 0, -1, 0, 1, 0, -1, -1, 1, 1], dtype=jnp.float32)
            * _PLAYER_SPEED
        )
        _dx = (
            jnp.array([0, 0, 0, 1, 0, -1, 1, -1, 1, -1], dtype=jnp.float32)
            * _PLAYER_SPEED
        )
        new_px = jnp.clip(state.player_x + _dx[action], 5.0, 155.0)
        new_py = jnp.clip(state.player_y + _dy[action], 20.0, 195.0)

        # Fire Glaive
        do_fire = (action == jnp.int32(1)) & ~state.glaive_active
        throw_angle = jnp.arctan2(
            jnp.float32(0.0), jnp.float32(1.0)
        )  # throw right by default
        gvx = jnp.where(do_fire, jnp.float32(_GLAIVE_SPEED), state.glaive_vx)
        gvy = jnp.where(do_fire, jnp.float32(-_GLAIVE_SPEED * 0.5), state.glaive_vy)
        new_gx = jnp.where(do_fire, new_px, state.glaive_x)
        new_gy = jnp.where(do_fire, new_py, state.glaive_y)
        new_gactive = jnp.where(do_fire, jnp.bool_(True), state.glaive_active)
        new_greturn = jnp.where(do_fire, jnp.bool_(False), state.glaive_returning)

        # Move Glaive
        new_gx = new_gx + jnp.where(new_gactive, gvx, 0.0)
        new_gy = new_gy + jnp.where(new_gactive, gvy, 0.0)

        # Bounce off walls (and begin returning)
        out = (new_gx < 5.0) | (new_gx > 155.0) | (new_gy < 20.0) | (new_gy > 195.0)
        new_greturn = new_greturn | (new_gactive & out)
        new_gx = jnp.clip(new_gx, 5.0, 155.0)
        new_gy = jnp.clip(new_gy, 20.0, 195.0)

        # Glaive returning: home toward player
        ret_dx = new_px - new_gx
        ret_dy = new_py - new_gy
        ret_dist = jnp.sqrt(ret_dx**2 + ret_dy**2) + 1e-6
        gvx = jnp.where(new_greturn, ret_dx / ret_dist * _GLAIVE_SPEED, gvx)
        gvy = jnp.where(new_greturn, ret_dy / ret_dist * _GLAIVE_SPEED, gvy)

        # Glaive caught by player
        near_player = new_greturn & (ret_dist < 8.0)
        new_gactive = new_gactive & ~near_player
        new_greturn = new_greturn & ~near_player
        new_gx = jnp.where(near_player, new_px, new_gx)
        new_gy = jnp.where(near_player, new_py, new_gy)

        # Glaive–enemy collision
        glaive_hits = (
            new_gactive
            & state.enemy_active
            & (jnp.abs(new_gx - state.enemy_x) < 10.0)
            & (jnp.abs(new_gy - state.enemy_y) < 10.0)
        )
        n_hits = jnp.sum(glaive_hits).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_hits * 50)
        new_enemy_active = state.enemy_active & ~glaive_hits
        new_greturn = new_greturn | jnp.any(glaive_hits)

        # Enemy movement (close in on player)
        edx = jnp.sign(new_px - state.enemy_x) * _ENEMY_SPEED
        edy = jnp.sign(new_py - state.enemy_y) * _ENEMY_SPEED
        new_ex = state.enemy_x + jnp.where(state.enemy_active, edx, 0.0)
        new_ey = state.enemy_y + jnp.where(state.enemy_active, edy, 0.0)

        # Enemy touches player
        enemy_catch = (
            new_enemy_active
            & (jnp.abs(new_ex - new_px) < 10.0)
            & (jnp.abs(new_ey - new_py) < 10.0)
        )
        caught = jnp.any(enemy_catch)
        new_lives = state.lives - jnp.where(caught, jnp.int32(1), jnp.int32(0))

        # Spawn new enemy
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = (new_spawn_timer <= jnp.int32(0)) & ~jnp.all(new_enemy_active)
        new_spawn_timer = jnp.where(
            do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )
        free_e = jnp.argmin(new_enemy_active.astype(jnp.int32))
        spawn_x = jax.random.uniform(sk) * 130.0 + 15.0
        new_ex = jnp.where(do_spawn, new_ex.at[free_e].set(spawn_x), new_ex)
        new_ey = jnp.where(do_spawn, new_ey.at[free_e].set(jnp.float32(40.0)), new_ey)
        new_enemy_active = jnp.where(
            do_spawn, new_enemy_active.at[free_e].set(True), new_enemy_active
        )

        # Wave clear
        all_clear = ~jnp.any(new_enemy_active)
        new_wave = state.wave + jnp.where(all_clear, jnp.int32(1), jnp.int32(0))
        new_enemy_active = jnp.where(
            all_clear, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        new_ex = jnp.where(
            all_clear,
            jnp.array([20.0, 60.0, 100.0, 140.0, 40.0, 120.0], dtype=jnp.float32),
            new_ex,
        )
        new_ey = jnp.where(
            all_clear,
            jnp.array([40.0, 40.0, 40.0, 40.0, 80.0, 80.0], dtype=jnp.float32),
            new_ey,
        )

        done = new_lives <= jnp.int32(0)

        return KrullState(
            player_x=new_px,
            player_y=new_py,
            glaive_x=new_gx,
            glaive_y=new_gy,
            glaive_vx=gvx,
            glaive_vy=gvy,
            glaive_active=new_gactive,
            glaive_returning=new_greturn,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_active=new_enemy_active,
            spawn_timer=new_spawn_timer,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: KrullState, action: jax.Array) -> KrullState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : KrullState
            Current game state.
        action : jax.Array
            int32 — Action index (0–9).

        Returns
        -------
        new_state : KrullState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: KrullState) -> KrullState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: KrullState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : KrullState
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
                & (_ROW_IDX >= ey - 6)
                & (_ROW_IDX <= ey + 6)
                & (_COL_IDX >= ex - 6)
                & (_COL_IDX <= ex + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Glaive
        gx = state.glaive_x.astype(jnp.int32)
        gy = state.glaive_y.astype(jnp.int32)
        glaive_mask = (
            state.glaive_active
            & (_ROW_IDX == gy)
            & (_COL_IDX >= gx - 3)
            & (_COL_IDX <= gx + 3)
        )
        frame = jnp.where(glaive_mask[:, :, None], _COLOR_GLAIVE, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 7)
            & (_ROW_IDX <= py + 7)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX <= px + 5)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Krull action indices.
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
