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

"""H.E.R.O. — JAX-native game implementation.

Roderick Hero uses a helicopter backpack and laser to descend through
mine shafts, rescuing trapped miners while blasting enemies and walls
and managing energy.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE  (shoot laser)
    2 — UP    (fly up)
    3 — RIGHT
    4 — DOWN  (descend)
    5 — LEFT
    6 — UP + FIRE
    7 — RIGHT + FIRE

Scoring:
    Enemy destroyed — +75
    Wall blasted    — +10 (per section)
    Miner rescued   — +1000
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
_N_ENEMIES: int = 5
_N_WALLS: int = 6
_N_BULLETS: int = 3

_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 5.0
_ENEMY_SPEED: float = 1.0
_ENERGY_MAX: int = 2000
_INIT_LIVES: int = 4

_MINER_X: int = 80
_MINER_Y: int = 195

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 10, 30], dtype=jnp.uint8)
_COLOR_WALL = jnp.array([80, 50, 20], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 180, 255], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 80], dtype=jnp.uint8)
_COLOR_MINER = jnp.array([255, 200, 80], dtype=jnp.uint8)
_COLOR_ENERGY = jnp.array([60, 200, 60], dtype=jnp.uint8)


@chex.dataclass
class HeroState(AtariState):
    """
    Complete H.E.R.O. game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    bullet_x : jax.Array
        float32[3] — Bullet x.
    bullet_y : jax.Array
        float32[3] — Bullet y.
    bullet_dir : jax.Array
        int32[3] — Bullet direction (+1=right, -1=left).
    bullet_active : jax.Array
        bool[3] — Bullets in-flight.
    enemy_x : jax.Array
        float32[5] — Enemy x.
    enemy_y : jax.Array
        float32[5] — Enemy y.
    enemy_dir : jax.Array
        int32[5] — Enemy directions.
    enemy_active : jax.Array
        bool[5] — Enemy alive.
    wall_x : jax.Array
        float32[6] — Destructible wall x positions.
    wall_active : jax.Array
        bool[6] — Wall intact.
    energy : jax.Array
        int32 — Remaining energy.
    level : jax.Array
        int32 — Current level.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dir: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_dir: jax.Array
    enemy_active: jax.Array
    wall_x: jax.Array
    wall_active: jax.Array
    energy: jax.Array
    level: jax.Array
    key: jax.Array


_WALL_X0 = jnp.array([30.0, 70.0, 110.0, 20.0, 60.0, 100.0], dtype=jnp.float32)
_WALL_Y0 = jnp.array([80.0, 80.0, 80.0, 140.0, 140.0, 140.0], dtype=jnp.float32)
_ENEMY_X0 = jnp.array([40.0, 80.0, 120.0, 50.0, 110.0], dtype=jnp.float32)
_ENEMY_Y0 = jnp.array([60.0, 100.0, 60.0, 120.0, 120.0], dtype=jnp.float32)


class Hero(AtariEnv):
    """
    H.E.R.O. implemented as a pure JAX function suite.

    Rescue the miner through enemies and walls.  Lives: 4.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> HeroState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : HeroState
            Player at top, full energy, 4 lives.
        """
        return HeroState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(30.0),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_dir=jnp.ones(_N_BULLETS, dtype=jnp.int32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            enemy_x=_ENEMY_X0.copy(),
            enemy_y=_ENEMY_Y0.copy(),
            enemy_dir=jnp.array([1, -1, 1, -1, 1], dtype=jnp.int32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            wall_x=_WALL_X0.copy(),
            wall_active=jnp.ones(_N_WALLS, dtype=jnp.bool_),
            energy=jnp.int32(_ENERGY_MAX),
            level=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: HeroState, action: jax.Array) -> HeroState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : HeroState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : HeroState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = (
            jnp.where(action == jnp.int32(3), _PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(5), -_PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(7), _PLAYER_SPEED, 0.0)
        )
        dy = (
            jnp.where(action == jnp.int32(2), -_PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(4), _PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(6), -_PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(state.player_x + dx, 5.0, 155.0)
        new_py = jnp.clip(state.player_y + dy, 20.0, 195.0)
        new_dir = jnp.where(
            dx > 0.0, jnp.int32(1), jnp.where(dx < 0.0, jnp.int32(-1), jnp.int32(1))
        )

        # Energy drain
        new_energy = state.energy - jnp.int32(1)

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
            state.bullet_x.at[free_slot].set(new_px),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_py),
            state.bullet_y,
        )
        new_bdir = jnp.where(
            do_fire & has_free,
            state.bullet_dir.at[free_slot].set(new_dir),
            state.bullet_dir,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )

        # Move bullets
        new_bx = new_bx + new_bdir.astype(jnp.float32) * _BULLET_SPEED
        new_bactive = new_bactive & (new_bx >= 0.0) & (new_bx <= 160.0)

        # Bullet–enemy collision
        bul_hits = (
            new_bactive[:, None]
            & state.enemy_active[None, :]
            & (jnp.abs(new_bx[:, None] - state.enemy_x[None, :]) < 8.0)
            & (jnp.abs(new_by[:, None] - state.enemy_y[None, :]) < 8.0)
        )
        enemy_killed = jnp.any(bul_hits, axis=0)
        bul_used = jnp.any(bul_hits, axis=1)
        n_killed = jnp.sum(enemy_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_killed * 75)
        new_enemy_active = state.enemy_active & ~enemy_killed
        new_bactive = new_bactive & ~bul_used

        # Bullet–wall collision
        bul_hits_wall = (
            new_bactive[:, None]
            & state.wall_active[None, :]
            & (jnp.abs(new_bx[:, None] - _WALL_X0[None, :]) < 12.0)
            & (jnp.abs(new_by[:, None] - _WALL_Y0[None, :]) < 6.0)
        )
        wall_hit = jnp.any(bul_hits_wall, axis=0)
        n_walls_hit = jnp.sum(wall_hit).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_walls_hit * 10)
        new_wall_active = state.wall_active & ~wall_hit

        # Enemy movement
        new_ex = state.enemy_x + state.enemy_dir.astype(jnp.float32) * _ENEMY_SPEED
        at_edge = (new_ex < 10.0) | (new_ex > 150.0)
        new_edir = jnp.where(at_edge, -state.enemy_dir, state.enemy_dir)
        new_ex = jnp.clip(new_ex, 10.0, 150.0)

        # Enemy catches player
        enemy_hits = (
            new_enemy_active
            & (jnp.abs(new_ex - new_px) < 10.0)
            & (jnp.abs(state.enemy_y - new_py) < 10.0)
        )
        hit = jnp.any(enemy_hits)

        # Reached miner
        rescued = (jnp.abs(new_px - _MINER_X) < 12.0) & (new_py >= 185.0)
        step_reward = step_reward + jnp.where(
            rescued, jnp.float32(1000.0), jnp.float32(0.0)
        )
        new_level = state.level + jnp.where(rescued, jnp.int32(1), jnp.int32(0))
        new_px = jnp.where(rescued, jnp.float32(80.0), new_px)
        new_py = jnp.where(rescued, jnp.float32(30.0), new_py)
        new_enemy_active = jnp.where(
            rescued, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        new_wall_active = jnp.where(
            rescued, jnp.ones(_N_WALLS, dtype=jnp.bool_), new_wall_active
        )
        new_ex = jnp.where(rescued, _ENEMY_X0, new_ex)

        # Life loss
        energy_out = new_energy <= jnp.int32(0)
        life_lost = hit | energy_out
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_energy = jnp.where(energy_out, jnp.int32(_ENERGY_MAX), new_energy)

        done = new_lives <= jnp.int32(0)

        return HeroState(
            player_x=new_px,
            player_y=new_py,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_dir=new_bdir,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=state.enemy_y,
            enemy_dir=new_edir,
            enemy_active=new_enemy_active,
            wall_x=_WALL_X0,
            wall_active=new_wall_active,
            energy=new_energy,
            level=new_level,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: HeroState, action: jax.Array) -> HeroState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : HeroState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : HeroState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: HeroState) -> HeroState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: HeroState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : HeroState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Energy bar
        en_frac = state.energy.astype(jnp.float32) / jnp.float32(_ENERGY_MAX)
        en_w = (en_frac * 120.0).astype(jnp.int32)
        frame = jnp.where(
            ((_ROW_IDX < 8) & (_COL_IDX < en_w))[:, :, None], _COLOR_ENERGY, frame
        )

        # Walls
        def draw_wall(frm, i):
            wx = _WALL_X0[i].astype(jnp.int32)
            wy = _WALL_Y0[i].astype(jnp.int32)
            mask = (
                state.wall_active[i]
                & (_ROW_IDX >= wy - 5)
                & (_ROW_IDX <= wy + 5)
                & (_COL_IDX >= wx - 10)
                & (_COL_IDX <= wx + 10)
            )
            return jnp.where(mask[:, :, None], _COLOR_WALL, frm), None

        frame, _ = jax.lax.scan(draw_wall, frame, jnp.arange(_N_WALLS))

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX <= ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX <= ex + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Miner
        miner_mask = (
            (_ROW_IDX >= _MINER_Y - 6)
            & (_ROW_IDX <= _MINER_Y + 6)
            & (_COL_IDX >= _MINER_X - 5)
            & (_COL_IDX <= _MINER_X + 5)
        )
        frame = jnp.where(miner_mask[:, :, None], _COLOR_MINER, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 6)
            & (_ROW_IDX <= py + 6)
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
            Mapping of pygame key constants to H.E.R.O. action indices.
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
