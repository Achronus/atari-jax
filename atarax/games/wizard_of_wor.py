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

"""Wizard of Wor — JAX-native game implementation.

Navigate a dungeon maze and destroy Worlings, Garwors, Thorwors, and
ultimately the Wizard himself.  Enemies can become invisible.  Two-player
capable in the original; this JAX version is single-player.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — FIRE+UP
    7 — FIRE+RIGHT
    8 — FIRE+LEFT

Scoring:
    Worling — +100
    Garwor  — +200
    Thorwor — +500
    Wizard  — +2500
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMIES: int = 5
_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 5.0
_ENEMY_SPEED: float = 0.8

_DUNGEON_LEFT: int = 10
_DUNGEON_RIGHT: int = 150
_DUNGEON_TOP: int = 30
_DUNGEON_BOTTOM: int = 185

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_WALL = jnp.array([0, 100, 200], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 50, 50], dtype=jnp.uint8)
_COLOR_WIZARD = jnp.array([200, 50, 200], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 80, 0], dtype=jnp.uint8)
_COLOR_FLOOR = jnp.array([20, 20, 40], dtype=jnp.uint8)

_ENEMY_SCORES = jnp.array([100, 200, 500, 500, 2500], dtype=jnp.int32)


@chex.dataclass
class WizardOfWorState(AtariState):
    """
    Complete Wizard of Wor game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    player_dir : jax.Array
        int32 — Facing direction: 0=right, 1=up, 2=left, 3=down.
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_dx : jax.Array
        float32 — Player bullet x velocity.
    bullet_dy : jax.Array
        float32 — Player bullet y velocity.
    bullet_active : jax.Array
        bool — Player bullet in flight.
    enemy_x : jax.Array
        float32[5] — Enemy x.
    enemy_y : jax.Array
        float32[5] — Enemy y.
    enemy_type : jax.Array
        int32[5] — Enemy type (0=Worling, …, 4=Wizard).
    enemy_active : jax.Array
        bool[5] — Enemy alive.
    enemy_bx : jax.Array
        float32 — Enemy bullet x.
    enemy_by : jax.Array
        float32 — Enemy bullet y.
    enemy_bdx : jax.Array
        float32 — Enemy bullet dx.
    enemy_bdy : jax.Array
        float32 — Enemy bullet dy.
    enemy_bactive : jax.Array
        bool — Enemy bullet active.
    wave : jax.Array
        int32 — Dungeon level.
    fire_timer : jax.Array
        int32 — Frames until enemy fires.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_dir: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dx: jax.Array
    bullet_dy: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_type: jax.Array
    enemy_active: jax.Array
    enemy_bx: jax.Array
    enemy_by: jax.Array
    enemy_bdx: jax.Array
    enemy_bdy: jax.Array
    enemy_bactive: jax.Array
    wave: jax.Array
    fire_timer: jax.Array
    key: jax.Array


_DIR_DX = jnp.array([_BULLET_SPEED, 0.0, -_BULLET_SPEED, 0.0], dtype=jnp.float32)
_DIR_DY = jnp.array([0.0, -_BULLET_SPEED, 0.0, _BULLET_SPEED], dtype=jnp.float32)


class WizardOfWor(AtariEnv):
    """
    Wizard of Wor implemented as a pure JAX function suite.

    Clear dungeon of enemies including the Wizard.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> WizardOfWorState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : WizardOfWorState
            Player at bottom, 5 enemies placed in dungeon.
        """
        return WizardOfWorState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(170.0),
            player_dir=jnp.int32(0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(170.0),
            bullet_dx=jnp.float32(0.0),
            bullet_dy=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            enemy_x=jnp.array([40.0, 80.0, 120.0, 50.0, 110.0], dtype=jnp.float32),
            enemy_y=jnp.array([60.0, 80.0, 60.0, 120.0, 120.0], dtype=jnp.float32),
            enemy_type=jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            enemy_bx=jnp.float32(0.0),
            enemy_by=jnp.float32(0.0),
            enemy_bdx=jnp.float32(0.0),
            enemy_bdy=jnp.float32(0.0),
            enemy_bactive=jnp.bool_(False),
            wave=jnp.int32(1),
            fire_timer=jnp.int32(90),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: WizardOfWorState, action: jax.Array
    ) -> WizardOfWorState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : WizardOfWorState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : WizardOfWorState
            State after one emulated frame.
        """
        key, k_shooter = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        move_r = (action == 3) | (action == 7)
        move_l = (action == 5) | (action == 8)
        move_u = (action == 2) | (action == 6)
        move_d = action == 4
        dx = jnp.where(move_r, _PLAYER_SPEED, jnp.where(move_l, -_PLAYER_SPEED, 0.0))
        dy = jnp.where(move_u, -_PLAYER_SPEED, jnp.where(move_d, _PLAYER_SPEED, 0.0))
        new_px = jnp.clip(
            state.player_x + dx, float(_DUNGEON_LEFT + 2), float(_DUNGEON_RIGHT - 8)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_DUNGEON_TOP + 2), float(_DUNGEON_BOTTOM - 12)
        )

        # Update facing
        new_dir = jnp.where(
            move_r,
            0,
            jnp.where(
                move_u, 1, jnp.where(move_l, 2, jnp.where(move_d, 3, state.player_dir))
            ),
        )

        # Fire
        fire = ((action == 1) | (action >= 6)) & ~state.bullet_active
        new_bdx = jnp.where(fire, _DIR_DX[new_dir], state.bullet_dx)
        new_bdy = jnp.where(fire, _DIR_DY[new_dir], state.bullet_dy)
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, new_py + jnp.float32(4.0), state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_bx = jnp.where(new_bactive, new_bx + new_bdx, new_bx)
        new_by = jnp.where(new_bactive, new_by + new_bdy, new_by)
        new_bactive = (
            new_bactive
            & (new_bx > _DUNGEON_LEFT)
            & (new_bx < _DUNGEON_RIGHT)
            & (new_by > _DUNGEON_TOP)
            & (new_by < _DUNGEON_BOTTOM)
        )

        # Enemy movement toward player
        edx = jnp.clip((new_px - state.enemy_x) * 0.04, -_ENEMY_SPEED, _ENEMY_SPEED)
        edy = jnp.clip((new_py - state.enemy_y) * 0.04, -_ENEMY_SPEED, _ENEMY_SPEED)
        new_ex = jnp.where(state.enemy_active, state.enemy_x + edx, state.enemy_x)
        new_ey = jnp.where(state.enemy_active, state.enemy_y + edy, state.enemy_y)
        new_ex = jnp.clip(new_ex, float(_DUNGEON_LEFT + 2), float(_DUNGEON_RIGHT - 8))
        new_ey = jnp.clip(new_ey, float(_DUNGEON_TOP + 2), float(_DUNGEON_BOTTOM - 12))

        # Bullet hits enemy
        b_hits_e = (
            new_bactive
            & state.enemy_active
            & (jnp.abs(new_bx - new_ex) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_ey) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(
            jnp.where(
                b_hits_e,
                _ENEMY_SCORES[state.enemy_type],
                jnp.zeros(_N_ENEMIES, dtype=jnp.int32),
            )
        ).astype(jnp.float32)
        new_enemy_active = state.enemy_active & ~b_hits_e
        new_bactive = new_bactive & ~jnp.any(b_hits_e)

        # Enemy fires
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & jnp.any(new_enemy_active)
        rand_e = jax.random.uniform(k_shooter, (_N_ENEMIES,))
        alive_e = jnp.where(new_enemy_active, rand_e, jnp.float32(-1.0))
        shooter_e = jnp.argmax(alive_e)
        etx = new_px - new_ex[shooter_e]
        ety = new_py - new_ey[shooter_e]
        emag = jnp.sqrt(etx**2 + ety**2 + jnp.float32(1e-6))
        new_ebx = jnp.where(can_fire, new_ex[shooter_e], state.enemy_bx)
        new_eby = jnp.where(can_fire, new_ey[shooter_e], state.enemy_by)
        new_ebdx = jnp.where(can_fire, etx / emag * jnp.float32(3.0), state.enemy_bdx)
        new_ebdy = jnp.where(can_fire, ety / emag * jnp.float32(3.0), state.enemy_bdy)
        new_ebactive = jnp.where(can_fire, jnp.bool_(True), state.enemy_bactive)
        new_fire_timer = jnp.where(can_fire, jnp.int32(60), new_fire_timer)
        new_ebx = jnp.where(new_ebactive, new_ebx + new_ebdx, new_ebx)
        new_eby = jnp.where(new_ebactive, new_eby + new_ebdy, new_eby)
        new_ebactive = (
            new_ebactive
            & (new_ebx > _DUNGEON_LEFT)
            & (new_ebx < _DUNGEON_RIGHT)
            & (new_eby > _DUNGEON_TOP)
            & (new_eby < _DUNGEON_BOTTOM)
        )

        # Enemy bullet hits player
        ebul_hits = (
            new_ebactive
            & (jnp.abs(new_ebx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_eby - new_py) < jnp.float32(8.0))
        )
        # Enemy touches player
        enemy_touches = (
            new_enemy_active
            & (jnp.abs(new_ex - new_px) < 8.0)
            & (jnp.abs(new_ey - new_py) < 8.0)
        )

        # Wave clear
        wave_clear = ~jnp.any(new_enemy_active)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_enemy_active2 = jnp.where(
            wave_clear, jnp.ones(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )

        life_lost = ebul_hits | jnp.any(enemy_touches)
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return WizardOfWorState(
            player_x=new_px,
            player_y=new_py,
            player_dir=new_dir,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_dx=new_bdx,
            bullet_dy=new_bdy,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_type=state.enemy_type,
            enemy_active=new_enemy_active2,
            enemy_bx=new_ebx,
            enemy_by=new_eby,
            enemy_bdx=new_ebdx,
            enemy_bdy=new_ebdy,
            enemy_bactive=new_ebactive,
            wave=new_wave,
            fire_timer=new_fire_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: WizardOfWorState, action: jax.Array) -> WizardOfWorState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : WizardOfWorState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : WizardOfWorState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: WizardOfWorState) -> WizardOfWorState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: WizardOfWorState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : WizardOfWorState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Floor
        floor_mask = (
            (_ROW_IDX >= _DUNGEON_TOP)
            & (_ROW_IDX < _DUNGEON_BOTTOM)
            & (_COL_IDX >= _DUNGEON_LEFT)
            & (_COL_IDX < _DUNGEON_RIGHT)
        )
        frame = jnp.where(floor_mask[:, :, None], _COLOR_FLOOR, frame)

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            is_wizard = state.enemy_type[i] == jnp.int32(4)
            color = jnp.where(is_wizard, _COLOR_WIZARD, _COLOR_ENEMY)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX < ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX < ex + 5)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Enemy bullet
        ebm = (
            state.enemy_bactive
            & (_ROW_IDX >= state.enemy_by.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.enemy_by.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.enemy_bx.astype(jnp.int32) - 2)
            & (_COL_IDX < state.enemy_bx.astype(jnp.int32) + 2)
        )
        frame = jnp.where(ebm[:, :, None], _COLOR_ENEMY_BULLET, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32) - 2)
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py - 6)
            & (_ROW_IDX < py + 6)
            & (_COL_IDX >= px - 4)
            & (_COL_IDX < px + 4)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Wizard of Wor action indices.
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
