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

"""Robotank — JAX-native game implementation.

Command a robot tank platoon to destroy enemy robot tanks.  The player
directs fire from a top-down perspective.  Enemy tanks approach from all
sides; shoot them before they destroy your tanks.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP    (aim up)
    3 — RIGHT (aim right)
    4 — DOWN  (aim down)
    5 — LEFT  (aim left)
    6 — UP+FIRE
    7 — RIGHT+FIRE
    8 — DOWN+FIRE

Scoring:
    Enemy tank destroyed — +1
    Episode ends when all player tanks are lost; lives: 5 (tanks).
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMY: int = 8
_PLAYER_X: int = 80
_PLAYER_Y: int = 105
_BULLET_SPEED: float = 5.0
_ENEMY_SPEED: float = 0.8

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([80, 100, 60], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 200, 100], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 80, 80], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class RobotankState(AtariState):
    """
    Complete Robotank game state — a JAX pytree.

    Parameters
    ----------
    aim_dx : jax.Array
        float32 — Aim direction x component.
    aim_dy : jax.Array
        float32 — Aim direction y component.
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_vx : jax.Array
        float32 — Bullet x velocity.
    bullet_vy : jax.Array
        float32 — Bullet y velocity.
    bullet_active : jax.Array
        bool — Bullet in flight.
    enemy_x : jax.Array
        float32[8] — Enemy x positions.
    enemy_y : jax.Array
        float32[8] — Enemy y positions.
    enemy_active : jax.Array
        bool[8] — Enemy alive.
    enemy_fire_timer : jax.Array
        int32 — Frames until enemy fires.
    enemy_bx : jax.Array
        float32 — Enemy bullet x.
    enemy_by : jax.Array
        float32 — Enemy bullet y.
    enemy_bvx : jax.Array
        float32 — Enemy bullet vx.
    enemy_bvy : jax.Array
        float32 — Enemy bullet vy.
    enemy_bactive : jax.Array
        bool — Enemy bullet active.
    wave : jax.Array
        int32 — Wave number.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    aim_dx: jax.Array
    aim_dy: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_vx: jax.Array
    bullet_vy: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    enemy_fire_timer: jax.Array
    enemy_bx: jax.Array
    enemy_by: jax.Array
    enemy_bvx: jax.Array
    enemy_bvy: jax.Array
    enemy_bactive: jax.Array
    wave: jax.Array
    key: jax.Array


class Robotank(AtariEnv):
    """
    Robotank implemented as a pure JAX function suite.

    Destroy enemy tanks before they destroy yours.  Lives (tanks): 5.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> RobotankState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : RobotankState
            Player at centre, 8 enemy tanks surrounding the field.
        """
        enemy_xs = jnp.array(
            [10.0, 80.0, 150.0, 10.0, 150.0, 10.0, 80.0, 150.0], dtype=jnp.float32
        )
        enemy_ys = jnp.array(
            [20.0, 20.0, 20.0, 105.0, 105.0, 190.0, 190.0, 190.0], dtype=jnp.float32
        )
        return RobotankState(
            aim_dx=jnp.float32(0.0),
            aim_dy=jnp.float32(-1.0),
            bullet_x=jnp.float32(float(_PLAYER_X)),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_vx=jnp.float32(0.0),
            bullet_vy=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            enemy_x=enemy_xs,
            enemy_y=enemy_ys,
            enemy_active=jnp.ones(_N_ENEMY, dtype=jnp.bool_),
            enemy_fire_timer=jnp.int32(120),
            enemy_bx=jnp.float32(0.0),
            enemy_by=jnp.float32(0.0),
            enemy_bvx=jnp.float32(0.0),
            enemy_bvy=jnp.float32(0.0),
            enemy_bactive=jnp.bool_(False),
            wave=jnp.int32(1),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: RobotankState, action: jax.Array) -> RobotankState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : RobotankState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : RobotankState
            State after one emulated frame.
        """
        key, k_shooter = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Aim direction
        new_adx = jnp.where(
            action == 3,
            jnp.float32(1.0),
            jnp.where(action == 5, jnp.float32(-1.0), state.aim_dx),
        )
        new_ady = jnp.where(
            action == 2,
            jnp.float32(-1.0),
            jnp.where(action == 4, jnp.float32(1.0), state.aim_dy),
        )
        # Normalize
        mag = jnp.sqrt(new_adx**2 + new_ady**2 + jnp.float32(1e-6))
        new_adx = new_adx / mag
        new_ady = new_ady / mag

        # Fire
        fire = (action == jnp.int32(1)) | (action >= jnp.int32(6))
        can_fire = fire & ~state.bullet_active
        new_bx = jnp.where(can_fire, jnp.float32(float(_PLAYER_X)), state.bullet_x)
        new_by = jnp.where(can_fire, jnp.float32(float(_PLAYER_Y)), state.bullet_y)
        new_bvx = jnp.where(can_fire, new_adx * _BULLET_SPEED, state.bullet_vx)
        new_bvy = jnp.where(can_fire, new_ady * _BULLET_SPEED, state.bullet_vy)
        new_bactive = state.bullet_active | can_fire

        new_bx = jnp.where(new_bactive, new_bx + new_bvx, new_bx)
        new_by = jnp.where(new_bactive, new_by + new_bvy, new_by)
        new_bactive = (
            new_bactive & (new_bx > 0) & (new_bx < 160) & (new_by > 0) & (new_by < 210)
        )

        # Enemy movement toward player
        edx = jnp.clip(
            (jnp.float32(_PLAYER_X) - state.enemy_x) * 0.02, -_ENEMY_SPEED, _ENEMY_SPEED
        )
        edy = jnp.clip(
            (jnp.float32(_PLAYER_Y) - state.enemy_y) * 0.02, -_ENEMY_SPEED, _ENEMY_SPEED
        )
        new_ex = jnp.where(state.enemy_active, state.enemy_x + edx, state.enemy_x)
        new_ey = jnp.where(state.enemy_active, state.enemy_y + edy, state.enemy_y)

        # Bullet hits enemy
        bullet_hit = (
            new_bactive
            & state.enemy_active
            & (jnp.abs(new_bx - new_ex) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_ey) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(bullet_hit).astype(jnp.float32)
        new_enemy_active = state.enemy_active & ~bullet_hit
        new_bactive = new_bactive & ~jnp.any(bullet_hit)

        # Enemy fires at player
        new_eft = state.enemy_fire_timer - jnp.int32(1)
        can_enemy_fire = (new_eft <= jnp.int32(0)) & jnp.any(new_enemy_active)
        rand_e = jax.random.uniform(k_shooter, (_N_ENEMY,))
        alive_scores_e = jnp.where(new_enemy_active, rand_e, jnp.float32(-1.0))
        shooter_e = jnp.argmax(alive_scores_e)
        etx = new_ex[shooter_e]
        ety = new_ey[shooter_e]
        edirx = jnp.float32(_PLAYER_X) - etx
        ediry = jnp.float32(_PLAYER_Y) - ety
        emag = jnp.sqrt(edirx**2 + ediry**2 + jnp.float32(1e-6))
        new_ebx = jnp.where(can_enemy_fire, etx, state.enemy_bx)
        new_eby = jnp.where(can_enemy_fire, ety, state.enemy_by)
        new_ebvx = jnp.where(
            can_enemy_fire, edirx / emag * _BULLET_SPEED * 0.7, state.enemy_bvx
        )
        new_ebvy = jnp.where(
            can_enemy_fire, ediry / emag * _BULLET_SPEED * 0.7, state.enemy_bvy
        )
        new_ebactive = jnp.where(can_enemy_fire, jnp.bool_(True), state.enemy_bactive)
        new_eft = jnp.where(can_enemy_fire, jnp.int32(90), new_eft)

        new_ebx = jnp.where(new_ebactive, new_ebx + new_ebvx, new_ebx)
        new_eby = jnp.where(new_ebactive, new_eby + new_ebvy, new_eby)
        new_ebactive = (
            new_ebactive
            & (new_ebx > 0)
            & (new_ebx < 160)
            & (new_eby > 0)
            & (new_eby < 210)
        )

        # Enemy bullet hits player
        enemy_bullet_hits = (
            new_ebactive
            & (jnp.abs(new_ebx - jnp.float32(_PLAYER_X)) < jnp.float32(8.0))
            & (jnp.abs(new_eby - jnp.float32(_PLAYER_Y)) < jnp.float32(8.0))
        )

        # Enemy reaches player
        enemy_reaches = (
            state.enemy_active
            & (jnp.abs(new_ex - _PLAYER_X) < 10.0)
            & (jnp.abs(new_ey - _PLAYER_Y) < 10.0)
        )

        # Wave clear
        wave_clear = ~jnp.any(new_enemy_active)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_enemy_active2 = jnp.where(
            wave_clear, jnp.ones(_N_ENEMY, dtype=jnp.bool_), new_enemy_active
        )

        life_lost = enemy_bullet_hits | jnp.any(enemy_reaches)
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return RobotankState(
            aim_dx=new_adx,
            aim_dy=new_ady,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_vx=new_bvx,
            bullet_vy=new_bvy,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_active=new_enemy_active2,
            enemy_fire_timer=new_eft,
            enemy_bx=new_ebx,
            enemy_by=new_eby,
            enemy_bvx=new_ebvx,
            enemy_bvy=new_ebvy,
            enemy_bactive=new_ebactive,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: RobotankState, action: jax.Array) -> RobotankState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : RobotankState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : RobotankState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: RobotankState) -> RobotankState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: RobotankState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : RobotankState
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
                & (_ROW_IDX < ey + 6)
                & (_COL_IDX >= ex - 6)
                & (_COL_IDX < ex + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMY))

        # Player tank
        pm = (
            (_ROW_IDX >= _PLAYER_Y - 8)
            & (_ROW_IDX < _PLAYER_Y + 8)
            & (_COL_IDX >= _PLAYER_X - 8)
            & (_COL_IDX < _PLAYER_X + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32) - 2)
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Robotank action indices.
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
