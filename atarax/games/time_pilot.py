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

"""Time Pilot — JAX-native game implementation.

Omnidirectional aerial combat through multiple eras.  Your plane is always
at the screen centre; enemies scroll relative to your heading.  Clear 25
enemies to advance to the next era.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP    (aim up)
    3 — RIGHT (aim right / clockwise)
    4 — DOWN  (aim down)
    5 — LEFT  (aim left / counter-clockwise)
    6 — UP+FIRE
    7 — RIGHT+FIRE
    8 — DOWN+FIRE

Scoring (per era):
    Era 1 (WWI)  — +100 per enemy
    Era 2 (WWII) — +150 per enemy
    Era 3 (Korea) — +200
    Era 4 (Vietnam) — +300
    Era 5 (Present) — +500
    Boss destroyed — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMIES: int = 8
_PLAYER_X: int = 80
_PLAYER_Y: int = 105
_BULLET_SPEED: float = 6.0
_ENEMY_SPEED: float = 1.0
_ENEMIES_TO_CLEAR: int = 25

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([100, 150, 220], dtype=jnp.uint8)  # sky
_COLOR_PLAYER = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 80, 80], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 80, 0], dtype=jnp.uint8)

_ERA_SCORES = jnp.array([100, 150, 200, 300, 500], dtype=jnp.int32)


@chex.dataclass
class TimePilotState(AtariState):
    """
    Complete Time Pilot game state — a JAX pytree.

    Parameters
    ----------
    heading_angle : jax.Array
        float32 — Player heading (radians; 0=right).
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
        float32[8] — Enemy x (screen coords, player at centre).
    enemy_y : jax.Array
        float32[8] — Enemy y (screen coords).
    enemy_active : jax.Array
        bool[8] — Enemy alive.
    abul_x : jax.Array
        float32 — Enemy bullet x.
    abul_y : jax.Array
        float32 — Enemy bullet y.
    abul_active : jax.Array
        bool — Enemy bullet active.
    enemies_killed : jax.Array
        int32 — Enemies killed this era.
    era : jax.Array
        int32 — Current era (0–4).
    fire_timer : jax.Array
        int32 — Frames until enemy fires.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    heading_angle: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_vx: jax.Array
    bullet_vy: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    abul_x: jax.Array
    abul_y: jax.Array
    abul_active: jax.Array
    enemies_killed: jax.Array
    era: jax.Array
    fire_timer: jax.Array
    key: jax.Array


class TimePilot(AtariEnv):
    """
    Time Pilot implemented as a pure JAX function suite.

    Omnidirectional combat through eras.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> TimePilotState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : TimePilotState
            Player at centre, enemies circling at varying angles.
        """
        angles = jnp.linspace(0.0, 2 * 3.14159, _N_ENEMIES, endpoint=False)
        enemy_xs = jnp.float32(80.0) + jnp.cos(angles) * jnp.float32(50.0)
        enemy_ys = jnp.float32(105.0) + jnp.sin(angles) * jnp.float32(50.0)
        return TimePilotState(
            heading_angle=jnp.float32(0.0),
            bullet_x=jnp.float32(float(_PLAYER_X)),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_vx=jnp.float32(0.0),
            bullet_vy=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            enemy_x=enemy_xs,
            enemy_y=enemy_ys,
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            abul_x=jnp.float32(0.0),
            abul_y=jnp.float32(0.0),
            abul_active=jnp.bool_(False),
            enemies_killed=jnp.int32(0),
            era=jnp.int32(0),
            fire_timer=jnp.int32(90),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: TimePilotState, action: jax.Array) -> TimePilotState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : TimePilotState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : TimePilotState
            State after one emulated frame.
        """
        key, k_shooter, k_spawn = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Heading control
        turn_cw = (action == jnp.int32(3)) | (action == jnp.int32(7))
        turn_ccw = action == jnp.int32(5)
        turn_up = (action == jnp.int32(2)) | (action == jnp.int32(6))
        turn_dn = (action == jnp.int32(4)) | (action == jnp.int32(8))
        da = (
            jnp.where(turn_cw, jnp.float32(0.1), jnp.float32(0.0))
            + jnp.where(turn_ccw, jnp.float32(-0.1), jnp.float32(0.0))
            + jnp.where(turn_up, jnp.float32(-0.1), jnp.float32(0.0))
            + jnp.where(turn_dn, jnp.float32(0.1), jnp.float32(0.0))
        )
        new_angle = state.heading_angle + da

        # Fire bullet in heading direction
        fire = (
            (action == jnp.int32(1)) | (action >= jnp.int32(6))
        ) & ~state.bullet_active
        bvx = jnp.cos(new_angle) * _BULLET_SPEED
        bvy = jnp.sin(new_angle) * _BULLET_SPEED
        new_bx = jnp.where(fire, jnp.float32(float(_PLAYER_X)), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(float(_PLAYER_Y)), state.bullet_y)
        new_bvx = jnp.where(fire, bvx, state.bullet_vx)
        new_bvy = jnp.where(fire, bvy, state.bullet_vy)
        new_bactive = state.bullet_active | fire
        new_bx = jnp.where(new_bactive, new_bx + new_bvx, new_bx)
        new_by = jnp.where(new_bactive, new_by + new_bvy, new_by)
        new_bactive = (
            new_bactive & (new_bx > 0) & (new_bx < 160) & (new_by > 0) & (new_by < 210)
        )

        # Enemies orbit player (scroll relative to heading)
        # Scroll entire enemy field by -heading velocity
        world_vx = jnp.cos(new_angle) * _ENEMY_SPEED
        world_vy = jnp.sin(new_angle) * _ENEMY_SPEED
        new_ex = state.enemy_x - world_vx
        new_ey = state.enemy_y - world_vy
        # Wrap enemies back onto screen
        new_ex = jnp.where(
            new_ex < 0, new_ex + 160.0, jnp.where(new_ex > 160, new_ex - 160.0, new_ex)
        )
        new_ey = jnp.where(
            new_ey < 0, new_ey + 210.0, jnp.where(new_ey > 210, new_ey - 210.0, new_ey)
        )

        # Bullet hits enemy
        b_hits_e = (
            new_bactive
            & state.enemy_active
            & (jnp.abs(new_bx - new_ex) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_ey) < jnp.float32(10.0))
        )
        era_score = _ERA_SCORES[jnp.minimum(state.era, jnp.int32(4))]
        step_reward = step_reward + jnp.sum(b_hits_e).astype(
            jnp.float32
        ) * era_score.astype(jnp.float32)
        new_enemy_active = state.enemy_active & ~b_hits_e
        new_bactive = new_bactive & ~jnp.any(b_hits_e)
        new_killed = state.enemies_killed + jnp.sum(b_hits_e, dtype=jnp.int32)

        # Respawn killed enemies
        respawn = ~new_enemy_active
        rand_x = jax.random.uniform(k_spawn, (_N_ENEMIES,), minval=10.0, maxval=150.0)
        rand_y = jax.random.uniform(k_spawn, (_N_ENEMIES,), minval=20.0, maxval=190.0)
        new_ex2 = jnp.where(respawn, rand_x, new_ex)
        new_ey2 = jnp.where(respawn, rand_y, new_ey)
        new_enemy_active2 = jnp.ones(_N_ENEMIES, dtype=jnp.bool_)

        # Era advance
        era_clear = new_killed >= jnp.int32(_ENEMIES_TO_CLEAR)
        new_era = state.era + jnp.where(era_clear, jnp.int32(1), jnp.int32(0))
        new_killed2 = jnp.where(era_clear, jnp.int32(0), new_killed)

        # Enemy fires
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = new_fire_timer <= jnp.int32(0)
        rand_e = jax.random.uniform(k_shooter, (_N_ENEMIES,))
        alive_e = jnp.where(new_enemy_active2, rand_e, jnp.float32(-1.0))
        shooter_e = jnp.argmax(alive_e)
        new_abx = jnp.where(can_fire, new_ex2[shooter_e], state.abul_x)
        new_aby = jnp.where(can_fire, new_ey2[shooter_e], state.abul_y)
        new_abactive = jnp.where(can_fire, jnp.bool_(True), state.abul_active)
        new_fire_timer = jnp.where(can_fire, jnp.int32(60), new_fire_timer)
        # Enemy bullet moves toward player
        ebvx = jnp.clip((jnp.float32(_PLAYER_X) - new_abx) * 0.1, -3.0, 3.0)
        ebvy = jnp.clip((jnp.float32(_PLAYER_Y) - new_aby) * 0.1, -3.0, 3.0)
        new_abx = jnp.where(new_abactive, new_abx + ebvx, new_abx)
        new_aby = jnp.where(new_abactive, new_aby + ebvy, new_aby)
        new_abactive = (
            new_abactive
            & (new_abx > 0)
            & (new_abx < 160)
            & (new_aby > 0)
            & (new_aby < 210)
        )

        # Enemy bullet hits player
        abul_hits = (
            new_abactive
            & (jnp.abs(new_abx - jnp.float32(_PLAYER_X)) < jnp.float32(8.0))
            & (jnp.abs(new_aby - jnp.float32(_PLAYER_Y)) < jnp.float32(8.0))
        )
        # Enemy flies into player
        enemy_hits = (
            new_enemy_active2
            & (jnp.abs(new_ex2 - _PLAYER_X) < 10.0)
            & (jnp.abs(new_ey2 - _PLAYER_Y) < 10.0)
        )

        life_lost = abul_hits | jnp.any(enemy_hits)
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return TimePilotState(
            heading_angle=new_angle,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_vx=new_bvx,
            bullet_vy=new_bvy,
            bullet_active=new_bactive,
            enemy_x=new_ex2,
            enemy_y=new_ey2,
            enemy_active=new_enemy_active2,
            abul_x=new_abx,
            abul_y=new_aby,
            abul_active=new_abactive,
            enemies_killed=new_killed2,
            era=new_era,
            fire_timer=new_fire_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: TimePilotState, action: jax.Array) -> TimePilotState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : TimePilotState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : TimePilotState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: TimePilotState) -> TimePilotState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: TimePilotState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : TimePilotState
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
                & (_ROW_IDX < ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX < ex + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Enemy bullet
        abm = (
            state.abul_active
            & (_ROW_IDX >= state.abul_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.abul_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.abul_x.astype(jnp.int32) - 2)
            & (_COL_IDX < state.abul_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(abm[:, :, None], _COLOR_ENEMY_BULLET, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32) - 2)
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player (always at centre)
        pm = (
            (_ROW_IDX >= _PLAYER_Y - 6)
            & (_ROW_IDX < _PLAYER_Y + 6)
            & (_COL_IDX >= _PLAYER_X - 6)
            & (_COL_IDX < _PLAYER_X + 6)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Time Pilot action indices.
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
