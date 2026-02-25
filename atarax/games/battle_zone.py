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

"""Battle Zone — JAX-native game implementation.

A first-person tank combat game.  Destroy enemy tanks and saucers
while avoiding their shells in a flat, featureless arena.

Since true 3-D perspective is prohibitively complex in JAX, this
implementation uses a top-down 2-D approximation that preserves the
core gameplay loop.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE
    2 — FORWARD
    3 — BACK
    4 — LEFT  (rotate counter-clockwise)
    5 — RIGHT (rotate clockwise)
    6 — FORWARD + FIRE
    7 — BACK + FIRE

Scoring:
    Tank destroyed    — +1000
    Saucer destroyed  — +5000
    Super-tank        — +3000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
_N_ENEMIES: int = 4
_N_BULLETS: int = 2

_PLAYER_SPEED: float = 2.0
_ROTATE_SPEED: float = 0.1  # radians per frame
_BULLET_SPEED: float = 5.0
_ENEMY_SPEED: float = 1.0
_ENEMY_FIRE_INTERVAL: int = 60

_WORLD_SIZE: float = 400.0  # arena is ±200 units
_SCREEN_CX: float = 80.0
_SCREEN_CY: float = 130.0  # screen centre (shifted for status bar)

_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([20, 60, 20], dtype=jnp.uint8)
_COLOR_SKY = jnp.array([0, 20, 60], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([80, 200, 80], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 80], dtype=jnp.uint8)


@chex.dataclass
class BattleZoneState(AtariState):
    """
    Complete Battle Zone game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player tank x (world coords).
    player_y : jax.Array
        float32 — Player tank y (world coords).
    player_angle : jax.Array
        float32 — Heading (radians, 0 = up/north).
    bullet_x : jax.Array
        float32[2] — Bullet world x.
    bullet_y : jax.Array
        float32[2] — Bullet world y.
    bullet_active : jax.Array
        bool[2] — Bullet in-flight.
    bullet_timer : jax.Array
        int32[2] — Expiry timer.
    enemy_x : jax.Array
        float32[4] — Enemy world x.
    enemy_y : jax.Array
        float32[4] — Enemy world y.
    enemy_active : jax.Array
        bool[4] — Enemy alive.
    enemy_fire_timer : jax.Array
        int32[4] — Frames until enemy fires.
    enemy_bx : jax.Array
        float32[4] — Enemy bullet world x.
    enemy_by : jax.Array
        float32[4] — Enemy bullet world y.
    enemy_bvx : jax.Array
        float32[4] — Enemy bullet x velocity.
    enemy_bvy : jax.Array
        float32[4] — Enemy bullet y velocity.
    enemy_bactive : jax.Array
        bool[4] — Enemy bullets in-flight.
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_angle: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    bullet_timer: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_active: jax.Array
    enemy_fire_timer: jax.Array
    enemy_bx: jax.Array
    enemy_by: jax.Array
    enemy_bvx: jax.Array
    enemy_bvy: jax.Array
    enemy_bactive: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class BattleZone(AtariEnv):
    """
    Battle Zone implemented as a pure JAX function suite.

    Destroy enemy tanks in a 2-D arena.  Lives: 3.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=500_000)

    def _reset(self, key: jax.Array) -> BattleZoneState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BattleZoneState
            Player at origin, 3 lives.
        """
        key, sk = jax.random.split(key)
        enemy_positions = jax.random.uniform(sk, (4, 2)) * 200.0 - 100.0
        return BattleZoneState(
            player_x=jnp.float32(0.0),
            player_y=jnp.float32(0.0),
            player_angle=jnp.float32(0.0),
            bullet_x=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
            bullet_y=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            bullet_timer=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            enemy_x=enemy_positions[:, 0],
            enemy_y=enemy_positions[:, 1],
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            enemy_fire_timer=jnp.full(
                _N_ENEMIES, _ENEMY_FIRE_INTERVAL, dtype=jnp.int32
            ),
            enemy_bx=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_by=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_bvx=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_bvy=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_bactive=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            spawn_timer=jnp.int32(200),
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(
        self, state: BattleZoneState, action: jax.Array
    ) -> BattleZoneState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BattleZoneState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : BattleZoneState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        rotate_left = action == jnp.int32(4)
        rotate_right = action == jnp.int32(5)
        forward = (action == jnp.int32(2)) | (action == jnp.int32(6))
        back = (action == jnp.int32(3)) | (action == jnp.int32(7))
        do_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(6))
            | (action == jnp.int32(7))
        )

        # Rotation
        new_angle = (
            state.player_angle
            + jnp.where(rotate_right, _ROTATE_SPEED, 0.0)
            - jnp.where(rotate_left, _ROTATE_SPEED, 0.0)
        )

        # Forward/back movement
        move = jnp.where(forward, _PLAYER_SPEED, 0.0) - jnp.where(
            back, _PLAYER_SPEED, 0.0
        )
        new_px = jnp.clip(state.player_x + jnp.sin(new_angle) * move, -200.0, 200.0)
        new_py = jnp.clip(state.player_y - jnp.cos(new_angle) * move, -200.0, 200.0)

        # Fire bullet
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        bvx = jnp.sin(new_angle) * _BULLET_SPEED
        bvy = -jnp.cos(new_angle) * _BULLET_SPEED
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_px + bvx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_py + bvy),
            state.bullet_y,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )
        new_btimer = jnp.where(
            do_fire & has_free,
            state.bullet_timer.at[free_slot].set(jnp.int32(60)),
            state.bullet_timer,
        )

        # Move bullets
        new_bx = new_bx + jnp.where(new_bactive, bvx, 0.0)
        new_by = new_by + jnp.where(new_bactive, bvy, 0.0)
        new_btimer = new_btimer - jnp.where(new_bactive, jnp.int32(1), jnp.int32(0))
        new_bactive = new_bactive & (new_btimer > jnp.int32(0))

        # Enemy movement (circle player)
        enemy_angle = jnp.arctan2(new_px - state.enemy_x, new_py - state.enemy_y)
        new_ex = state.enemy_x + jnp.sin(enemy_angle) * _ENEMY_SPEED
        new_ey = state.enemy_y - jnp.cos(enemy_angle) * _ENEMY_SPEED

        # Bullet–enemy collision
        bul_hits = (
            new_bactive[:, None]
            & state.enemy_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_ex[None, :]) < 10.0)
            & (jnp.abs(new_by[:, None] - new_ey[None, :]) < 10.0)
        )
        enemy_killed = jnp.any(bul_hits, axis=0)
        bul_used = jnp.any(bul_hits, axis=1)
        n_killed = jnp.sum(enemy_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_killed * 1000)
        new_enemy_active = state.enemy_active & ~enemy_killed
        new_bactive = new_bactive & ~bul_used

        # Enemy fires
        new_eft = state.enemy_fire_timer - jnp.where(
            state.enemy_active, jnp.int32(1), jnp.int32(0)
        )
        fire_now = (new_eft <= jnp.int32(0)) & new_enemy_active
        new_eft = jnp.where(fire_now, jnp.int32(_ENEMY_FIRE_INTERVAL), new_eft)
        aim_vx = jnp.where(
            fire_now, jnp.sin(enemy_angle) * -_BULLET_SPEED * 0.8, state.enemy_bvx
        )
        aim_vy = jnp.where(
            fire_now, -jnp.cos(enemy_angle) * -_BULLET_SPEED * 0.8, state.enemy_bvy
        )
        new_ebx = jnp.where(fire_now, new_ex, state.enemy_bx) + jnp.where(
            state.enemy_bactive, state.enemy_bvx, 0.0
        )
        new_eby = jnp.where(fire_now, new_ey, state.enemy_by) + jnp.where(
            state.enemy_bactive, state.enemy_bvy, 0.0
        )
        new_ebactive = jnp.where(fire_now, jnp.bool_(True), state.enemy_bactive)
        # Expire far enemy bullets
        dist_eb = jnp.sqrt((new_ebx - new_px) ** 2 + (new_eby - new_py) ** 2)
        new_ebactive = new_ebactive & (dist_eb < 250.0)

        # Enemy bullet hits player
        eb_hit_player = (
            new_ebactive
            & (jnp.abs(new_ebx - new_px) < 10.0)
            & (jnp.abs(new_eby - new_py) < 10.0)
        )
        hit = jnp.any(eb_hit_player)
        new_ebactive = new_ebactive & ~eb_hit_player
        new_lives = state.lives - jnp.where(hit, jnp.int32(1), jnp.int32(0))

        # Enemy touches player
        enemy_ram = new_enemy_active & (
            jnp.sqrt((new_ex - new_px) ** 2 + (new_ey - new_py) ** 2) < 15.0
        )
        hit = hit | jnp.any(enemy_ram)
        new_lives = jnp.where(jnp.any(enemy_ram), new_lives - jnp.int32(1), new_lives)

        # Respawn enemy
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = (new_spawn_timer <= jnp.int32(0)) & ~jnp.all(new_enemy_active)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(200), new_spawn_timer)
        free_slot_e = jnp.argmin(new_enemy_active.astype(jnp.int32))
        spawn_pos = jax.random.uniform(sk2, (2,)) * 200.0 - 100.0
        new_ex = jnp.where(do_spawn, new_ex.at[free_slot_e].set(spawn_pos[0]), new_ex)
        new_ey = jnp.where(do_spawn, new_ey.at[free_slot_e].set(spawn_pos[1]), new_ey)
        new_enemy_active = jnp.where(
            do_spawn, new_enemy_active.at[free_slot_e].set(True), new_enemy_active
        )

        done = new_lives <= jnp.int32(0)

        return BattleZoneState(
            player_x=new_px,
            player_y=new_py,
            player_angle=new_angle,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            bullet_timer=new_btimer,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_active=new_enemy_active,
            enemy_fire_timer=new_eft,
            enemy_bx=new_ebx,
            enemy_by=new_eby,
            enemy_bvx=aim_vx,
            enemy_bvy=aim_vy,
            enemy_bactive=new_ebactive,
            spawn_timer=new_spawn_timer,
            wave=state.wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: BattleZoneState, action: jax.Array) -> BattleZoneState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BattleZoneState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : BattleZoneState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BattleZoneState) -> BattleZoneState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BattleZoneState) -> jax.Array:
        """
        Render the current game state as an RGB frame (top-down view).

        Parameters
        ----------
        state : BattleZoneState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.where(
            (_ROW_IDX < 110)[:, :, None],
            jnp.full((210, 160, 3), _COLOR_SKY, dtype=jnp.uint8),
            jnp.full((210, 160, 3), _COLOR_GROUND, dtype=jnp.uint8),
        )

        # Scale factor: world 200 units → screen 70 pixels
        scale = 70.0 / 200.0

        def world_to_screen(wx, wy):
            # Relative to player
            rx = (wx - state.player_x) * scale + _SCREEN_CX
            ry = (wy - state.player_y) * scale + _SCREEN_CY
            return rx.astype(jnp.int32), ry.astype(jnp.int32)

        # Enemies
        def draw_enemy(frm, i):
            ex, ey = world_to_screen(state.enemy_x[i], state.enemy_y[i])
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX <= ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX <= ex + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Player (always at screen centre)
        sx, sy = int(_SCREEN_CX), int(_SCREEN_CY)
        player_mask = (
            (_ROW_IDX >= sy - 5)
            & (_ROW_IDX <= sy + 5)
            & (_COL_IDX >= sx - 5)
            & (_COL_IDX <= sx + 5)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Battle Zone action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 3,
            pygame.K_s: 3,
            pygame.K_LEFT: 4,
            pygame.K_a: 4,
            pygame.K_RIGHT: 5,
            pygame.K_d: 5,
        }
