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

"""Seaquest — JAX-native game implementation.

Pilot a submarine underwater, rescuing divers and shooting enemy
submarines and sharks.  Surface periodically to replenish oxygen before
it runs out.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — UP    (surface / ascend)
    3 — RIGHT
    4 — DOWN  (descend)
    5 — LEFT

Scoring:
    Enemy sub shot — +20
    Shark shot     — +20
    Diver rescued  — +50 (per diver carried when surfacing)
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
_N_ENEMIES: int = 6  # enemy subs + sharks
_N_DIVERS: int = 6  # divers swimming around
_N_BULLETS: int = 3

_SURFACE_Y: float = 20.0  # y-coordinate of the surface
_BOTTOM_Y: float = 190.0

_SUB_SPEED: float = 1.2
_BULLET_SPEED: float = 4.0
_DIVER_SPEED: float = 0.8
_PLAYER_SPEED: float = 2.0

_OXYGEN_MAX: int = 600  # emulated frames of oxygen
_MAX_DIVERS_CARRIED: int = 6  # max divers sub can carry

_SPAWN_INTERVAL: int = 40
_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 80, 160], dtype=jnp.uint8)  # water
_COLOR_SURFACE = jnp.array([40, 140, 220], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([200, 200, 80], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_DIVER = jnp.array([100, 220, 100], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_OXYGEN = jnp.array([255, 200, 50], dtype=jnp.uint8)


@chex.dataclass
class SeaquestState(AtariState):
    """
    Complete Seaquest game state — a JAX pytree.

    Parameters
    ----------
    sub_x : jax.Array
        float32 — Player submarine x.
    sub_y : jax.Array
        float32 — Player submarine y.
    sub_dir : jax.Array
        int32 — Facing direction (+1=right, -1=left).
    bullet_x : jax.Array
        float32[3] — Bullet x positions.
    bullet_y : jax.Array
        float32[3] — Bullet y positions.
    bullet_dir : jax.Array
        int32[3] — Bullet directions.
    bullet_active : jax.Array
        bool[3] — Bullets in-flight.
    enemy_x : jax.Array
        float32[6] — Enemy x positions.
    enemy_y : jax.Array
        float32[6] — Enemy y positions.
    enemy_dir : jax.Array
        int32[6] — Enemy directions.
    enemy_active : jax.Array
        bool[6] — Enemy alive.
    diver_x : jax.Array
        float32[6] — Diver x positions.
    diver_y : jax.Array
        float32[6] — Diver y positions.
    diver_active : jax.Array
        bool[6] — Diver unrescued.
    oxygen : jax.Array
        int32 — Frames of oxygen remaining.
    divers_carried : jax.Array
        int32 — Divers currently aboard.
    spawn_timer : jax.Array
        int32 — Frames until next enemy/diver spawn.
    wave : jax.Array
        int32 — Current wave (increases difficulty).
    key : jax.Array
        uint32[2] — PRNG key.
    """

    sub_x: jax.Array
    sub_y: jax.Array
    sub_dir: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dir: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_dir: jax.Array
    enemy_active: jax.Array
    diver_x: jax.Array
    diver_y: jax.Array
    diver_active: jax.Array
    oxygen: jax.Array
    divers_carried: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class Seaquest(AtariEnv):
    """
    Seaquest implemented as a pure JAX function suite.

    Rescue divers and shoot enemies while managing oxygen.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=200_000)

    def _reset(self, key: jax.Array) -> SeaquestState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : SeaquestState
            Sub at surface, full oxygen, 3 lives.
        """
        return SeaquestState(
            sub_x=jnp.float32(80.0),
            sub_y=jnp.float32(100.0),
            sub_dir=jnp.int32(1),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_dir=jnp.ones(_N_BULLETS, dtype=jnp.int32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            enemy_x=jnp.full(_N_ENEMIES, -20.0, dtype=jnp.float32),
            enemy_y=jnp.array(
                [60.0, 80.0, 100.0, 120.0, 140.0, 160.0], dtype=jnp.float32
            ),
            enemy_dir=jnp.array([1, -1, 1, -1, 1, -1], dtype=jnp.int32),
            enemy_active=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            diver_x=jnp.full(_N_DIVERS, -20.0, dtype=jnp.float32),
            diver_y=jnp.array(
                [70.0, 90.0, 110.0, 130.0, 150.0, 170.0], dtype=jnp.float32
            ),
            diver_active=jnp.zeros(_N_DIVERS, dtype=jnp.bool_),
            oxygen=jnp.int32(_OXYGEN_MAX),
            divers_carried=jnp.int32(0),
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

    def _step_physics(self, state: SeaquestState, action: jax.Array) -> SeaquestState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : SeaquestState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : SeaquestState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(action == jnp.int32(3), _PLAYER_SPEED, 0.0) + jnp.where(
            action == jnp.int32(5), -_PLAYER_SPEED, 0.0
        )
        dy = jnp.where(action == jnp.int32(2), -_PLAYER_SPEED, 0.0) + jnp.where(
            action == jnp.int32(4), _PLAYER_SPEED, 0.0
        )
        new_sx = jnp.clip(state.sub_x + dx, 0.0, 152.0)
        new_sy = jnp.clip(state.sub_y + dy, _SURFACE_Y, _BOTTOM_Y)
        new_dir = jnp.where(
            dx > 0.0, jnp.int32(1), jnp.where(dx < 0.0, jnp.int32(-1), state.sub_dir)
        )

        # Surfacing: restore oxygen and score rescued divers
        surfaced = new_sy <= _SURFACE_Y + 2.0
        diver_score = jnp.float32(state.divers_carried * 50)
        step_reward = step_reward + jnp.where(surfaced, diver_score, jnp.float32(0.0))
        new_oxygen = jnp.where(
            surfaced, jnp.int32(_OXYGEN_MAX), state.oxygen - jnp.int32(1)
        )
        new_divers_carried = jnp.where(surfaced, jnp.int32(0), state.divers_carried)

        # Fire bullet
        do_fire = action == jnp.int32(1)
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_sx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_sy),
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

        # Move enemies
        new_ex = state.enemy_x + state.enemy_dir.astype(jnp.float32) * _SUB_SPEED
        out = (new_ex < -10.0) | (new_ex > 170.0)
        new_enemy_dir = jnp.where(out, -state.enemy_dir, state.enemy_dir)
        new_ex = jnp.clip(new_ex, 0.0, 160.0)
        new_enemy_active = state.enemy_active

        # Move divers
        new_dx = (
            state.diver_x
            + state.enemy_dir[:_N_DIVERS].astype(jnp.float32) * _DIVER_SPEED
        )
        new_dx = jnp.clip(new_dx, 0.0, 160.0)

        # Bullet–enemy collision
        bul_hits_enemy = (
            new_bactive[:, None]
            & new_enemy_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_ex[None, :]) < 8.0)
            & (jnp.abs(new_by[:, None] - state.enemy_y[None, :]) < 8.0)
        )
        enemy_killed = jnp.any(bul_hits_enemy, axis=0)
        bul_used = jnp.any(bul_hits_enemy, axis=1)
        n_killed = jnp.sum(enemy_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_killed * 20)
        new_enemy_active = new_enemy_active & ~enemy_killed
        new_bactive = new_bactive & ~bul_used

        # Player collects divers
        collects = (
            state.diver_active
            & (jnp.abs(new_dx - new_sx) < 10.0)
            & (jnp.abs(state.diver_y - new_sy) < 10.0)
            & (new_divers_carried < jnp.int32(_MAX_DIVERS_CARRIED))
        )
        n_collected = jnp.sum(collects).astype(jnp.int32)
        new_diver_active = state.diver_active & ~collects
        new_divers_carried = new_divers_carried + n_collected

        # Spawn enemies and divers
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(
            do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )
        spawn_side = jax.random.randint(sk1, (), 0, 2)
        spawn_x = jnp.where(spawn_side == 0, jnp.float32(0.0), jnp.float32(160.0))
        spawn_dir = jnp.where(spawn_side == 0, jnp.int32(1), jnp.int32(-1))
        spawn_y = (
            jax.random.uniform(sk2) * (_BOTTOM_Y - _SURFACE_Y - 30.0)
            + _SURFACE_Y
            + 15.0
        )

        free_enemy = jnp.argmin(new_enemy_active.astype(jnp.int32))
        new_ex = jnp.where(do_spawn, new_ex.at[free_enemy].set(spawn_x), new_ex)
        new_enemy_dir = jnp.where(
            do_spawn, new_enemy_dir.at[free_enemy].set(spawn_dir), new_enemy_dir
        )
        new_ey = jnp.where(
            do_spawn, state.enemy_y.at[free_enemy].set(spawn_y), state.enemy_y
        )
        new_enemy_active = jnp.where(
            do_spawn, new_enemy_active.at[free_enemy].set(True), new_enemy_active
        )

        # Enemy collides with player
        enemy_hits = (
            new_enemy_active
            & (jnp.abs(new_ex - new_sx) < 12.0)
            & (jnp.abs(new_ey - new_sy) < 8.0)
        )
        hit = jnp.any(enemy_hits)

        # Oxygen depletion
        oxygen_out = new_oxygen <= jnp.int32(0)

        life_lost = hit | oxygen_out
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_oxygen = jnp.where(oxygen_out, jnp.int32(_OXYGEN_MAX), new_oxygen)

        done = new_lives <= jnp.int32(0)

        return SeaquestState(
            sub_x=new_sx,
            sub_y=new_sy,
            sub_dir=new_dir,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_dir=new_bdir,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey,
            enemy_dir=new_enemy_dir,
            enemy_active=new_enemy_active,
            diver_x=new_dx,
            diver_y=state.diver_y,
            diver_active=new_diver_active,
            oxygen=new_oxygen,
            divers_carried=new_divers_carried,
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

    def _step(self, state: SeaquestState, action: jax.Array) -> SeaquestState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : SeaquestState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : SeaquestState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: SeaquestState) -> SeaquestState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: SeaquestState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : SeaquestState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Surface line
        frame = jnp.where(
            (_ROW_IDX < int(_SURFACE_Y + 4))[:, :, None],
            _COLOR_SURFACE,
            frame,
        )

        # Oxygen bar (top strip)
        ox_frac = state.oxygen.astype(jnp.float32) / jnp.float32(_OXYGEN_MAX)
        ox_width = (ox_frac * 140.0).astype(jnp.int32)
        ox_mask = (_ROW_IDX < 10) & (_COL_IDX < ox_width)
        frame = jnp.where(ox_mask[:, :, None], _COLOR_OXYGEN, frame)

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX <= ey + 5)
                & (_COL_IDX >= ex - 8)
                & (_COL_IDX <= ex + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Divers
        def draw_diver(frm, i):
            dvx = state.diver_x[i].astype(jnp.int32)
            dvy = state.diver_y[i].astype(jnp.int32)
            mask = (
                state.diver_active[i]
                & (_ROW_IDX >= dvy - 4)
                & (_ROW_IDX <= dvy + 4)
                & (_COL_IDX >= dvx - 3)
                & (_COL_IDX <= dvx + 3)
            )
            return jnp.where(mask[:, :, None], _COLOR_DIVER, frm), None

        frame, _ = jax.lax.scan(draw_diver, frame, jnp.arange(_N_DIVERS))

        # Bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = state.bullet_y[i].astype(jnp.int32)
            mask = (
                state.bullet_active[i]
                & (_ROW_IDX >= by - 2)
                & (_ROW_IDX <= by + 2)
                & (_COL_IDX >= bx - 2)
                & (_COL_IDX <= bx + 2)
            )
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Player sub
        sx = state.sub_x.astype(jnp.int32)
        sy = state.sub_y.astype(jnp.int32)
        sub_mask = (
            (_ROW_IDX >= sy - 5)
            & (_ROW_IDX <= sy + 5)
            & (_COL_IDX >= sx - 8)
            & (_COL_IDX <= sx + 8)
        )
        frame = jnp.where(sub_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Seaquest action indices.
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
