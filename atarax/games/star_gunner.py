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

"""Star Gunner — JAX-native game implementation.

Destroy waves of alien fighters that swoop in formation.  Players can
move in all four directions; enemies fly in diagonal patterns.

Action space (7 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — FIRE+UP

Scoring:
    Alien fighter — +100 (wave 1) to +400 (wave 4+)
    Wave clear bonus — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ALIENS: int = 12
_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 6.0
_ALIEN_SPEED_BASE: float = 1.2
_ALIEN_BULLET_SPEED: float = 3.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 200, 255], dtype=jnp.uint8)
_COLOR_ALIEN = jnp.array([255, 100, 50], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ALIEN_BULLET = jnp.array([255, 50, 50], dtype=jnp.uint8)


@chex.dataclass
class StarGunnerState(AtariState):
    """
    Complete Star Gunner game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    bullet_x : jax.Array
        float32 — Player bullet x.
    bullet_y : jax.Array
        float32 — Player bullet y.
    bullet_active : jax.Array
        bool — Player bullet in flight.
    alien_x : jax.Array
        float32[12] — Alien x positions.
    alien_y : jax.Array
        float32[12] — Alien y positions.
    alien_dx : jax.Array
        float32[12] — Alien x velocities.
    alien_dy : jax.Array
        float32[12] — Alien y velocities.
    alien_active : jax.Array
        bool[12] — Alien alive.
    abul_x : jax.Array
        float32 — Alien bullet x.
    abul_y : jax.Array
        float32 — Alien bullet y.
    abul_active : jax.Array
        bool — Alien bullet active.
    fire_timer : jax.Array
        int32 — Frames until alien fires.
    wave : jax.Array
        int32 — Wave number.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    alien_x: jax.Array
    alien_y: jax.Array
    alien_dx: jax.Array
    alien_dy: jax.Array
    alien_active: jax.Array
    abul_x: jax.Array
    abul_y: jax.Array
    abul_active: jax.Array
    fire_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class StarGunner(AtariEnv):
    """
    Star Gunner implemented as a pure JAX function suite.

    Destroy alien waves.  Lives: 3.
    """

    num_actions: int = 7

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> StarGunnerState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : StarGunnerState
            Player at centre-bottom, 12 aliens in formation at top.
        """
        alien_xs = jnp.tile(jnp.linspace(10.0, 150.0, 6, dtype=jnp.float32), 2)
        alien_ys = jnp.concatenate(
            [
                jnp.full(6, 20.0, dtype=jnp.float32),
                jnp.full(6, 40.0, dtype=jnp.float32),
            ]
        )
        alien_dxs = jnp.concatenate(
            [
                jnp.full(6, 1.0, dtype=jnp.float32),
                jnp.full(6, -1.0, dtype=jnp.float32),
            ]
        )
        return StarGunnerState(
            player_x=jnp.float32(76.0),
            player_y=jnp.float32(170.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(170.0),
            bullet_active=jnp.bool_(False),
            alien_x=alien_xs,
            alien_y=alien_ys,
            alien_dx=alien_dxs,
            alien_dy=jnp.zeros(_N_ALIENS, dtype=jnp.float32),
            alien_active=jnp.ones(_N_ALIENS, dtype=jnp.bool_),
            abul_x=jnp.float32(80.0),
            abul_y=jnp.float32(0.0),
            abul_active=jnp.bool_(False),
            fire_timer=jnp.int32(80),
            wave=jnp.int32(1),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: StarGunnerState, action: jax.Array
    ) -> StarGunnerState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : StarGunnerState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : StarGunnerState
            State after one emulated frame.
        """
        key, k_shooter = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        dy = jnp.where(
            action == 2, -_PLAYER_SPEED, jnp.where(action == 4, _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(state.player_x + dx, jnp.float32(5.0), jnp.float32(147.0))
        new_py = jnp.clip(state.player_y + dy, jnp.float32(100.0), jnp.float32(185.0))

        # Fire
        fire = (
            (action == jnp.int32(1)) | (action == jnp.int32(6))
        ) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, new_py - jnp.float32(8.0), state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(10.0))

        # Alien movement
        alien_speed = _ALIEN_SPEED_BASE + (
            state.wave.astype(jnp.float32) - jnp.float32(1.0)
        ) * jnp.float32(0.2)
        new_ax = state.alien_x + state.alien_dx * alien_speed
        new_ay = state.alien_y + state.alien_dy
        # Bounce at edges
        at_edge = (new_ax < jnp.float32(5.0)) | (new_ax > jnp.float32(150.0))
        new_adx = jnp.where(
            at_edge & state.alien_active, -state.alien_dx, state.alien_dx
        )
        new_ax = jnp.clip(new_ax, jnp.float32(5.0), jnp.float32(150.0))

        # Bullet hits alien
        b_hits_a = (
            new_bactive
            & state.alien_active
            & (jnp.abs(new_bx - new_ax) < jnp.float32(8.0))
            & (jnp.abs(new_by - new_ay) < jnp.float32(8.0))
        )
        wave_score = jnp.minimum(
            state.wave.astype(jnp.float32), jnp.float32(4.0)
        ) * jnp.float32(100.0)
        step_reward = step_reward + jnp.sum(b_hits_a).astype(jnp.float32) * wave_score
        new_alien_active = state.alien_active & ~b_hits_a
        new_bactive = new_bactive & ~jnp.any(b_hits_a)

        # Alien reaches player level → life lost
        alien_at_player = new_alien_active & (new_ay >= new_py - jnp.float32(8.0))
        hit_by_alien = jnp.any(alien_at_player)

        # Alien fires
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & jnp.any(new_alien_active)
        rand_a = jax.random.uniform(k_shooter, (_N_ALIENS,))
        alive_a = jnp.where(new_alien_active, rand_a, jnp.float32(-1.0))
        shooter_a = jnp.argmax(alive_a)
        new_abx = jnp.where(can_fire, new_ax[shooter_a], state.abul_x)
        new_aby = jnp.where(can_fire, new_ay[shooter_a], state.abul_y)
        new_abactive = jnp.where(can_fire, jnp.bool_(True), state.abul_active)
        new_fire_timer = jnp.where(can_fire, jnp.int32(60), new_fire_timer)
        new_aby = jnp.where(new_abactive, new_aby + _ALIEN_BULLET_SPEED, new_aby)
        new_abactive = new_abactive & (new_aby < jnp.float32(200.0))

        abul_hits_player = (
            new_abactive
            & (jnp.abs(new_abx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_aby - new_py) < jnp.float32(8.0))
        )

        # Wave clear
        wave_clear = ~jnp.any(new_alien_active)
        step_reward = step_reward + jnp.where(
            wave_clear, jnp.float32(1000.0), jnp.float32(0.0)
        )
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_alien_active2 = jnp.where(
            wave_clear, jnp.ones(_N_ALIENS, dtype=jnp.bool_), new_alien_active
        )
        new_ax2 = jnp.where(
            wave_clear,
            jnp.tile(jnp.linspace(10.0, 150.0, 6, dtype=jnp.float32), 2),
            new_ax,
        )
        new_ay2 = jnp.where(
            wave_clear,
            jnp.concatenate(
                [
                    jnp.full(6, 20.0, dtype=jnp.float32),
                    jnp.full(6, 40.0, dtype=jnp.float32),
                ]
            ),
            new_ay,
        )

        life_lost = hit_by_alien | abul_hits_player
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return StarGunnerState(
            player_x=new_px,
            player_y=new_py,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            alien_x=new_ax2,
            alien_y=new_ay2,
            alien_dx=new_adx,
            alien_dy=state.alien_dy,
            alien_active=new_alien_active2,
            abul_x=new_abx,
            abul_y=new_aby,
            abul_active=new_abactive,
            fire_timer=new_fire_timer,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: StarGunnerState, action: jax.Array) -> StarGunnerState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : StarGunnerState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : StarGunnerState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: StarGunnerState) -> StarGunnerState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: StarGunnerState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : StarGunnerState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Aliens
        def draw_alien(frm, i):
            ax = state.alien_x[i].astype(jnp.int32)
            ay = state.alien_y[i].astype(jnp.int32)
            mask = (
                state.alien_active[i]
                & (_ROW_IDX >= ay - 5)
                & (_ROW_IDX < ay + 5)
                & (_COL_IDX >= ax - 5)
                & (_COL_IDX < ax + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ALIEN, frm), None

        frame, _ = jax.lax.scan(draw_alien, frame, jnp.arange(_N_ALIENS))

        # Alien bullet
        abm = (
            state.abul_active
            & (_ROW_IDX >= state.abul_y.astype(jnp.int32))
            & (_ROW_IDX < state.abul_y.astype(jnp.int32) + 5)
            & (_COL_IDX >= state.abul_x.astype(jnp.int32))
            & (_COL_IDX < state.abul_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(abm[:, :, None], _COLOR_ALIEN_BULLET, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32))
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 6)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32))
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py - 6)
            & (_ROW_IDX < py + 6)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Star Gunner action indices.
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
