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

"""Defender — JAX-native game implementation.

Protect humanoids on the ground from alien abductors.  Your ship scrolls
horizontally; shoot landers and mutants while keeping the humanoids alive.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — DOWN
    4 — THRUST (accelerate right)
    5 — REVERSE (flip direction)
    6 — SMART BOMB
    7 — HYPERSPACE
    8 — FIRE+THRUST

Scoring:
    Lander shot — +150
    Mutant shot — +150
    Humanoid rescued (catches falling humanoid) — +500
    Episode ends when all lives lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_LANDERS: int = 8
_N_HUMANOIDS: int = 6
_LANDER_SPEED: float = 1.0
_BULLET_SPEED: float = 6.0
_PLAYER_SPEED_MAX: float = 4.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 30], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([60, 120, 60], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_LANDER = jnp.array([200, 50, 200], dtype=jnp.uint8)
_COLOR_HUMANOID = jnp.array([100, 200, 255], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 200, 0], dtype=jnp.uint8)

_GROUND_Y: int = 185
_PLAYER_Y_INIT: float = 100.0


@chex.dataclass
class DefenderState(AtariState):
    """
    Complete Defender game state — a JAX pytree.

    Parameters
    ----------
    ship_x : jax.Array
        float32 — Ship x (scrolling coordinate).
    ship_y : jax.Array
        float32 — Ship y.
    ship_vx : jax.Array
        float32 — Ship horizontal velocity.
    ship_dir : jax.Array
        int32 — Direction (1=right, -1=left).
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    lander_x : jax.Array
        float32[8] — Lander x (world coords).
    lander_y : jax.Array
        float32[8] — Lander y.
    lander_active : jax.Array
        bool[8] — Lander alive.
    humanoid_x : jax.Array
        float32[6] — Humanoid x positions.
    humanoid_alive : jax.Array
        bool[6] — Humanoid alive.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    ship_x: jax.Array
    ship_y: jax.Array
    ship_vx: jax.Array
    ship_dir: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    lander_x: jax.Array
    lander_y: jax.Array
    lander_active: jax.Array
    humanoid_x: jax.Array
    humanoid_alive: jax.Array
    wave: jax.Array
    key: jax.Array


class Defender(AtariEnv):
    """
    Defender implemented as a pure JAX function suite.

    Protect humanoids from alien landers.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> DefenderState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : DefenderState
            Ship at left, 8 landers spread out, 6 humanoids on ground.
        """
        lander_xs = jnp.linspace(20.0, 620.0, _N_LANDERS, dtype=jnp.float32)
        humanoid_xs = jnp.linspace(30.0, 590.0, _N_HUMANOIDS, dtype=jnp.float32)
        return DefenderState(
            ship_x=jnp.float32(0.0),
            ship_y=jnp.float32(_PLAYER_Y_INIT),
            ship_vx=jnp.float32(0.0),
            ship_dir=jnp.int32(1),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(_PLAYER_Y_INIT),
            bullet_active=jnp.bool_(False),
            lander_x=lander_xs,
            lander_y=jnp.full(_N_LANDERS, 50.0, dtype=jnp.float32),
            lander_active=jnp.ones(_N_LANDERS, dtype=jnp.bool_),
            humanoid_x=humanoid_xs,
            humanoid_alive=jnp.ones(_N_HUMANOIDS, dtype=jnp.bool_),
            wave=jnp.int32(1),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: DefenderState, action: jax.Array) -> DefenderState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : DefenderState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : DefenderState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Thrust / reverse
        thrust = (action == jnp.int32(4)) | (action == jnp.int32(8))
        reverse = action == jnp.int32(5)
        new_dir = jnp.where(reverse, -state.ship_dir, state.ship_dir)
        new_vx = jnp.clip(
            state.ship_vx
            + jnp.where(
                thrust, jnp.float32(0.3) * new_dir.astype(jnp.float32), jnp.float32(0.0)
            ),
            -_PLAYER_SPEED_MAX,
            _PLAYER_SPEED_MAX,
        )
        # Friction
        new_vx = new_vx * jnp.float32(0.95)

        # Vertical movement
        up = action == jnp.int32(2)
        dn = action == jnp.int32(3)
        new_sy = jnp.clip(
            state.ship_y
            + jnp.where(
                up, jnp.float32(-2.0), jnp.where(dn, jnp.float32(2.0), jnp.float32(0.0))
            ),
            jnp.float32(10.0),
            jnp.float32(float(_GROUND_Y - 20)),
        )
        new_sx = state.ship_x + new_vx

        # Fire bullet
        fire = (
            (action == jnp.int32(1)) | (action == jnp.int32(8))
        ) & ~state.bullet_active
        new_bx = jnp.where(
            fire,
            new_sx + jnp.float32(10.0) * new_dir.astype(jnp.float32),
            state.bullet_x,
        )
        new_by = jnp.where(fire, new_sy, state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_bx = jnp.where(
            new_bactive, new_bx + _BULLET_SPEED * new_dir.astype(jnp.float32), new_bx
        )
        new_bactive = new_bactive & (jnp.abs(new_bx - new_sx) < jnp.float32(200.0))

        # Landers move downward toward humanoids
        new_ly = state.lander_y + jnp.where(
            state.lander_active,
            jnp.float32(_LANDER_SPEED),
            jnp.float32(0.0),
        )
        # Landers that reach ground grab humanoids (simplified: deactivate lander)
        lander_at_ground = state.lander_active & (new_ly >= jnp.float32(_GROUND_Y - 15))
        new_lander_active = state.lander_active & ~lander_at_ground
        # Kill nearest humanoid when lander lands
        # (simplified: any lander at ground kills 1 humanoid)
        any_landed = jnp.any(lander_at_ground)
        first_alive_humanoid = jnp.argmax(state.humanoid_alive.astype(jnp.int32))
        new_humanoid_alive = jnp.where(
            any_landed,
            state.humanoid_alive.at[first_alive_humanoid].set(False),
            state.humanoid_alive,
        )

        # Bullet hits lander (world coords; screen shows window around ship_x)
        screen_bx = new_bx  # bullet in world coords
        bullet_hit = (
            new_bactive
            & new_lander_active
            & (jnp.abs(screen_bx - state.lander_x) < jnp.float32(12.0))
            & (jnp.abs(new_by - new_ly) < jnp.float32(12.0))
        )
        step_reward = step_reward + jnp.sum(bullet_hit).astype(
            jnp.float32
        ) * jnp.float32(150.0)
        new_lander_active = new_lander_active & ~bullet_hit
        new_bactive = new_bactive & ~jnp.any(bullet_hit)

        # Wave clear
        wave_clear = ~jnp.any(new_lander_active)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        # Respawn landers on wave clear
        new_lander_active2 = jnp.where(
            wave_clear, jnp.ones(_N_LANDERS, dtype=jnp.bool_), new_lander_active
        )
        new_ly2 = jnp.where(
            wave_clear, jnp.full(_N_LANDERS, 50.0, dtype=jnp.float32), new_ly
        )

        # Episode ends when no humanoids remain
        n_humanoids = jnp.sum(new_humanoid_alive)
        life_lost = n_humanoids <= jnp.int32(0)
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return DefenderState(
            ship_x=new_sx,
            ship_y=new_sy,
            ship_vx=new_vx,
            ship_dir=new_dir,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            lander_x=state.lander_x,
            lander_y=new_ly2,
            lander_active=new_lander_active2,
            humanoid_x=state.humanoid_x,
            humanoid_alive=new_humanoid_alive,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: DefenderState, action: jax.Array) -> DefenderState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : DefenderState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : DefenderState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: DefenderState) -> DefenderState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: DefenderState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : DefenderState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground
        ground_mask = _ROW_IDX >= _GROUND_Y
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Humanoids on ground
        def draw_humanoid(frm, i):
            # Offset to screen: humanoid_x - ship_x + 80
            hx = (state.humanoid_x[i] - state.ship_x + jnp.float32(80.0)).astype(
                jnp.int32
            )
            mask = (
                state.humanoid_alive[i]
                & (_ROW_IDX >= _GROUND_Y - 10)
                & (_ROW_IDX < _GROUND_Y)
                & (_COL_IDX >= hx)
                & (_COL_IDX < hx + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_HUMANOID, frm), None

        frame, _ = jax.lax.scan(draw_humanoid, frame, jnp.arange(_N_HUMANOIDS))

        # Landers
        def draw_lander(frm, i):
            lx = (state.lander_x[i] - state.ship_x + jnp.float32(80.0)).astype(
                jnp.int32
            )
            ly = state.lander_y[i].astype(jnp.int32)
            mask = (
                state.lander_active[i]
                & (_ROW_IDX >= ly)
                & (_ROW_IDX < ly + 10)
                & (_COL_IDX >= lx)
                & (_COL_IDX < lx + 10)
            )
            return jnp.where(mask[:, :, None], _COLOR_LANDER, frm), None

        frame, _ = jax.lax.scan(draw_lander, frame, jnp.arange(_N_LANDERS))

        # Player ship (always at centre x=80)
        sy = state.ship_y.astype(jnp.int32)
        pm = (_ROW_IDX >= sy) & (_ROW_IDX < sy + 8) & (_COL_IDX >= 76) & (_COL_IDX < 92)
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Bullet (offset to screen)
        bx_screen = (state.bullet_x - state.ship_x + jnp.float32(80.0)).astype(
            jnp.int32
        )
        by = state.bullet_y.astype(jnp.int32)
        bm = (
            state.bullet_active
            & (_ROW_IDX >= by)
            & (_ROW_IDX < by + 3)
            & (_COL_IDX >= bx_screen)
            & (_COL_IDX < bx_screen + 4)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Defender action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 3,
            pygame.K_s: 3,
            pygame.K_RIGHT: 4,
            pygame.K_d: 4,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
