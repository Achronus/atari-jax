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

"""Asteroids — JAX-native SDF game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Playfield  : 160 × 210 px, all edges wrap toroidally
    Ship start : x = 80,  y = 105,  angle = -pi/2 (pointing up)
    Rock radii : large = 12 px,  medium = 7 px,  small = 4 px

Rock pool layout (28 slots):
    Slots  0-3  : large rocks  (size = 2)
    Slots  4-11 : medium rocks (size = 1)  — children of large slots 0-3
    Slots 12-27 : small rocks  (size = 0)  — children of medium slots 4-11

Action space (14 actions):
    0 — NOOP           1 — FIRE
    2 — UP (thrust)    3 — RIGHT (CW)   4 — LEFT (CCW)
    5 — DOWN (warp)    6 — UP+FIRE      7 — RIGHT+FIRE
    8 — LEFT+FIRE      9 — DOWN+FIRE   10 — UP+RIGHT
   11 — UP+LEFT       12 — UP+RIGHT+FIRE  13 — UP+LEFT+FIRE
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.free_2d_shooter import Free2DShooterGame, Free2DShooterState
from atarax.env.hud import render_life_pips, render_score
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    render_circle_pool,
    render_variable_circle_pool,
    sdf_capsule,
    sdf_ship_triangle,
)
from atarax.game import AtaraxParams

# ── World
_W: float = 160.0
_H: float = 210.0

# ── Rock geometry
# Indexed by size value: _ROCK_RADIUS[size], _ROCK_SCORES[size]
# size=0 (small), size=1 (medium), size=2 (large)
_ROCK_RADIUS = (4.0, 7.0, 12.0)
_ROCK_SCORES = (100, 50, 20)

# Pool structure — 28 slots: 4 large + 8 medium + 16 small.
_N_ROCKS: int = 28
_N_FRAGMENTABLE: int = 12  # slots 0-11 can spawn children

# Child slot indices for fragmentation (used in unrolled Python for-loop).
_CHILD_LO = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
_CHILD_HI = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]

# Sizes for each slot (static assignment).
_SLOT_SIZE = [2] * 4 + [1] * 8 + [0] * 16

# Initial large-rock positions (one per quadrant, away from ship centre).
_INIT_ROCK_XY = [
    (30.0, 35.0),
    (130.0, 35.0),
    (30.0, 175.0),
    (130.0, 175.0),
]

# ── Physics
_SHIP_SIZE: float = 8.0  # sdf_ship_triangle size parameter
_SHIP_COLLISION_R: float = 5.0
_BULLET_RADIUS: float = 1.5
_MAX_BULLETS: int = 4

# ── Colours (float32 RGB in [0, 1])
_COL_BG = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
_COL_ROCK = jnp.array([0.549, 0.549, 0.549], dtype=jnp.float32)  # grey
_COL_SHIP = jnp.array([0.361, 0.729, 0.361], dtype=jnp.float32)  # green
_COL_BULLET = jnp.array([1.0, 1.0, 0.392], dtype=jnp.float32)  # yellow
_COL_FLAME = jnp.array([1.0, 0.6, 0.1], dtype=jnp.float32)  # orange

# Trail colours at 33 % brightness.
_COL_ROCK_TRAIL = _COL_ROCK / jnp.float32(3.0)
_COL_SHIP_TRAIL = _COL_SHIP / jnp.float32(3.0)


@chex.dataclass
class AsteroidsParams(AtaraxParams):
    """
    Static configuration for Asteroids.

    Parameters
    ----------
    max_steps : int
        Maximum agent steps per episode.
    turn_speed : float
        Rotation in radians per emulated frame.
    thrust : float
        Speed added per thrust frame.
    drag : float
        Fractional velocity reduction per frame.
    max_speed : float
        Maximum ship speed in pixels per frame.
    bullet_speed : float
        Bullet speed in pixels per frame.
    bullet_lifetime : float
        Frames until a bullet expires.
    bullet_cooldown : int
        Minimum frames between shots.
    tip_offset : float
        Spawn distance ahead of ship centre.
    invincible_frames : int
        Post-respawn invincibility duration.
    respawn_frames : int
        Frames between death and respawn.
    warp_invincible : int
        Invincibility frames granted by hyperspace warp.
    num_lives : int
        Lives at episode start.
    rock_speed : float
        Initial speed multiplier for rock velocities.
    """

    max_steps: int = 27000
    turn_speed: float = 0.10
    thrust: float = 0.25
    drag: float = 0.01
    max_speed: float = 6.0
    bullet_speed: float = 7.0
    bullet_lifetime: float = 28.0
    bullet_cooldown: int = 6
    tip_offset: float = 9.0
    invincible_frames: int = 90
    respawn_frames: int = 60
    warp_invincible: int = 45
    num_lives: int = 3
    rock_speed: float = 1.0


@chex.dataclass
class AsteroidsState(Free2DShooterState):
    """
    Asteroids game state.

    Extends `Free2DShooterState` with rock and bullet pools.

    Inherited from `Free2DShooterState`:
        `ship_x`, `ship_y`, `ship_vx`, `ship_vy`, `ship_angle`,
        `ship_alive`, `fire_cooldown`, `thrust_active`, `invincible`.

    Inherited from `AtariState`:
        `reward`, `done`, `step`, `episode_step`, `lives`, `score`, `level`, `key`.

    Parameters
    ----------
    rocks : chex.Array
        (28, 6) float32 — rock pool `[x, y, vx, vy, size, active]`.
        `size`: 2 = large, 1 = medium, 0 = small. `active`: 1.0 = alive.
    bullets : chex.Array
        (4, 6) float32 — bullet pool `[x, y, vx, vy, lifetime, active]`.
    respawn_timer : chex.Array
        int32 scalar — countdown frames until respawn. 0 when alive.
    wave : chex.Array
        int32 scalar — current wave number (starts at 0).
    """

    rocks: chex.Array
    bullets: chex.Array
    respawn_timer: chex.Array
    wave: chex.Array


class Asteroids(Free2DShooterGame):
    """
    Asteroids implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    Ship physics are delegated to `Free2DShooterGame` helpers.
    Rock fragmentation uses a fixed tree-structured slot pool.
    """

    num_actions: int = 14
    game_id: ClassVar[str] = "asteroids"

    def _init_rocks(self, rng: chex.PRNGKey, n_extra: int = 0) -> chex.Array:
        """
        Build the initial rock pool for one wave.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for velocity randomisation.
        n_extra : int
            Extra large rocks beyond the base 4 (capped at 4 for pool safety).

        Returns
        -------
        rocks : chex.Array
            (28, 6) float32 initialised rock pool.
        """
        n_large = min(4 + n_extra, 4)
        rocks = jnp.zeros((_N_ROCKS, 6), dtype=jnp.float32)

        for k in range(n_large):
            rng, sub = jax.random.split(rng)
            angle = jax.random.uniform(sub, minval=0.0, maxval=2.0 * jnp.pi)
            spd = jax.random.uniform(jax.random.fold_in(sub, 1), minval=0.4, maxval=0.9)
            rx, ry = _INIT_ROCK_XY[k]
            row = jnp.stack(
                [
                    jnp.float32(rx),
                    jnp.float32(ry),
                    jnp.cos(angle) * spd,
                    jnp.sin(angle) * spd,
                    jnp.float32(2.0),
                    jnp.float32(1.0),
                ]
            )
            rocks = rocks.at[k].set(row)

        return rocks

    def _reset(self, rng: chex.PRNGKey) -> AsteroidsState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : AsteroidsState
            Ship at centre, 4 large rocks spawned, 3 lives, 4 empty bullet slots.
        """
        rng, rock_rng = jax.random.split(rng)
        rocks = self._init_rocks(rock_rng)

        return AsteroidsState(
            # Free2DShooterState fields
            ship_x=jnp.float32(80.0),
            ship_y=jnp.float32(105.0),
            ship_vx=jnp.float32(0.0),
            ship_vy=jnp.float32(0.0),
            ship_angle=jnp.float32(-jnp.pi / 2.0),
            ship_alive=jnp.bool_(True),
            fire_cooldown=jnp.int32(0),
            thrust_active=jnp.bool_(False),
            invincible=jnp.int32(0),
            # AsteroidsState fields
            rocks=rocks,
            bullets=jnp.zeros((_MAX_BULLETS, 6), dtype=jnp.float32),
            respawn_timer=jnp.int32(0),
            wave=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(3),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=rng,
        )

    def _step_physics(
        self,
        state: AsteroidsState,
        action: chex.Array,
        params: AsteroidsParams,
        rng: chex.PRNGKey,
    ) -> AsteroidsState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : AsteroidsState
            Current game state.
        action : chex.Array
            int32 — action index (0–13).
        params : AsteroidsParams
            Static environment parameters.
        rng : chex.PRNGKey
            PRNG key for randomness (warp teleport, wave spawn).

        Returns
        -------
        new_state : AsteroidsState
            State after one emulated frame.
        """
        rng, warp_rng, wave_rng = jax.random.split(rng, 3)

        # ── 1. Decode action flags
        fire = (
            (action == 1)
            | (action == 6)
            | (action == 7)
            | (action == 8)
            | (action == 9)
            | (action == 12)
            | (action == 13)
        )
        thrust = (
            (action == 2)
            | (action == 6)
            | (action == 10)
            | (action == 11)
            | (action == 12)
            | (action == 13)
        )
        rotate_cw = (action == 3) | (action == 7) | (action == 10) | (action == 12)
        rotate_ccw = (action == 4) | (action == 8) | (action == 11) | (action == 13)
        hyperspace = (action == 5) | (action == 9)

        # ── 2. Rotation
        angle = state.ship_angle + jnp.where(
            rotate_cw,
            jnp.float32(params.turn_speed),
            jnp.where(rotate_ccw, jnp.float32(-params.turn_speed), jnp.float32(0.0)),
        )

        # ── 3. Thrust (only when ship is alive)
        t = jnp.where(
            thrust & state.ship_alive, jnp.float32(params.thrust), jnp.float32(0.0)
        )
        ship_vx, ship_vy = self._apply_thrust(
            state.ship_vx,
            state.ship_vy,
            angle,
            t,
            jnp.float32(params.drag),
            jnp.float32(params.max_speed),
        )

        # ── 4. Move ship + wrap
        ship_x = jnp.where(state.ship_alive, state.ship_x + ship_vx, state.ship_x)
        ship_y = jnp.where(state.ship_alive, state.ship_y + ship_vy, state.ship_y)
        ship_x, ship_y = self._wrap_torus(
            ship_x, ship_y, jnp.float32(_W), jnp.float32(_H)
        )

        # ── 5. Thrust flame flag
        thrust_active = thrust & state.ship_alive

        # ── 6. Hyperspace warp
        warp_x = jax.random.uniform(
            warp_rng, minval=jnp.float32(15.0), maxval=jnp.float32(145.0)
        )
        warp_y = jax.random.uniform(
            jax.random.fold_in(warp_rng, 1),
            minval=jnp.float32(15.0),
            maxval=jnp.float32(195.0),
        )
        do_warp = hyperspace & state.ship_alive
        ship_x = jnp.where(do_warp, warp_x, ship_x)
        ship_y = jnp.where(do_warp, warp_y, ship_y)
        ship_vx = jnp.where(do_warp, jnp.float32(0.0), ship_vx)
        ship_vy = jnp.where(do_warp, jnp.float32(0.0), ship_vy)
        invincible = jnp.where(
            do_warp,
            jnp.int32(params.warp_invincible),
            jnp.maximum(state.invincible - jnp.int32(1), jnp.int32(0)),
        )

        # ── 7. Respawn countdown
        respawn_timer = jnp.maximum(state.respawn_timer - jnp.int32(1), jnp.int32(0))
        respawning = (
            (respawn_timer == jnp.int32(0))
            & ~state.ship_alive
            & (state.lives > jnp.int32(0))
        )
        ship_alive = state.ship_alive | respawning
        ship_x = jnp.where(respawning, jnp.float32(80.0), ship_x)
        ship_y = jnp.where(respawning, jnp.float32(105.0), ship_y)
        ship_vx = jnp.where(respawning, jnp.float32(0.0), ship_vx)
        ship_vy = jnp.where(respawning, jnp.float32(0.0), ship_vy)
        invincible = jnp.where(
            respawning, jnp.int32(params.invincible_frames), invincible
        )

        # ── 8. Fire bullet (gated by fire action and ship alive)
        can_fire = (state.fire_cooldown <= jnp.int32(0)) & fire & ship_alive
        fire_slot = jnp.argmin(state.bullets[:, 5])
        fire_slot_free = state.bullets[fire_slot, 5] < jnp.float32(0.5)
        do_fire = can_fire & fire_slot_free
        spawn_x = ship_x + jnp.cos(angle) * jnp.float32(params.tip_offset)
        spawn_y = ship_y + jnp.sin(angle) * jnp.float32(params.tip_offset)
        new_bullet = jnp.stack(
            [
                spawn_x,
                spawn_y,
                jnp.cos(angle) * jnp.float32(params.bullet_speed),
                jnp.sin(angle) * jnp.float32(params.bullet_speed),
                jnp.float32(params.bullet_lifetime),
                jnp.float32(1.0),
            ]
        )
        bullets = jnp.where(
            do_fire, state.bullets.at[fire_slot].set(new_bullet), state.bullets
        )
        fire_cooldown = jnp.where(
            do_fire,
            jnp.int32(params.bullet_cooldown),
            jnp.maximum(state.fire_cooldown - jnp.int32(1), jnp.int32(0)),
        )

        # ── 9. Move bullets (toroidal wrap, lifetime decrement)
        bullets = self._move_bullets(bullets, jnp.float32(_W), jnp.float32(_H))

        # ── 10. Move rocks + wrap
        rock_active = state.rocks[:, 5]
        new_rx = (state.rocks[:, 0] + state.rocks[:, 2] * rock_active) % jnp.float32(_W)
        new_ry = (state.rocks[:, 1] + state.rocks[:, 3] * rock_active) % jnp.float32(_H)
        rocks = state.rocks.at[:, 0].set(new_rx).at[:, 1].set(new_ry)

        # ── 11. Bullet-rock collision
        bx = bullets[:, 0][:, None]
        by = bullets[:, 1][:, None]
        ba = bullets[:, 5][:, None]
        rx = rocks[:, 0][None, :]
        ry = rocks[:, 1][None, :]
        ra = rocks[:, 5][None, :]

        radii = jnp.where(
            rocks[:, 4] == jnp.float32(2.0),
            jnp.float32(_ROCK_RADIUS[2]),
            jnp.where(
                rocks[:, 4] == jnp.float32(1.0),
                jnp.float32(_ROCK_RADIUS[1]),
                jnp.float32(_ROCK_RADIUS[0]),
            ),
        )

        dist_sq = (bx - rx) ** 2 + (by - ry) ** 2
        hit_matrix = (
            (dist_sq < (radii[None, :] ** 2))
            & (ba > jnp.float32(0.5))
            & (ra > jnp.float32(0.5))
        )
        rock_hit = jnp.any(hit_matrix, axis=0)
        bullet_hit = jnp.any(hit_matrix, axis=1)

        # Deactivate hit bullets
        bullets = bullets.at[:, 5].set(
            bullets[:, 5] * (~bullet_hit).astype(jnp.float32)
        )

        # Score per hit rock
        rock_score = jnp.where(
            rocks[:, 4] == jnp.float32(2.0),
            jnp.int32(_ROCK_SCORES[2]),
            jnp.where(
                rocks[:, 4] == jnp.float32(1.0),
                jnp.int32(_ROCK_SCORES[1]),
                jnp.int32(_ROCK_SCORES[0]),
            ),
        )
        delta_score = jnp.sum(
            rock_hit.astype(jnp.int32) * rocks[:, 5].astype(jnp.int32) * rock_score
        )

        # Deactivate hit rocks
        rocks = rocks.at[:, 5].set(rocks[:, 5] * (~rock_hit).astype(jnp.float32))

        # Fragment spawning — unrolled Python loop over fragmentable slots (0-11)
        cos_lo = jnp.float32(jnp.cos(jnp.pi / 5.0))
        sin_lo = jnp.float32(jnp.sin(jnp.pi / 5.0))
        cos_hi = jnp.float32(jnp.cos(-jnp.pi / 5.0))
        sin_hi = jnp.float32(jnp.sin(-jnp.pi / 5.0))

        for i, (cl, ch) in enumerate(zip(_CHILD_LO, _CHILD_HI)):
            should_spawn = rock_hit[i] & (rocks[i, 4] > jnp.float32(0.0))
            pvx = rocks[i, 2]
            pvy = rocks[i, 3]
            pspd = jnp.sqrt(pvx**2 + pvy**2) * jnp.float32(1.4) + jnp.float32(0.4)
            nx = pvx / jnp.maximum(jnp.sqrt(pvx**2 + pvy**2), jnp.float32(1e-6))
            ny = pvy / jnp.maximum(jnp.sqrt(pvx**2 + pvy**2), jnp.float32(1e-6))
            flo_vx = (cos_lo * nx - sin_lo * ny) * pspd
            flo_vy = (sin_lo * nx + cos_lo * ny) * pspd
            fhi_vx = (cos_hi * nx - sin_hi * ny) * pspd
            fhi_vy = (sin_hi * nx + cos_hi * ny) * pspd
            frag_size = rocks[i, 4] - jnp.float32(1.0)
            frag_lo = jnp.stack(
                [rocks[i, 0], rocks[i, 1], flo_vx, flo_vy, frag_size, jnp.float32(1.0)]
            )
            frag_hi = jnp.stack(
                [rocks[i, 0], rocks[i, 1], fhi_vx, fhi_vy, frag_size, jnp.float32(1.0)]
            )
            rocks = rocks.at[cl].set(jnp.where(should_spawn, frag_lo, rocks[cl]))
            rocks = rocks.at[ch].set(jnp.where(should_spawn, frag_hi, rocks[ch]))

        # ── 12. Ship-rock collision
        sdist_sq = (rocks[:, 0] - ship_x) ** 2 + (rocks[:, 1] - ship_y) ** 2
        ship_col_r = jnp.float32(_SHIP_COLLISION_R)
        ship_hit = jnp.any(
            (sdist_sq < (radii + ship_col_r) ** 2) & (rocks[:, 5] > jnp.float32(0.5))
        )
        ship_hit = ship_hit & ship_alive & (invincible <= jnp.int32(0))
        new_lives = state.lives - jnp.where(ship_hit, jnp.int32(1), jnp.int32(0))
        ship_alive = ship_alive & ~ship_hit
        respawn_timer = jnp.where(
            ship_hit, jnp.int32(params.respawn_frames), respawn_timer
        )

        # ── 13. Wave clear — spawn next wave when all rocks inactive
        wave_clear = ~jnp.any(rocks[:, 5] > jnp.float32(0.5))
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_level = state.level + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))

        # Build fresh rocks for the new wave
        next_rocks = self._init_rocks(wave_rng)
        rocks = jnp.where(wave_clear, next_rocks, rocks)
        bullets = jnp.where(
            wave_clear, jnp.zeros((_MAX_BULLETS, 6), dtype=jnp.float32), bullets
        )

        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            ship_x=ship_x,
            ship_y=ship_y,
            ship_vx=ship_vx,
            ship_vy=ship_vy,
            ship_angle=angle,
            ship_alive=ship_alive,
            fire_cooldown=fire_cooldown,
            thrust_active=thrust_active,
            invincible=invincible,
            rocks=rocks,
            bullets=bullets,
            respawn_timer=respawn_timer,
            wave=new_wave,
            lives=new_lives,
            score=state.score + delta_score,
            reward=state.reward + delta_score.astype(jnp.float32),
            level=new_level,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: AsteroidsState,
        action: chex.Array,
        params: AsteroidsParams,
    ) -> AsteroidsState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for in-step randomness.
        state : AsteroidsState
            Current game state.
        action : chex.Array
            int32 — Action index.
        params : AsteroidsParams
            Static environment parameters.

        Returns
        -------
        new_state : AsteroidsState
            State after 4 emulated frames with `episode_step` incremented once.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: AsteroidsState) -> AsteroidsState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: AsteroidsState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AsteroidsState
            Current game state.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Rocks
        rock_radii = jnp.where(
            state.rocks[:, 4] == jnp.float32(2.0),
            jnp.float32(_ROCK_RADIUS[2]),
            jnp.where(
                state.rocks[:, 4] == jnp.float32(1.0),
                jnp.float32(_ROCK_RADIUS[1]),
                jnp.float32(_ROCK_RADIUS[0]),
            ),
        )
        rock_render_pool = jnp.stack(
            [state.rocks[:, 0], state.rocks[:, 1], rock_radii, state.rocks[:, 5]],
            axis=1,
        )
        rock_trail_pool = jnp.stack(
            [
                state.rocks[:, 0] - state.rocks[:, 2],
                state.rocks[:, 1] - state.rocks[:, 3],
                rock_radii,
                state.rocks[:, 5],
            ],
            axis=1,
        )
        canvas = paint_layer(
            canvas, render_variable_circle_pool(rock_trail_pool), _COL_ROCK_TRAIL
        )
        canvas = paint_layer(
            canvas, render_variable_circle_pool(rock_render_pool), _COL_ROCK
        )

        # Layer 2 — Bullets
        bullet_render_pool = jnp.stack(
            [state.bullets[:, 0], state.bullets[:, 1], state.bullets[:, 5]], axis=1
        )
        canvas = paint_layer(
            canvas, render_circle_pool(bullet_render_pool, _BULLET_RADIUS), _COL_BULLET
        )

        # Layer 3 — Thrust flame (rendered before ship so ship overwrites it)
        flame_dist = jnp.float32(_SHIP_SIZE * 0.55)
        flame_ax = state.ship_x - jnp.cos(state.ship_angle) * flame_dist
        flame_ay = state.ship_y - jnp.sin(state.ship_angle) * flame_dist
        flame_bx = state.ship_x - jnp.cos(state.ship_angle) * (
            flame_dist + jnp.float32(6.0)
        )
        flame_by = state.ship_y - jnp.sin(state.ship_angle) * (
            flame_dist + jnp.float32(6.0)
        )
        flame_sdf = sdf_capsule(
            flame_ax, flame_ay, flame_bx, flame_by, jnp.float32(2.0)
        )
        flame_visible = state.thrust_active & state.ship_alive
        canvas = paint_layer(
            canvas, (flame_sdf < jnp.float32(0.0)) & flame_visible, _COL_FLAME
        )

        # Layer 4 — Ship trail then ship
        ship_trail_sdf = sdf_ship_triangle(
            state.ship_x - state.ship_vx,
            state.ship_y - state.ship_vy,
            state.ship_angle,
            _SHIP_SIZE,
        )
        canvas = paint_layer(
            canvas,
            (ship_trail_sdf < jnp.float32(0.0)) & state.ship_alive,
            _COL_SHIP_TRAIL,
        )
        ship_sdf = sdf_ship_triangle(
            state.ship_x, state.ship_y, state.ship_angle, _SHIP_SIZE
        )
        canvas = paint_layer(
            canvas, (ship_sdf < jnp.float32(0.0)) & state.ship_alive, _COL_SHIP
        )

        # ── HUD (top 30 px) — ship-triangle life pips + score ──────────────
        canvas = render_life_pips(
            canvas,
            state.lives,
            pip_sdf_fn=lambda cx, cy: sdf_ship_triangle(cx, cy, -jnp.pi / 2, 5.0),
            pip_colour=_COL_SHIP,
        )
        canvas = render_score(canvas, state.score)

        return finalise_rgb(canvas)
