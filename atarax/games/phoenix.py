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

"""Phoenix — JAX-native game implementation.

Space shooter: waves of birds dive at the player from the top.  Shoot
them before they reach the bottom.  A shield can block alien fire briefly.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — SHIELD
    5 — FIRE+RIGHT

Scoring:
    Bird (row 0-1) shot — +10
    Bird (row 2-3) shot — +20
    Phoenix (boss)      — +200
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ROWS: int = 4
_N_COLS: int = 8
_N_BIRDS: int = _N_ROWS * _N_COLS  # 32

_PLAYER_Y: int = 175
_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 6.0
_BIRD_SPEED_BASE: float = 1.0
_ALIEN_BULLET_SPEED: float = 2.5
_MOVE_INTERVAL: int = 20

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 200, 255], dtype=jnp.uint8)
_COLOR_BIRD_LOW = jnp.array([255, 100, 100], dtype=jnp.uint8)
_COLOR_BIRD_HIGH = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ALIEN_BULLET = jnp.array([255, 80, 0], dtype=jnp.uint8)
_COLOR_SHIELD = jnp.array([0, 200, 100], dtype=jnp.uint8)

_BIRD_SCORES = jnp.array([10, 10, 20, 20], dtype=jnp.int32)


@chex.dataclass
class PhoenixState(AtariState):
    """
    Complete Phoenix game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Player bullet in flight.
    birds : jax.Array
        bool[32] — Alive birds (row-major).
    formation_x : jax.Array
        float32 — Formation left edge x.
    formation_dx : jax.Array
        float32 — Formation scroll direction (+1 or -1).
    formation_y : jax.Array
        float32 — Formation top y.
    abul_x : jax.Array
        float32 — Alien bullet x.
    abul_y : jax.Array
        float32 — Alien bullet y.
    abul_active : jax.Array
        bool — Alien bullet in flight.
    shield_active : jax.Array
        bool — Shield on.
    shield_timer : jax.Array
        int32 — Frames shield remains active.
    move_timer : jax.Array
        int32 — Frames until next formation step.
    fire_timer : jax.Array
        int32 — Frames until alien fires.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    birds: jax.Array
    formation_x: jax.Array
    formation_dx: jax.Array
    formation_y: jax.Array
    abul_x: jax.Array
    abul_y: jax.Array
    abul_active: jax.Array
    shield_active: jax.Array
    shield_timer: jax.Array
    move_timer: jax.Array
    fire_timer: jax.Array
    wave: jax.Array
    key: jax.Array


# Precompute per-bird row and col indices (flat [32])
_BIRD_ROWS = jnp.repeat(jnp.arange(_N_ROWS, dtype=jnp.int32), _N_COLS)
_BIRD_COLS = jnp.tile(jnp.arange(_N_COLS, dtype=jnp.int32), _N_ROWS)
_COL_SPACING: int = 16
_ROW_SPACING: int = 12


class Phoenix(AtariEnv):
    """
    Phoenix implemented as a pure JAX function suite.

    Shoot waves of birds.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> PhoenixState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : PhoenixState
            All birds alive, player at centre, no bullets.
        """
        return PhoenixState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_active=jnp.bool_(False),
            birds=jnp.ones(_N_BIRDS, dtype=jnp.bool_),
            formation_x=jnp.float32(8.0),
            formation_dx=jnp.float32(1.0),
            formation_y=jnp.float32(20.0),
            abul_x=jnp.float32(80.0),
            abul_y=jnp.float32(0.0),
            abul_active=jnp.bool_(False),
            shield_active=jnp.bool_(False),
            shield_timer=jnp.int32(0),
            move_timer=jnp.int32(_MOVE_INTERVAL),
            fire_timer=jnp.int32(90),
            wave=jnp.int32(1),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: PhoenixState, action: jax.Array) -> PhoenixState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : PhoenixState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : PhoenixState
            State after one emulated frame.
        """
        key, k_shooter = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        move_r = (action == jnp.int32(2)) | (action == jnp.int32(5))
        move_l = action == jnp.int32(3)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, jnp.where(move_l, -_PLAYER_SPEED, 0.0)),
            jnp.float32(5.0),
            jnp.float32(147.0),
        )

        # Shield
        activate_shield = action == jnp.int32(4)
        new_shield_timer = jnp.where(
            activate_shield & ~state.shield_active,
            jnp.int32(30),
            state.shield_timer - jnp.int32(1),
        )
        new_shield_timer = jnp.maximum(new_shield_timer, jnp.int32(0))
        new_shield_active = new_shield_timer > jnp.int32(0)

        # Fire bullet
        fire = (
            (action == jnp.int32(1)) | (action == jnp.int32(5))
        ) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(float(_PLAYER_Y - 8)), state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(10.0))

        # Move formation
        new_move_timer = state.move_timer - jnp.int32(1)
        step_formation = new_move_timer <= jnp.int32(0)
        new_fx = jnp.where(
            step_formation,
            state.formation_x + state.formation_dx * _BIRD_SPEED_BASE,
            state.formation_x,
        )
        at_edge_r = new_fx + _N_COLS * _COL_SPACING >= 152
        at_edge_l = new_fx <= 5
        new_fdx = jnp.where(
            at_edge_r | at_edge_l, -state.formation_dx, state.formation_dx
        )
        new_move_timer = jnp.where(
            step_formation, jnp.int32(_MOVE_INTERVAL), new_move_timer
        )

        # Bird positions (world coords)
        bird_xs = new_fx + _BIRD_COLS.astype(jnp.float32) * jnp.float32(_COL_SPACING)
        bird_ys = state.formation_y + _BIRD_ROWS.astype(jnp.float32) * jnp.float32(
            _ROW_SPACING
        )

        # Player bullet hits bird
        bullet_hit = (
            new_bactive
            & state.birds
            & (jnp.abs(new_bx - bird_xs) < jnp.float32(7.0))
            & (jnp.abs(new_by - bird_ys) < jnp.float32(6.0))
        )
        bird_scores = jnp.where(
            bullet_hit, _BIRD_SCORES[_BIRD_ROWS], jnp.zeros(_N_BIRDS, dtype=jnp.int32)
        )
        step_reward = step_reward + jnp.sum(bird_scores).astype(jnp.float32)
        new_birds = state.birds & ~bullet_hit
        new_bactive = new_bactive & ~jnp.any(bullet_hit)

        # Bird reaches player level → life lost
        bird_reaches_player = state.birds & (bird_ys >= float(_PLAYER_Y - 8))
        hit_by_bird = jnp.any(bird_reaches_player)

        # Alien bullet
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & jnp.any(new_birds)
        # Pick random alive bird to fire from
        rand = jax.random.uniform(k_shooter, (_N_BIRDS,))
        alive_scores_f = jnp.where(new_birds, rand, jnp.float32(-1.0))
        shooter_idx = jnp.argmax(alive_scores_f)
        new_abx = jnp.where(can_fire, bird_xs[shooter_idx], state.abul_x)
        new_aby = jnp.where(can_fire, bird_ys[shooter_idx], state.abul_y)
        new_abactive = jnp.where(can_fire, jnp.bool_(True), state.abul_active)
        new_fire_timer = jnp.where(can_fire, jnp.int32(60), new_fire_timer)

        new_aby = jnp.where(new_abactive, new_aby + _ALIEN_BULLET_SPEED, new_aby)
        new_abactive = new_abactive & (new_aby < jnp.float32(float(_PLAYER_Y + 10)))

        # Alien bullet hits player (shield blocks)
        alien_hits_player = (
            new_abactive
            & ~new_shield_active
            & (jnp.abs(new_abx - new_px) < jnp.float32(8.0))
            & (new_aby >= jnp.float32(float(_PLAYER_Y - 8)))
        )
        # Shield blocks bullet
        new_abactive = new_abactive & ~(alien_hits_player | new_shield_active)

        # Wave clear
        wave_clear = ~jnp.any(new_birds)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        new_birds2 = jnp.where(
            wave_clear, jnp.ones(_N_BIRDS, dtype=jnp.bool_), new_birds
        )
        new_fy = jnp.where(wave_clear, jnp.float32(20.0), state.formation_y)

        life_lost = hit_by_bird | alien_hits_player
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return PhoenixState(
            player_x=new_px,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            birds=new_birds2,
            formation_x=new_fx,
            formation_dx=new_fdx,
            formation_y=new_fy,
            abul_x=new_abx,
            abul_y=new_aby,
            abul_active=new_abactive,
            shield_active=new_shield_active,
            shield_timer=new_shield_timer,
            move_timer=new_move_timer,
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

    def _step(self, state: PhoenixState, action: jax.Array) -> PhoenixState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : PhoenixState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : PhoenixState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: PhoenixState) -> PhoenixState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: PhoenixState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : PhoenixState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        bird_xs = (
            state.formation_x + _BIRD_COLS.astype(jnp.float32) * _COL_SPACING
        ).astype(jnp.int32)
        bird_ys = (
            state.formation_y + _BIRD_ROWS.astype(jnp.float32) * _ROW_SPACING
        ).astype(jnp.int32)
        bird_colors = jnp.where(
            (_BIRD_ROWS < 2)[:, None], _COLOR_BIRD_HIGH, _COLOR_BIRD_LOW
        )

        def draw_bird(frm, i):
            bx = bird_xs[i]
            by = bird_ys[i]
            mask = (
                state.birds[i]
                & (_ROW_IDX >= by)
                & (_ROW_IDX < by + 8)
                & (_COL_IDX >= bx)
                & (_COL_IDX < bx + 8)
            )
            return jnp.where(mask[:, :, None], bird_colors[i], frm), None

        frame, _ = jax.lax.scan(draw_bird, frame, jnp.arange(_N_BIRDS))

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

        # Shield
        px = state.player_x.astype(jnp.int32)
        shield_mask = (
            state.shield_active
            & (_ROW_IDX >= _PLAYER_Y - 10)
            & (_ROW_IDX < _PLAYER_Y + 10)
            & (_COL_IDX >= px - 4)
            & (_COL_IDX < px + 12)
        )
        frame = jnp.where(shield_mask[:, :, None], _COLOR_SHIELD, frame)

        # Player
        pm = (
            (_ROW_IDX >= _PLAYER_Y - 6)
            & (_ROW_IDX < _PLAYER_Y + 6)
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
            Mapping of pygame key constants to Phoenix action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
            pygame.K_LSHIFT: 4,
            pygame.K_RSHIFT: 4,
        }
