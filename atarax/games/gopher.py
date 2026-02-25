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

"""Gopher — JAX-native game implementation.

Protect three carrots from a tunnelling gopher.  The gopher emerges from a
tunnel on the left or right side, then moves toward a carrot and attempts
to steal it.  Shoot the gopher before it reaches the carrots or fill in
the hole it digs to save them.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (shoot)
    2 — UP   (dig filler / move up)
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Gopher shot — +200
    Episode ends when all 3 carrots are stolen (lives = 0).
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_GROUND_Y: int = 150
_CARROT_Y: int = 160
_CARROT_W: int = 8
_CARROT_H: int = 12
_N_CARROTS: int = 3

# Carrot positions (centre x)
_CARROT_XS = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float32)

_PLAYER_Y: int = 130  # player stands on ground
_PLAYER_W: int = 8
_PLAYER_H: int = 12
_PLAYER_SPEED: float = 2.0

_BULLET_W: int = 2
_BULLET_H: int = 6
_BULLET_SPEED: float = 5.0

_GOPHER_W: int = 10
_GOPHER_H: int = 10
_GOPHER_SPEED: float = 1.2
_GOPHER_SPAWN_Y: float = float(_GROUND_Y + 5)

_RESPAWN_DELAY: int = 60  # sub-steps before gopher respawns

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([100, 180, 80], dtype=jnp.uint8)
_COLOR_SOIL = jnp.array([140, 90, 50], dtype=jnp.uint8)
_COLOR_CARROT = jnp.array([255, 120, 0], dtype=jnp.uint8)
_COLOR_CARROT_TOP = jnp.array([0, 160, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_GOPHER = jnp.array([180, 130, 80], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 0], dtype=jnp.uint8)


@chex.dataclass
class GopherState(AtariState):
    """
    Complete Gopher game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x (left edge).
    carrots : jax.Array
        bool[3] — Surviving carrots (= lives).
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    gopher_x : jax.Array
        float32 — Gopher x position.
    gopher_y : jax.Array
        float32 — Gopher y position.
    gopher_active : jax.Array
        bool — Gopher on screen.
    gopher_target : jax.Array
        int32 — Index of carrot the gopher is targeting.
    gopher_from_left : jax.Array
        bool — Gopher entered from the left side.
    respawn_timer : jax.Array
        int32 — Sub-steps until gopher respawns after being shot.
    """

    player_x: jax.Array
    carrots: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    gopher_x: jax.Array
    gopher_y: jax.Array
    gopher_active: jax.Array
    gopher_target: jax.Array
    gopher_from_left: jax.Array
    respawn_timer: jax.Array


class Gopher(AtariEnv):
    """
    Gopher implemented as a pure JAX function suite.

    Protect all three carrots.  Episode ends when all carrots are stolen.
    Lives = number of surviving carrots.
    """

    num_actions: int = 6

    def _reset(self, key: jax.Array) -> GopherState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : GopherState
            All carrots present, player at centre, gopher off-screen.
        """
        return GopherState(
            player_x=jnp.float32(76.0),
            carrots=jnp.ones(_N_CARROTS, dtype=jnp.bool_),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_active=jnp.bool_(False),
            gopher_x=jnp.float32(-20.0),
            gopher_y=jnp.float32(_GOPHER_SPAWN_Y),
            gopher_active=jnp.bool_(False),
            gopher_target=jnp.int32(1),  # centre carrot
            gopher_from_left=jnp.bool_(True),
            respawn_timer=jnp.int32(30),  # initial delay
            lives=jnp.int32(_N_CARROTS),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: GopherState, action: jax.Array) -> GopherState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : GopherState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=DOWN, 5=LEFT.

        Returns
        -------
        new_state : GopherState
            State after one emulated frame.
        """
        key, k_spawn = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement (horizontal only)
        pdx = jnp.where(
            action == jnp.int32(3),
            _PLAYER_SPEED,
            jnp.where(action == jnp.int32(5), -_PLAYER_SPEED, jnp.float32(0.0)),
        )
        new_px = jnp.clip(state.player_x + pdx, jnp.float32(8.0), jnp.float32(144.0))

        # Fire bullet
        fire = (action == jnp.int32(1)) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(_PLAYER_W // 2), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(float(_PLAYER_Y)), state.bullet_y)
        new_bactive = state.bullet_active | fire

        # Bullet moves down toward gopher (gopher is below player)
        new_by = jnp.where(new_bactive, new_by + _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by < jnp.float32(_CARROT_Y + _CARROT_H))

        # Gopher movement toward its target carrot
        target_x = _CARROT_XS[state.gopher_target]
        gdx = jnp.clip((target_x - state.gopher_x) * 0.1, -_GOPHER_SPEED, _GOPHER_SPEED)
        new_gx = jnp.where(state.gopher_active, state.gopher_x + gdx, state.gopher_x)
        new_gy = state.gopher_y

        # Gopher reaches carrot: steal it
        gopher_at_carrot = state.gopher_active & (
            jnp.abs(new_gx - target_x) < jnp.float32(6.0)
        )
        new_carrots = jnp.where(
            gopher_at_carrot,
            state.carrots.at[state.gopher_target].set(False),
            state.carrots,
        )
        # Gopher exits after stealing
        new_gopher_active = state.gopher_active & ~gopher_at_carrot
        new_gx = jnp.where(gopher_at_carrot, jnp.float32(-20.0), new_gx)

        # Bullet hits gopher
        gopher_hit = (
            new_bactive
            & new_gopher_active
            & (jnp.abs(new_bx - new_gx) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_gy) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.where(
            gopher_hit, jnp.float32(200.0), jnp.float32(0.0)
        )
        new_gopher_active = new_gopher_active & ~gopher_hit
        new_bactive = new_bactive & ~gopher_hit
        new_gx = jnp.where(gopher_hit, jnp.float32(-20.0), new_gx)

        # Respawn timer
        new_respawn_timer = jnp.where(
            ~new_gopher_active & (state.respawn_timer > jnp.int32(0)),
            state.respawn_timer - jnp.int32(1),
            state.respawn_timer,
        )
        can_spawn = (
            ~new_gopher_active
            & (new_respawn_timer <= jnp.int32(0))
            & jnp.any(new_carrots)
        )

        # Choose a random alive carrot as target
        rand = jax.random.uniform(k_spawn, (3,))
        alive_scores = jnp.where(new_carrots, rand, jnp.float32(-1.0))
        new_target = jnp.argmax(alive_scores).astype(jnp.int32)

        # Spawn gopher from left or right
        from_left = jax.random.uniform(k_spawn) > jnp.float32(0.5)
        spawn_x = jnp.where(from_left, jnp.float32(0.0), jnp.float32(152.0))

        new_gopher_active = jnp.where(can_spawn, jnp.bool_(True), new_gopher_active)
        new_gx = jnp.where(can_spawn, spawn_x, new_gx)
        new_gy = jnp.where(can_spawn, jnp.float32(_GOPHER_SPAWN_Y), new_gy)
        new_target = jnp.where(can_spawn, new_target, state.gopher_target)
        new_respawn_timer = jnp.where(
            can_spawn, jnp.int32(_RESPAWN_DELAY), new_respawn_timer
        )

        n_carrots = jnp.sum(new_carrots).astype(jnp.int32)
        done = n_carrots <= jnp.int32(0)

        return GopherState(
            player_x=new_px,
            carrots=new_carrots,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            gopher_x=new_gx,
            gopher_y=new_gy,
            gopher_active=new_gopher_active,
            gopher_target=new_target,
            gopher_from_left=from_left,
            respawn_timer=new_respawn_timer,
            lives=n_carrots,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: GopherState, action: jax.Array) -> GopherState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : GopherState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : GopherState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: GopherState) -> GopherState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: GopherState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : GopherState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)
        frame = jnp.where(jnp.ones((210, 160, 1), dtype=jnp.bool_), _COLOR_BG, frame)

        # Soil
        soil = _ROW_IDX >= _GROUND_Y
        frame = jnp.where(soil[:, :, None], _COLOR_SOIL, frame)

        # Carrots
        def draw_carrot(frm, i):
            cx = _CARROT_XS[i]
            alive = state.carrots[i]
            body_mask = (
                alive
                & (_ROW_IDX >= _CARROT_Y)
                & (_ROW_IDX < _CARROT_Y + _CARROT_H)
                & (_COL_IDX >= jnp.int32(cx) - _CARROT_W // 2)
                & (_COL_IDX < jnp.int32(cx) + _CARROT_W // 2)
            )
            top_mask = (
                alive
                & (_ROW_IDX >= _CARROT_Y - 6)
                & (_ROW_IDX < _CARROT_Y)
                & (_COL_IDX >= jnp.int32(cx) - 2)
                & (_COL_IDX < jnp.int32(cx) + 2)
            )
            frm = jnp.where(body_mask[:, :, None], _COLOR_CARROT, frm)
            frm = jnp.where(top_mask[:, :, None], _COLOR_CARROT_TOP, frm)
            return frm, None

        frame, _ = jax.lax.scan(draw_carrot, frame, jnp.arange(_N_CARROTS))

        # Gopher
        gm = (
            state.gopher_active
            & (_ROW_IDX >= jnp.int32(state.gopher_y))
            & (_ROW_IDX < jnp.int32(state.gopher_y) + _GOPHER_H)
            & (_COL_IDX >= jnp.int32(state.gopher_x))
            & (_COL_IDX < jnp.int32(state.gopher_x) + _GOPHER_W)
        )
        frame = jnp.where(gm[:, :, None], _COLOR_GOPHER, frame)

        # Player
        pm = (
            (_ROW_IDX >= _PLAYER_Y)
            & (_ROW_IDX < _PLAYER_Y + _PLAYER_H)
            & (_COL_IDX >= jnp.int32(state.player_x))
            & (_COL_IDX < jnp.int32(state.player_x) + _PLAYER_W)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= jnp.int32(state.bullet_y))
            & (_ROW_IDX < jnp.int32(state.bullet_y) + _BULLET_H)
            & (_COL_IDX >= jnp.int32(state.bullet_x))
            & (_COL_IDX < jnp.int32(state.bullet_x) + _BULLET_W)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Gopher action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
