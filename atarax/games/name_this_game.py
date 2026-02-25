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

"""Name This Game — JAX-native game implementation.

Underwater shooter: a scuba diver collects treasures while an octopus and
a shark descend from above.  Shoot the octopus' tentacles and the shark.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — RIGHT+FIRE
    5 — LEFT+FIRE

Scoring:
    Tentacle shot — +100
    Shark shot    — +200
    Treasure picked up — +50
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_TENTACLES: int = 4
_N_TREASURE: int = 3

_PLAYER_Y: int = 160
_OCTOPUS_Y: int = 40
_SHARK_START_Y: float = 60.0

_PLAYER_SPEED: float = 2.5
_BULLET_SPEED: float = 5.0
_SHARK_SPEED: float = 1.0
_TENTACLE_SPEED: float = 0.8

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 60, 140], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_OCTOPUS = jnp.array([180, 50, 180], dtype=jnp.uint8)
_COLOR_TENTACLE = jnp.array([200, 80, 200], dtype=jnp.uint8)
_COLOR_SHARK = jnp.array([150, 150, 200], dtype=jnp.uint8)
_COLOR_TREASURE = jnp.array([255, 215, 0], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class NameThisGameState(AtariState):
    """
    Complete Name This Game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    tentacle_x : jax.Array
        float32[4] — Tentacle x positions.
    tentacle_y : jax.Array
        float32[4] — Tentacle y positions.
    tentacle_active : jax.Array
        bool[4] — Tentacle alive.
    shark_x : jax.Array
        float32 — Shark x.
    shark_y : jax.Array
        float32 — Shark y.
    shark_dir : jax.Array
        int32 — Shark direction.
    treasure_x : jax.Array
        float32[3] — Treasure x.
    treasure_active : jax.Array
        bool[3] — Treasure available.
    spawn_timer : jax.Array
        int32 — Frames until tentacle respawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    tentacle_x: jax.Array
    tentacle_y: jax.Array
    tentacle_active: jax.Array
    shark_x: jax.Array
    shark_y: jax.Array
    shark_dir: jax.Array
    treasure_x: jax.Array
    treasure_active: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class NameThisGame(AtariEnv):
    """
    Name This Game implemented as a pure JAX function suite.

    Shoot octopus tentacles and the shark.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> NameThisGameState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : NameThisGameState
            Player centred, octopus at top, shark at mid-left.
        """
        tentacle_xs = jnp.array([30.0, 60.0, 100.0, 130.0], dtype=jnp.float32)
        return NameThisGameState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_active=jnp.bool_(False),
            tentacle_x=tentacle_xs,
            tentacle_y=jnp.full(
                _N_TENTACLES, float(_OCTOPUS_Y + 10), dtype=jnp.float32
            ),
            tentacle_active=jnp.ones(_N_TENTACLES, dtype=jnp.bool_),
            shark_x=jnp.float32(10.0),
            shark_y=jnp.float32(_SHARK_START_Y),
            shark_dir=jnp.int32(1),
            treasure_x=jnp.array([40.0, 80.0, 120.0], dtype=jnp.float32),
            treasure_active=jnp.ones(_N_TREASURE, dtype=jnp.bool_),
            spawn_timer=jnp.int32(0),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: NameThisGameState, action: jax.Array
    ) -> NameThisGameState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : NameThisGameState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : NameThisGameState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement
        move_r = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_l = (action == jnp.int32(3)) | (action == jnp.int32(5))
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, jnp.where(move_l, -_PLAYER_SPEED, 0.0)),
            jnp.float32(5.0),
            jnp.float32(147.0),
        )

        # Fire
        fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(4))
            | (action == jnp.int32(5))
        ) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(float(_PLAYER_Y - 8)), state.bullet_y)
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by > jnp.float32(10.0))

        # Tentacles drift downward
        new_ty = state.tentacle_y + jnp.where(
            state.tentacle_active, _TENTACLE_SPEED, 0.0
        )
        tentacle_at_player = (
            state.tentacle_active
            & (jnp.abs(state.tentacle_x - new_px) < 10.0)
            & (new_ty >= float(_PLAYER_Y - 10))
        )
        hit_tentacle = jnp.any(tentacle_at_player)

        # Bullet hits tentacle
        b_hits_t = (
            new_bactive
            & state.tentacle_active
            & (jnp.abs(new_bx - state.tentacle_x) < 10.0)
            & (jnp.abs(new_by - new_ty) < 10.0)
        )
        step_reward = step_reward + jnp.sum(b_hits_t).astype(jnp.float32) * 100.0
        new_tentacle_active = state.tentacle_active & ~b_hits_t
        new_bactive = new_bactive & ~jnp.any(b_hits_t)

        # Respawn tentacles from octopus
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_respawn = new_spawn_timer <= jnp.int32(0)
        free_t = jnp.argmin(new_tentacle_active.astype(jnp.int32))
        new_tentacle_active2 = jnp.where(
            can_respawn, new_tentacle_active.at[free_t].set(True), new_tentacle_active
        )
        new_ty2 = jnp.where(
            can_respawn, new_ty.at[free_t].set(float(_OCTOPUS_Y + 10)), new_ty
        )
        new_spawn_timer = jnp.where(can_respawn, jnp.int32(90), new_spawn_timer)

        # Shark moves horizontally, bounces
        new_shark_x = state.shark_x + state.shark_dir.astype(jnp.float32) * _SHARK_SPEED
        at_edge = (new_shark_x < 5.0) | (new_shark_x > 145.0)
        new_shark_dir = jnp.where(at_edge, -state.shark_dir, state.shark_dir)
        new_shark_x = jnp.clip(new_shark_x, 5.0, 145.0)

        # Bullet hits shark
        b_hits_shark = (
            new_bactive
            & (jnp.abs(new_bx - state.shark_x) < 14.0)
            & (jnp.abs(new_by - state.shark_y) < 8.0)
        )
        step_reward = step_reward + jnp.where(b_hits_shark, 200.0, 0.0)
        new_bactive = new_bactive & ~b_hits_shark

        # Shark hits player
        shark_hits_player = (jnp.abs(state.shark_x - new_px) < 12.0) & (
            jnp.abs(state.shark_y - float(_PLAYER_Y)) < 12.0
        )

        # Treasure pickup
        treasure_hit = state.treasure_active & (
            jnp.abs(state.treasure_x - new_px) < 10.0
        )
        step_reward = step_reward + jnp.sum(treasure_hit).astype(jnp.float32) * 50.0
        new_treasure_active = state.treasure_active & ~treasure_hit

        life_lost = hit_tentacle | shark_hits_player
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return NameThisGameState(
            player_x=new_px,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            tentacle_x=state.tentacle_x,
            tentacle_y=new_ty2,
            tentacle_active=new_tentacle_active2,
            shark_x=new_shark_x,
            shark_y=state.shark_y,
            shark_dir=new_shark_dir,
            treasure_x=state.treasure_x,
            treasure_active=new_treasure_active,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: NameThisGameState, action: jax.Array) -> NameThisGameState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : NameThisGameState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : NameThisGameState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: NameThisGameState) -> NameThisGameState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: NameThisGameState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : NameThisGameState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Octopus body
        om = (
            (_ROW_IDX >= _OCTOPUS_Y - 8)
            & (_ROW_IDX <= _OCTOPUS_Y + 8)
            & (_COL_IDX >= 68)
            & (_COL_IDX <= 92)
        )
        frame = jnp.where(om[:, :, None], _COLOR_OCTOPUS, frame)

        # Tentacles
        def draw_tentacle(frm, i):
            tx = state.tentacle_x[i].astype(jnp.int32)
            ty = state.tentacle_y[i].astype(jnp.int32)
            mask = (
                state.tentacle_active[i]
                & (_ROW_IDX >= ty - 5)
                & (_ROW_IDX < ty + 5)
                & (_COL_IDX >= tx - 4)
                & (_COL_IDX < tx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_TENTACLE, frm), None

        frame, _ = jax.lax.scan(draw_tentacle, frame, jnp.arange(_N_TENTACLES))

        # Shark
        sx = state.shark_x.astype(jnp.int32)
        sy = state.shark_y.astype(jnp.int32)
        sm = (
            (_ROW_IDX >= sy - 5)
            & (_ROW_IDX < sy + 5)
            & (_COL_IDX >= sx)
            & (_COL_IDX < sx + 14)
        )
        frame = jnp.where(sm[:, :, None], _COLOR_SHARK, frame)

        # Treasures
        def draw_treasure(frm, i):
            trx = state.treasure_x[i].astype(jnp.int32)
            mask = (
                state.treasure_active[i]
                & (_ROW_IDX >= _PLAYER_Y - 4)
                & (_ROW_IDX < _PLAYER_Y + 4)
                & (_COL_IDX >= trx - 4)
                & (_COL_IDX < trx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_TREASURE, frm), None

        frame, _ = jax.lax.scan(draw_treasure, frame, jnp.arange(_N_TREASURE))

        # Bullet
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
            Mapping of pygame key constants to Name This Game action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
        }
