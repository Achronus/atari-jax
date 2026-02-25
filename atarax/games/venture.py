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

"""Venture — JAX-native game implementation.

Explore a multi-room dungeon, collecting treasures while fighting monsters.
A large "Hallmonster" pursues you in the hall; smaller monsters guard
each room's treasure.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (shoot arrow)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Treasure collected — +100 to +2500
    Monster shot       — +100
    Episode ends when all lives are lost; lives: 4.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_MONSTERS: int = 3
_N_TREASURES: int = 3
_PLAYER_SPEED: float = 2.0
_ARROW_SPEED: float = 5.0
_MONSTER_SPEED: float = 0.8
_HALLMONSTER_SPEED: float = 0.6

_HALL_Y: int = 105  # y of main corridor
_ROOM_DOOR_X: int = 80  # x of room entrance (simplified: one room)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_HALL = jnp.array([50, 50, 80], dtype=jnp.uint8)
_COLOR_ROOM = jnp.array([40, 60, 40], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_MONSTER = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_HALLMONSTER = jnp.array([255, 100, 0], dtype=jnp.uint8)
_COLOR_TREASURE = jnp.array([255, 215, 0], dtype=jnp.uint8)
_COLOR_ARROW = jnp.array([200, 200, 200], dtype=jnp.uint8)


@chex.dataclass
class VentureState(AtariState):
    """
    Complete Venture game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    in_room : jax.Array
        bool — Player is inside a room.
    arrow_x : jax.Array
        float32 — Arrow x.
    arrow_y : jax.Array
        float32 — Arrow y.
    arrow_dx : jax.Array
        float32 — Arrow x velocity.
    arrow_dy : jax.Array
        float32 — Arrow y velocity.
    arrow_active : jax.Array
        bool — Arrow in flight.
    monster_x : jax.Array
        float32[3] — Room monster x.
    monster_y : jax.Array
        float32[3] — Room monster y.
    monster_active : jax.Array
        bool[3] — Room monster alive.
    treasure_x : jax.Array
        float32[3] — Treasure x.
    treasure_y : jax.Array
        float32[3] — Treasure y.
    treasure_active : jax.Array
        bool[3] — Treasure not collected.
    hallmonster_x : jax.Array
        float32 — Hallmonster x.
    room : jax.Array
        int32 — Current room index.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    in_room: jax.Array
    arrow_x: jax.Array
    arrow_y: jax.Array
    arrow_dx: jax.Array
    arrow_dy: jax.Array
    arrow_active: jax.Array
    monster_x: jax.Array
    monster_y: jax.Array
    monster_active: jax.Array
    treasure_x: jax.Array
    treasure_y: jax.Array
    treasure_active: jax.Array
    hallmonster_x: jax.Array
    room: jax.Array
    key: jax.Array


class Venture(AtariEnv):
    """
    Venture implemented as a pure JAX function suite.

    Collect dungeon treasures; avoid monsters.  Lives: 4.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> VentureState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : VentureState
            Player in hall, hall monster at left, room monsters and treasures placed.
        """
        return VentureState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(float(_HALL_Y)),
            in_room=jnp.bool_(False),
            arrow_x=jnp.float32(80.0),
            arrow_y=jnp.float32(float(_HALL_Y)),
            arrow_dx=jnp.float32(0.0),
            arrow_dy=jnp.float32(0.0),
            arrow_active=jnp.bool_(False),
            monster_x=jnp.array([40.0, 80.0, 120.0], dtype=jnp.float32),
            monster_y=jnp.array([60.0, 80.0, 60.0], dtype=jnp.float32),
            monster_active=jnp.ones(_N_MONSTERS, dtype=jnp.bool_),
            treasure_x=jnp.array([40.0, 80.0, 120.0], dtype=jnp.float32),
            treasure_y=jnp.array([50.0, 70.0, 50.0], dtype=jnp.float32),
            treasure_active=jnp.ones(_N_TREASURES, dtype=jnp.bool_),
            hallmonster_x=jnp.float32(-20.0),
            room=jnp.int32(0),
            lives=jnp.int32(4),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: VentureState, action: jax.Array) -> VentureState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : VentureState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : VentureState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        dy = jnp.where(
            action == 2, -_PLAYER_SPEED, jnp.where(action == 4, _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(state.player_x + dx, jnp.float32(5.0), jnp.float32(150.0))
        new_py = jnp.clip(state.player_y + dy, jnp.float32(30.0), jnp.float32(180.0))

        # Enter/exit room
        entering_room = (
            ~state.in_room
            & (jnp.abs(new_px - jnp.float32(_ROOM_DOOR_X)) < jnp.float32(10.0))
            & (new_py < jnp.float32(float(_HALL_Y) - 10.0))
        )
        in_room = state.in_room | entering_room
        exiting_room = in_room & (new_py >= jnp.float32(float(_HALL_Y) - 5.0))
        in_room = in_room & ~exiting_room

        # Fire arrow
        fire = (action == jnp.int32(1)) & ~state.arrow_active
        new_adx = jnp.where(fire, dx if True else jnp.float32(0.0), state.arrow_dx)
        new_ady = jnp.where(fire, dy if True else jnp.float32(0.0), state.arrow_dy)
        # Default: fire right
        fire_dx = jnp.where(
            dx != 0,
            dx / jnp.maximum(jnp.abs(dx), jnp.float32(0.01)) * _ARROW_SPEED,
            jnp.float32(_ARROW_SPEED),
        )
        fire_dy = jnp.where(
            dy != 0,
            dy / jnp.maximum(jnp.abs(dy), jnp.float32(0.01)) * _ARROW_SPEED,
            jnp.float32(0.0),
        )
        new_adx = jnp.where(fire, fire_dx, state.arrow_dx)
        new_ady = jnp.where(fire, fire_dy, state.arrow_dy)
        new_ax = jnp.where(fire, new_px, state.arrow_x)
        new_ay = jnp.where(fire, new_py, state.arrow_y)
        new_aactive = state.arrow_active | fire
        new_ax = jnp.where(new_aactive, new_ax + new_adx, new_ax)
        new_ay = jnp.where(new_aactive, new_ay + new_ady, new_ay)
        new_aactive = (
            new_aactive & (new_ax > 5) & (new_ax < 155) & (new_ay > 30) & (new_ay < 185)
        )

        # Monsters move toward player when in room
        mdx = jnp.clip(
            (new_px - state.monster_x) * 0.03, -_MONSTER_SPEED, _MONSTER_SPEED
        )
        mdy = jnp.clip(
            (new_py - state.monster_y) * 0.03, -_MONSTER_SPEED, _MONSTER_SPEED
        )
        new_mx = jnp.where(
            state.monster_active & in_room, state.monster_x + mdx, state.monster_x
        )
        new_my = jnp.where(
            state.monster_active & in_room, state.monster_y + mdy, state.monster_y
        )

        # Arrow hits monster
        a_hits_m = (
            new_aactive
            & state.monster_active
            & in_room
            & (jnp.abs(new_ax - new_mx) < jnp.float32(8.0))
            & (jnp.abs(new_ay - new_my) < jnp.float32(8.0))
        )
        step_reward = step_reward + jnp.sum(a_hits_m).astype(jnp.float32) * jnp.float32(
            100.0
        )
        new_monster_active = state.monster_active & ~a_hits_m
        new_aactive = new_aactive & ~jnp.any(a_hits_m)

        # Treasure collection
        treas_hit = (
            state.treasure_active
            & in_room
            & (jnp.abs(state.treasure_x - new_px) < jnp.float32(10.0))
            & (jnp.abs(state.treasure_y - new_py) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(treas_hit).astype(
            jnp.float32
        ) * jnp.float32(100.0)
        new_treas_active = state.treasure_active & ~treas_hit

        # Monster touches player
        monster_hits_player = (
            state.monster_active
            & in_room
            & (jnp.abs(new_mx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_my - new_py) < jnp.float32(8.0))
        )

        # Hall monster pursues player
        new_hm_x = state.hallmonster_x + jnp.clip(
            (new_px - state.hallmonster_x) * 0.02,
            -_HALLMONSTER_SPEED,
            _HALLMONSTER_SPEED,
        )
        hallmonster_hits = (~in_room) & (jnp.abs(new_hm_x - new_px) < jnp.float32(10.0))

        life_lost = jnp.any(monster_hits_player) | hallmonster_hits
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return VentureState(
            player_x=new_px,
            player_y=new_py,
            in_room=in_room,
            arrow_x=new_ax,
            arrow_y=new_ay,
            arrow_dx=new_adx,
            arrow_dy=new_ady,
            arrow_active=new_aactive,
            monster_x=new_mx,
            monster_y=new_my,
            monster_active=new_monster_active,
            treasure_x=state.treasure_x,
            treasure_y=state.treasure_y,
            treasure_active=new_treas_active,
            hallmonster_x=new_hm_x,
            room=state.room,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: VentureState, action: jax.Array) -> VentureState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : VentureState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : VentureState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: VentureState) -> VentureState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: VentureState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : VentureState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Hall
        hall_mask = (_ROW_IDX >= _HALL_Y - 15) & (_ROW_IDX < _HALL_Y + 15)
        frame = jnp.where(hall_mask[:, :, None], _COLOR_HALL, frame)

        # Room (above hall)
        room_mask = (
            (_ROW_IDX >= 30)
            & (_ROW_IDX < _HALL_Y - 15)
            & (_COL_IDX >= 20)
            & (_COL_IDX < 140)
        )
        frame = jnp.where(room_mask[:, :, None], _COLOR_ROOM, frame)

        # Hall monster
        hm = state.hallmonster_x.astype(jnp.int32)
        hmm = (
            (_ROW_IDX >= _HALL_Y - 12)
            & (_ROW_IDX < _HALL_Y + 12)
            & (_COL_IDX >= hm - 6)
            & (_COL_IDX < hm + 6)
        )
        frame = jnp.where(hmm[:, :, None], _COLOR_HALLMONSTER, frame)

        # Treasures
        def draw_treasure(frm, i):
            tx = state.treasure_x[i].astype(jnp.int32)
            ty = state.treasure_y[i].astype(jnp.int32)
            mask = (
                state.treasure_active[i]
                & (_ROW_IDX >= ty - 4)
                & (_ROW_IDX < ty + 4)
                & (_COL_IDX >= tx - 4)
                & (_COL_IDX < tx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_TREASURE, frm), None

        frame, _ = jax.lax.scan(draw_treasure, frame, jnp.arange(_N_TREASURES))

        # Monsters
        def draw_monster(frm, i):
            mx = state.monster_x[i].astype(jnp.int32)
            my = state.monster_y[i].astype(jnp.int32)
            mask = (
                state.monster_active[i]
                & (_ROW_IDX >= my - 5)
                & (_ROW_IDX < my + 5)
                & (_COL_IDX >= mx - 5)
                & (_COL_IDX < mx + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_MONSTER, frm), None

        frame, _ = jax.lax.scan(draw_monster, frame, jnp.arange(_N_MONSTERS))

        # Arrow
        am = (
            state.arrow_active
            & (_ROW_IDX >= state.arrow_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.arrow_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.arrow_x.astype(jnp.int32) - 3)
            & (_COL_IDX < state.arrow_x.astype(jnp.int32) + 3)
        )
        frame = jnp.where(am[:, :, None], _COLOR_ARROW, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py - 6)
            & (_ROW_IDX < py + 6)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX < px + 5)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Venture action indices.
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
