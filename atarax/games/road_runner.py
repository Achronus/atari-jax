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

"""Road Runner — JAX-native game implementation.

Run as fast as possible down the road while Wile E. Coyote chases you.
Eat birdseed for points; dodge trucks, boulders, and the coyote.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (unused / beep)
    2 — RIGHT (run faster)
    3 — LEFT  (slow down)
    4 — UP    (jump)
    5 — DOWN  (duck — no-op in most contexts)

Scoring:
    Birdseed eaten — +100
    Truck dodged (passes off-screen) — no direct score
    Coyote caught — life lost
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_SCREEN_W: int = 160
_SCREEN_H: int = 210

_ROAD_Y: int = 160  # road surface y
_ROAD_TOP: int = 100  # top of visible road area

_PLAYER_W: int = 10
_PLAYER_H: int = 12
_PLAYER_X: int = 30  # player x fixed (road scrolls)
_PLAYER_GROUND_Y: int = _ROAD_Y - _PLAYER_H

_COYOTE_W: int = 10
_COYOTE_H: int = 12

_TRUCK_W: int = 20
_TRUCK_H: int = 14

_SEED_W: int = 6
_SEED_H: int = 6

_N_TRUCKS: int = 3
_N_SEEDS: int = 5

_JUMP_VY: float = -6.0
_GRAVITY: float = 0.5

_SCROLL_SPEED_BASE: float = 2.0

_ROW_IDX = jnp.arange(_SCREEN_H)[:, None]
_COL_IDX = jnp.arange(_SCREEN_W)[None, :]

_COLOR_BG = jnp.array([140, 200, 100], dtype=jnp.uint8)  # desert
_COLOR_ROAD = jnp.array([180, 160, 120], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 180, 220], dtype=jnp.uint8)
_COLOR_COYOTE = jnp.array([180, 140, 80], dtype=jnp.uint8)
_COLOR_TRUCK = jnp.array([200, 80, 40], dtype=jnp.uint8)
_COLOR_SEED = jnp.array([240, 200, 60], dtype=jnp.uint8)


@chex.dataclass
class RoadRunnerState(AtariState):
    """
    Complete Road Runner game state — a JAX pytree.

    Parameters
    ----------
    player_y : jax.Array
        float32 — Player y (top of sprite).
    player_vy : jax.Array
        float32 — Player vertical velocity.
    scroll_speed : jax.Array
        float32 — Road scroll speed.
    coyote_x : jax.Array
        float32 — Coyote x on screen.
    truck_x : jax.Array
        float32[3] — Truck x positions (scroll left).
    truck_active : jax.Array
        bool[3] — Trucks on screen.
    seed_x : jax.Array
        float32[5] — Birdseed x positions.
    seed_active : jax.Array
        bool[5] — Seeds not yet eaten.
    spawn_timer : jax.Array
        int32 — Frames until next truck/seed spawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_y: jax.Array
    player_vy: jax.Array
    scroll_speed: jax.Array
    coyote_x: jax.Array
    truck_x: jax.Array
    truck_active: jax.Array
    seed_x: jax.Array
    seed_active: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class RoadRunner(AtariEnv):
    """
    Road Runner implemented as a pure JAX function suite.

    Run and survive.  Lives: 5.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> RoadRunnerState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : RoadRunnerState
            Player on ground, coyote far behind, 5 lives.
        """
        return RoadRunnerState(
            player_y=jnp.float32(float(_PLAYER_GROUND_Y)),
            player_vy=jnp.float32(0.0),
            scroll_speed=jnp.float32(_SCROLL_SPEED_BASE),
            coyote_x=jnp.float32(-30.0),
            truck_x=jnp.array([200.0, 260.0, 320.0], dtype=jnp.float32),
            truck_active=jnp.ones(_N_TRUCKS, dtype=jnp.bool_),
            seed_x=jnp.array([80.0, 110.0, 140.0, 170.0, 200.0], dtype=jnp.float32),
            seed_active=jnp.ones(_N_SEEDS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(80),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: RoadRunnerState, action: jax.Array
    ) -> RoadRunnerState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : RoadRunnerState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : RoadRunnerState
            State after one emulated frame.
        """
        key, k_spawn, k_sx = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Speed control
        on_ground = state.player_y >= jnp.float32(_PLAYER_GROUND_Y - 1)
        speed_up = action == jnp.int32(2)
        speed_dn = action == jnp.int32(3)
        new_speed = jnp.clip(
            state.scroll_speed
            + jnp.where(speed_up, jnp.float32(0.1), jnp.float32(0.0))
            + jnp.where(speed_dn, jnp.float32(-0.1), jnp.float32(0.0)),
            jnp.float32(1.0),
            jnp.float32(4.0),
        )

        # Jump
        jump = (action == jnp.int32(4)) & on_ground
        new_vy = jnp.where(jump, jnp.float32(_JUMP_VY), state.player_vy + _GRAVITY)
        new_py = jnp.minimum(state.player_y + new_vy, jnp.float32(_PLAYER_GROUND_Y))
        new_vy = jnp.where(
            new_py >= jnp.float32(_PLAYER_GROUND_Y), jnp.float32(0.0), new_vy
        )

        # Scroll all objects left
        scroll = new_speed
        new_tx = state.truck_x - scroll
        new_sx = state.seed_x - scroll

        # Despawn trucks that scroll off left
        new_truck_active = state.truck_active & (new_tx > jnp.float32(-_TRUCK_W))

        # Seed collision with player (player always at _PLAYER_X)
        seed_hit = (
            state.seed_active
            & (
                jnp.abs(state.seed_x - jnp.float32(_PLAYER_X))
                < jnp.float32(_SEED_W + _PLAYER_W // 2)
            )
            & (
                jnp.abs(jnp.float32(_PLAYER_GROUND_Y) - jnp.float32(_ROAD_Y - _SEED_H))
                < jnp.float32(_SEED_H + _PLAYER_H)
            )
        )
        step_reward = step_reward + jnp.sum(seed_hit).astype(jnp.float32) * jnp.float32(
            100.0
        )
        new_seed_active = state.seed_active & ~seed_hit

        # Despawn seeds off-screen
        new_seed_active = new_seed_active & (new_sx > jnp.float32(-_SEED_W))

        # Truck hits player (player on ground)
        truck_hit_player = (
            new_truck_active
            & on_ground
            & (
                jnp.abs(new_tx - jnp.float32(_PLAYER_X))
                < jnp.float32(_TRUCK_W + _PLAYER_W // 2)
            )
        )
        hit_truck = jnp.any(truck_hit_player)

        # Coyote chases player
        coyote_speed = new_speed * jnp.float32(0.7)
        coyote_target = jnp.float32(_PLAYER_X - 20.0)
        new_coyote_x = state.coyote_x + jnp.clip(
            (coyote_target - state.coyote_x) * jnp.float32(0.05),
            -coyote_speed,
            coyote_speed,
        )
        coyote_catches_player = (
            jnp.abs(new_coyote_x - jnp.float32(_PLAYER_X)) < jnp.float32(8.0)
        ) & on_ground

        # Spawn new trucks and seeds
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_spawn = new_spawn_timer <= jnp.int32(0)

        # Spawn truck
        free_truck = jnp.argmin(new_truck_active.astype(jnp.int32))
        new_tx2 = jnp.where(
            can_spawn,
            new_tx.at[free_truck].set(jnp.float32(170.0)),
            new_tx,
        )
        new_truck_active2 = jnp.where(
            can_spawn,
            new_truck_active.at[free_truck].set(True),
            new_truck_active,
        )

        # Spawn seed
        free_seed = jnp.argmin(new_seed_active.astype(jnp.int32))
        seed_spawn_x = jax.random.uniform(k_sx, minval=100.0, maxval=160.0)
        new_sx2 = jnp.where(
            can_spawn,
            new_sx.at[free_seed].set(seed_spawn_x),
            new_sx,
        )
        new_seed_active2 = jnp.where(
            can_spawn,
            new_seed_active.at[free_seed].set(True),
            new_seed_active,
        )
        new_spawn_timer = jnp.where(can_spawn, jnp.int32(60), new_spawn_timer)

        # Life loss
        life_lost = hit_truck | coyote_catches_player
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return RoadRunnerState(
            player_y=new_py,
            player_vy=new_vy,
            scroll_speed=new_speed,
            coyote_x=new_coyote_x,
            truck_x=new_tx2,
            truck_active=new_truck_active2,
            seed_x=new_sx2,
            seed_active=new_seed_active2,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: RoadRunnerState, action: jax.Array) -> RoadRunnerState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : RoadRunnerState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : RoadRunnerState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: RoadRunnerState) -> RoadRunnerState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: RoadRunnerState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : RoadRunnerState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Road surface
        road_mask = _ROW_IDX >= _ROAD_Y
        frame = jnp.where(road_mask[:, :, None], _COLOR_ROAD, frame)

        # Birdseed
        def draw_seed(frm, i):
            sx = state.seed_x[i].astype(jnp.int32)
            mask = (
                state.seed_active[i]
                & (_ROW_IDX >= _ROAD_Y - _SEED_H)
                & (_ROW_IDX < _ROAD_Y)
                & (_COL_IDX >= sx)
                & (_COL_IDX < sx + _SEED_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_SEED, frm), None

        frame, _ = jax.lax.scan(draw_seed, frame, jnp.arange(_N_SEEDS))

        # Trucks
        def draw_truck(frm, i):
            tx = state.truck_x[i].astype(jnp.int32)
            mask = (
                state.truck_active[i]
                & (_ROW_IDX >= _ROAD_Y - _TRUCK_H)
                & (_ROW_IDX < _ROAD_Y)
                & (_COL_IDX >= tx)
                & (_COL_IDX < tx + _TRUCK_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_TRUCK, frm), None

        frame, _ = jax.lax.scan(draw_truck, frame, jnp.arange(_N_TRUCKS))

        # Coyote
        cx = state.coyote_x.astype(jnp.int32)
        coyote_mask = (
            (_ROW_IDX >= _ROAD_Y - _COYOTE_H)
            & (_ROW_IDX < _ROAD_Y)
            & (_COL_IDX >= cx)
            & (_COL_IDX < cx + _COYOTE_W)
        )
        frame = jnp.where(coyote_mask[:, :, None], _COLOR_COYOTE, frame)

        # Player
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py)
            & (_ROW_IDX < py + _PLAYER_H)
            & (_COL_IDX >= _PLAYER_X)
            & (_COL_IDX < _PLAYER_X + _PLAYER_W)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Road Runner action indices.
        """
        import pygame

        return {
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
            pygame.K_UP: 4,
            pygame.K_w: 4,
            pygame.K_SPACE: 4,
        }
