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

"""Beam Rider — JAX-native game implementation.

Ride the tractor beam toward a mothership, destroying enemy drones
in 15 circular lanes while dodging projectiles and debris.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (shoot forward)
    2 — LEFT  (move one lane counter-clockwise)
    3 — RIGHT (move one lane clockwise)
    4 — LEFT + FIRE
    5 — RIGHT + FIRE

Scoring:
    Drone destroyed  — +100
    Debris cleared   — +50
    Sector complete  — +1000 bonus
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
_N_LANES: int = 15   # circular lanes
_N_DRONES: int = 15  # maximum concurrent drones (one per lane)
_N_BULLETS: int = 3  # max player bullets in flight
_N_ENEMY_BULLETS: int = 6
_SECTOR_DRONES: int = 15  # drones per sector (kill all = advance)

_PLAYER_START_LANE: int = 7   # centre lane
_BULLET_SPEED: float = 4.0    # toward top of screen
_DRONE_SPEED: float = 1.0
_ENEMY_BULLET_SPEED: float = 2.5
_INIT_LIVES: int = 3

# Screen coords: lanes at fixed y=170, different x; drones move down
_LANE_X = jnp.array([8 + i * 10 for i in range(_N_LANES)], dtype=jnp.int32)  # [15]
_PLAYER_Y: int = 170
_BULLET_Y0: int = 155   # where player bullets start
_DRONE_Y0: int = 30     # where drones start

_SPAWN_INTERVAL: int = 20

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_BEAM = jnp.array([40, 40, 100], dtype=jnp.uint8)
_COLOR_DRONE = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ENEMY_BULLET = jnp.array([255, 100, 50], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([80, 200, 255], dtype=jnp.uint8)


@chex.dataclass
class BeamRiderState(AtariState):
    """
    Complete Beam Rider game state — a JAX pytree.

    Parameters
    ----------
    player_lane : jax.Array
        int32 — Current lane (0–14).
    bullet_y : jax.Array
        float32[3] — Player bullet y positions.
    bullet_lane : jax.Array
        int32[3] — Player bullet lane indices.
    bullet_active : jax.Array
        bool[3] — Player bullet in-flight.
    drone_y : jax.Array
        float32[15] — Drone y positions.
    drone_active : jax.Array
        bool[15] — Drone alive.
    enemy_bullet_y : jax.Array
        float32[6] — Enemy bullet y positions.
    enemy_bullet_lane : jax.Array
        int32[6] — Enemy bullet lanes.
    enemy_bullet_active : jax.Array
        bool[6] — Enemy bullet in-flight.
    spawn_timer : jax.Array
        int32 — Frames until next drone spawn.
    sector : jax.Array
        int32 — Current sector number.
    drones_killed : jax.Array
        int32 — Drones destroyed in current sector.
    fire_timer : jax.Array
        int32 — Frames until next enemy fire.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_lane: jax.Array
    bullet_y: jax.Array
    bullet_lane: jax.Array
    bullet_active: jax.Array
    drone_y: jax.Array
    drone_active: jax.Array
    enemy_bullet_y: jax.Array
    enemy_bullet_lane: jax.Array
    enemy_bullet_active: jax.Array
    spawn_timer: jax.Array
    sector: jax.Array
    drones_killed: jax.Array
    fire_timer: jax.Array
    key: jax.Array


class BeamRider(AtariEnv):
    """
    Beam Rider implemented as a pure JAX function suite.

    Destroy drones in each sector to advance.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=200_000)

    def _reset(self, key: jax.Array) -> BeamRiderState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BeamRiderState
            Player at centre lane, no drones or bullets, 3 lives.
        """
        return BeamRiderState(
            player_lane=jnp.int32(_PLAYER_START_LANE),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_lane=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            drone_y=jnp.full(_N_DRONES, jnp.float32(_DRONE_Y0), dtype=jnp.float32),
            drone_active=jnp.zeros(_N_DRONES, dtype=jnp.bool_),
            enemy_bullet_y=jnp.full(_N_ENEMY_BULLETS, -10.0, dtype=jnp.float32),
            enemy_bullet_lane=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.int32),
            enemy_bullet_active=jnp.zeros(_N_ENEMY_BULLETS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            sector=jnp.int32(0),
            drones_killed=jnp.int32(0),
            fire_timer=jnp.int32(30),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: BeamRiderState, action: jax.Array) -> BeamRiderState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BeamRiderState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : BeamRiderState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Player lane movement
        move_left = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_right = (action == jnp.int32(3)) | (action == jnp.int32(5))
        new_lane = jnp.clip(
            state.player_lane
            - jnp.where(move_left, jnp.int32(1), jnp.int32(0))
            + jnp.where(move_right, jnp.int32(1), jnp.int32(0)),
            0, _N_LANES - 1,
        )

        # Fire player bullet
        do_fire = (action == jnp.int32(1)) | (action == jnp.int32(4)) | (action == jnp.int32(5))
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        new_bul_y = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(jnp.float32(_BULLET_Y0)),
            state.bullet_y,
        )
        new_bul_lane = jnp.where(
            do_fire & has_free,
            state.bullet_lane.at[free_slot].set(new_lane),
            state.bullet_lane,
        )
        new_bul_active = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )

        # Move player bullets up
        new_bul_y = new_bul_y - jnp.float32(_BULLET_SPEED)
        new_bul_active = new_bul_active & (new_bul_y > jnp.float32(0))

        # Move drones down
        new_drone_y = state.drone_y + jnp.float32(_DRONE_SPEED)
        # Drones reaching player height: remove and trigger life loss
        drone_reach_player = state.drone_active & (new_drone_y >= jnp.float32(_PLAYER_Y))
        any_drone_at_player = jnp.any(drone_reach_player)
        new_drone_active = state.drone_active & ~drone_reach_player

        # Bullet–drone collision (per bullet, check lane match)
        def check_bullet(i, carry):
            bul_y, bul_lane, bul_active, drone_active_, drone_y_, killed, reward_ = carry
            by = bul_y[i]
            bl = bul_lane[i]
            ba = bul_active[i]
            # drone in same lane
            lane_match = (jnp.arange(_N_DRONES, dtype=jnp.int32) == bl) & drone_active_
            # drone position overlap
            y_hit = (new_drone_y >= by - 6.0) & (new_drone_y <= by + 6.0)
            hit = ba & lane_match & y_hit
            any_hit = jnp.any(hit)
            bul_active = bul_active.at[i].set(bul_active[i] & ~any_hit)
            drone_active_ = drone_active_ & ~hit
            n_hit = jnp.sum(hit).astype(jnp.int32)
            killed = killed + n_hit
            reward_ = reward_ + jnp.float32(n_hit * 100)
            return bul_y, bul_lane, bul_active, drone_active_, drone_y_, killed, reward_

        _, _, new_bul_active, new_drone_active, new_drone_y, n_killed, bullet_reward = (
            jax.lax.fori_loop(
                0, _N_BULLETS, check_bullet,
                (new_bul_y, new_bul_lane, new_bul_active, new_drone_active, new_drone_y,
                 jnp.int32(0), jnp.float32(0.0)),
            )
        )
        step_reward = step_reward + bullet_reward

        # Spawn new drone
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer)
        spawn_lane = jax.random.randint(sk1, (), 0, _N_LANES)
        free_drone = jnp.argmin(new_drone_active.astype(jnp.int32))
        has_free_drone = ~jnp.all(new_drone_active)

        # Place drone in spawn_lane slot
        target = jnp.where(has_free_drone, free_drone, jnp.int32(0))
        new_drone_y = jnp.where(
            do_spawn & has_free_drone,
            new_drone_y.at[target].set(jnp.float32(_DRONE_Y0)),
            new_drone_y,
        )
        # Use drone index as lane (distribute across lanes)
        new_drone_active = jnp.where(
            do_spawn & has_free_drone,
            new_drone_active.at[target].set(True),
            new_drone_active,
        )

        # Enemy fires bullet toward player
        new_fire_timer = state.fire_timer - jnp.int32(1)
        do_enemy_fire = new_fire_timer <= jnp.int32(0)
        new_fire_timer = jnp.where(do_enemy_fire, jnp.int32(25), new_fire_timer)
        fire_drone_lane = jax.random.randint(sk2, (), 0, _N_LANES)
        free_eb = jnp.argmin(state.enemy_bullet_active.astype(jnp.int32))
        new_eb_y = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_y.at[free_eb].set(jnp.float32(_DRONE_Y0 + 20)),
            state.enemy_bullet_y,
        )
        new_eb_lane = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_lane.at[free_eb].set(fire_drone_lane),
            state.enemy_bullet_lane,
        )
        new_eb_active = jnp.where(
            do_enemy_fire,
            state.enemy_bullet_active.at[free_eb].set(True),
            state.enemy_bullet_active,
        )

        # Move enemy bullets down
        new_eb_y = new_eb_y + jnp.float32(_ENEMY_BULLET_SPEED)
        new_eb_active = new_eb_active & (new_eb_y < jnp.float32(200))

        # Enemy bullet hits player
        eb_hits_player = (
            new_eb_active
            & (new_eb_lane == new_lane)
            & (new_eb_y >= jnp.float32(_PLAYER_Y - 5))
        )
        any_eb_hit = jnp.any(eb_hits_player)
        new_eb_active = new_eb_active & ~eb_hits_player

        # Sector complete
        new_drones_killed = state.drones_killed + n_killed
        sector_complete = new_drones_killed >= jnp.int32(_SECTOR_DRONES)
        step_reward = step_reward + jnp.where(sector_complete, jnp.float32(1000.0), jnp.float32(0.0))
        new_sector = state.sector + jnp.where(sector_complete, jnp.int32(1), jnp.int32(0))
        new_drones_killed = jnp.where(sector_complete, jnp.int32(0), new_drones_killed)

        # Life loss
        life_lost = any_drone_at_player | any_eb_hit
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))

        done = new_lives <= jnp.int32(0)

        return BeamRiderState(
            player_lane=new_lane,
            bullet_y=new_bul_y,
            bullet_lane=new_bul_lane,
            bullet_active=new_bul_active,
            drone_y=new_drone_y,
            drone_active=new_drone_active,
            enemy_bullet_y=new_eb_y,
            enemy_bullet_lane=new_eb_lane,
            enemy_bullet_active=new_eb_active,
            spawn_timer=new_spawn_timer,
            sector=new_sector,
            drones_killed=new_drones_killed,
            fire_timer=new_fire_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: BeamRiderState, action: jax.Array) -> BeamRiderState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BeamRiderState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : BeamRiderState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BeamRiderState) -> BeamRiderState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BeamRiderState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BeamRiderState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Draw beam lanes (vertical lines)
        def draw_lane(frm, i):
            x = _LANE_X[i]
            mask = (_COL_IDX == x) & (_ROW_IDX < 200)
            return jnp.where(mask[:, :, None], _COLOR_BEAM, frm), None

        frame, _ = jax.lax.scan(draw_lane, frame, jnp.arange(_N_LANES))

        # Draw drones (use lane index as x position)
        def draw_drone(frm, i):
            dx = _LANE_X[i % _N_LANES]
            dy = state.drone_y[i].astype(jnp.int32)
            mask = (
                state.drone_active[i]
                & (_ROW_IDX >= dy - 4) & (_ROW_IDX <= dy + 4)
                & (_COL_IDX >= dx - 4) & (_COL_IDX <= dx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_DRONE, frm), None

        frame, _ = jax.lax.scan(draw_drone, frame, jnp.arange(_N_DRONES))

        # Draw player bullets
        def draw_bullet(frm, i):
            bx = _LANE_X[state.bullet_lane[i]]
            by = state.bullet_y[i].astype(jnp.int32)
            mask = state.bullet_active[i] & (_ROW_IDX >= by - 3) & (_ROW_IDX <= by) & (_COL_IDX == bx)
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Draw player ship
        px = _LANE_X[state.player_lane]
        player_mask = (
            (_ROW_IDX >= _PLAYER_Y - 5) & (_ROW_IDX <= _PLAYER_Y + 5)
            & (_COL_IDX >= px - 5) & (_COL_IDX <= px + 5)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Beam Rider action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_LEFT: 2,
            pygame.K_a: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
        }
