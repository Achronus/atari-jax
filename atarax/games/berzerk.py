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

"""Berzerk — JAX-native game implementation.

Navigate a maze of electrified walls while shooting robots and evading
Evil Otto (the bouncing smiley face).  Touch a wall = death; robots shoot
back; Evil Otto cannot be killed.

Action space (9 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — UP+FIRE
    7 — RIGHT+FIRE
    8 — DOWN+FIRE
    (LEFT+FIRE would be action 9, but we use 9 actions total)

Scoring:
    Robot shot — +50
    Room cleared — extra life bonus; player exits right
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 5.0
_ROBOT_SPEED: float = 0.8
_OTTO_SPEED: float = 1.2
_N_ROBOTS: int = 6

_WALL_LEFT: int = 10
_WALL_RIGHT: int = 150
_WALL_TOP: int = 30
_WALL_BOTTOM: int = 190

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_WALL = jnp.array([0, 200, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 255], dtype=jnp.uint8)
_COLOR_ROBOT = jnp.array([200, 0, 0], dtype=jnp.uint8)
_COLOR_OTTO = jnp.array([255, 200, 0], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ROBOT_BULLET = jnp.array([200, 100, 0], dtype=jnp.uint8)


@chex.dataclass
class BerzerkState(AtariState):
    """
    Complete Berzerk game state — a JAX pytree.

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
    bullet_dx : jax.Array
        float32 — Player bullet x velocity.
    bullet_dy : jax.Array
        float32 — Player bullet y velocity.
    bullet_active : jax.Array
        bool — Player bullet in flight.
    robot_x : jax.Array
        float32[6] — Robot x positions.
    robot_y : jax.Array
        float32[6] — Robot y positions.
    robot_active : jax.Array
        bool[6] — Robot alive.
    robot_bx : jax.Array
        float32[6] — Robot bullet x.
    robot_by : jax.Array
        float32[6] — Robot bullet y.
    robot_bdx : jax.Array
        float32[6] — Robot bullet x velocity.
    robot_bdy : jax.Array
        float32[6] — Robot bullet y velocity.
    robot_bactive : jax.Array
        bool[6] — Robot bullet active.
    otto_x : jax.Array
        float32 — Evil Otto x.
    otto_y : jax.Array
        float32 — Evil Otto y.
    fire_timer : jax.Array
        int32 — Frames until robots fire again.
    room : jax.Array
        int32 — Current room index.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dx: jax.Array
    bullet_dy: jax.Array
    bullet_active: jax.Array
    robot_x: jax.Array
    robot_y: jax.Array
    robot_active: jax.Array
    robot_bx: jax.Array
    robot_by: jax.Array
    robot_bdx: jax.Array
    robot_bdy: jax.Array
    robot_bactive: jax.Array
    otto_x: jax.Array
    otto_y: jax.Array
    fire_timer: jax.Array
    room: jax.Array
    key: jax.Array


_DIR_DX = jnp.array(
    [0.0, _BULLET_SPEED, 0.0, -_BULLET_SPEED, 0.0, _BULLET_SPEED, 0.0, -_BULLET_SPEED],
    dtype=jnp.float32,
)
_DIR_DY = jnp.array(
    [-_BULLET_SPEED, 0.0, _BULLET_SPEED, 0.0, -_BULLET_SPEED, 0.0, _BULLET_SPEED, 0.0],
    dtype=jnp.float32,
)


class Berzerk(AtariEnv):
    """
    Berzerk implemented as a pure JAX function suite.

    Shoot robots; don't touch electrified walls or Evil Otto.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> BerzerkState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BerzerkState
            Player in centre, 6 robots arranged around room, Otto off-screen.
        """
        robot_xs = jnp.array([30.0, 80.0, 130.0, 30.0, 80.0, 130.0], dtype=jnp.float32)
        robot_ys = jnp.array([60.0, 60.0, 60.0, 140.0, 140.0, 140.0], dtype=jnp.float32)
        return BerzerkState(
            player_x=jnp.float32(75.0),
            player_y=jnp.float32(105.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(105.0),
            bullet_dx=jnp.float32(0.0),
            bullet_dy=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            robot_x=robot_xs,
            robot_y=robot_ys,
            robot_active=jnp.ones(_N_ROBOTS, dtype=jnp.bool_),
            robot_bx=robot_xs.copy(),
            robot_by=robot_ys.copy(),
            robot_bdx=jnp.zeros(_N_ROBOTS, dtype=jnp.float32),
            robot_bdy=jnp.zeros(_N_ROBOTS, dtype=jnp.float32),
            robot_bactive=jnp.zeros(_N_ROBOTS, dtype=jnp.bool_),
            otto_x=jnp.float32(-20.0),
            otto_y=jnp.float32(105.0),
            fire_timer=jnp.int32(60),
            room=jnp.int32(0),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: BerzerkState, action: jax.Array) -> BerzerkState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BerzerkState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : BerzerkState
            State after one emulated frame.
        """
        key, k_fire = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        dy = jnp.where(
            action == 2, -_PLAYER_SPEED, jnp.where(action == 4, _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(
            state.player_x + dx, float(_WALL_LEFT + 2), float(_WALL_RIGHT - 8)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_WALL_TOP + 2), float(_WALL_BOTTOM - 12)
        )

        # Wall collision — simplified (boundary only)
        hit_wall = (
            (new_px <= jnp.float32(_WALL_LEFT + 2))
            | (new_px >= jnp.float32(_WALL_RIGHT - 8))
            | (new_py <= jnp.float32(_WALL_TOP + 2))
            | (new_py >= jnp.float32(_WALL_BOTTOM - 12))
        )

        # Fire bullet — action maps: 1=N, 2=E, 3=S, 4=W... mapped 6-8 to NE SE SW NW
        fire = (action >= jnp.int32(1)) & ~state.bullet_active
        fire_dir = jnp.clip(action - jnp.int32(1), 0, 7)
        new_bx = jnp.where(fire, new_px + jnp.float32(4.0), state.bullet_x)
        new_by = jnp.where(fire, new_py + jnp.float32(4.0), state.bullet_y)
        new_bdx = jnp.where(fire, _DIR_DX[fire_dir], state.bullet_dx)
        new_bdy = jnp.where(fire, _DIR_DY[fire_dir], state.bullet_dy)
        new_bactive = state.bullet_active | fire

        new_bx = jnp.where(new_bactive, new_bx + new_bdx, new_bx)
        new_by = jnp.where(new_bactive, new_by + new_bdy, new_by)
        bullet_out = (
            (new_bx < _WALL_LEFT)
            | (new_bx > _WALL_RIGHT)
            | (new_by < _WALL_TOP)
            | (new_by > _WALL_BOTTOM)
        )
        new_bactive = new_bactive & ~bullet_out

        # Robot movement toward player
        rdx = jnp.clip((new_px - state.robot_x) * 0.05, -_ROBOT_SPEED, _ROBOT_SPEED)
        rdy = jnp.clip((new_py - state.robot_y) * 0.05, -_ROBOT_SPEED, _ROBOT_SPEED)
        new_rx = jnp.where(state.robot_active, state.robot_x + rdx, state.robot_x)
        new_ry = jnp.where(state.robot_active, state.robot_y + rdy, state.robot_y)

        # Robots hit walls → die
        robot_hit_wall = (
            (new_rx <= jnp.float32(_WALL_LEFT))
            | (new_rx >= jnp.float32(_WALL_RIGHT - 8))
            | (new_ry <= jnp.float32(_WALL_TOP))
            | (new_ry >= jnp.float32(_WALL_BOTTOM - 8))
        )
        robot_wall_kills = state.robot_active & robot_hit_wall
        new_robot_active = state.robot_active & ~robot_wall_kills

        # Bullet hits robot
        bullet_hit = (
            new_bactive
            & new_robot_active
            & (jnp.abs(new_bx - new_rx) < jnp.float32(8.0))
            & (jnp.abs(new_by - new_ry) < jnp.float32(8.0))
        )
        step_reward = step_reward + jnp.sum(bullet_hit).astype(
            jnp.float32
        ) * jnp.float32(50.0)
        new_robot_active = new_robot_active & ~bullet_hit
        new_bactive = new_bactive & ~jnp.any(bullet_hit)

        # Robot bullet logic
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & new_robot_active
        shooter = jnp.argmax(can_fire.astype(jnp.int32))
        fire_dir_to_player_dx = jnp.where(
            new_px > new_rx[shooter], jnp.float32(2.0), jnp.float32(-2.0)
        )
        fire_dir_to_player_dy = jnp.where(
            new_py > new_ry[shooter], jnp.float32(2.0), jnp.float32(-2.0)
        )
        shot = jnp.any(can_fire)
        new_rbx = jnp.where(
            shot, state.robot_bx.at[shooter].set(new_rx[shooter] + 4.0), state.robot_bx
        )
        new_rby = jnp.where(
            shot, state.robot_by.at[shooter].set(new_ry[shooter] + 4.0), state.robot_by
        )
        new_rbdx = jnp.where(
            shot,
            state.robot_bdx.at[shooter].set(fire_dir_to_player_dx),
            state.robot_bdx,
        )
        new_rbdy = jnp.where(
            shot,
            state.robot_bdy.at[shooter].set(fire_dir_to_player_dy),
            state.robot_bdy,
        )
        new_rbactive = jnp.where(
            shot, state.robot_bactive.at[shooter].set(True), state.robot_bactive
        )
        new_fire_timer = jnp.where(shot, jnp.int32(45), new_fire_timer)

        new_rbx = new_rbx + new_rbdx * new_rbactive.astype(jnp.float32)
        new_rby = new_rby + new_rbdy * new_rbactive.astype(jnp.float32)
        rbout = (
            (new_rbx < _WALL_LEFT)
            | (new_rbx > _WALL_RIGHT)
            | (new_rby < _WALL_TOP)
            | (new_rby > _WALL_BOTTOM)
        )
        new_rbactive = new_rbactive & ~rbout

        # Robot bullet hits player
        robot_bullet_hits_player = (
            new_rbactive
            & (jnp.abs(new_rbx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_rby - new_py) < jnp.float32(8.0))
        )
        hit_by_robot_bullet = jnp.any(robot_bullet_hits_player)

        # Robot touches player
        robot_touches_player = (
            new_robot_active
            & (jnp.abs(new_rx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_ry - new_py) < jnp.float32(8.0))
        )
        hit_by_robot = jnp.any(robot_touches_player)

        # Evil Otto moves toward player
        new_otto_x = state.otto_x + jnp.clip(
            (new_px - state.otto_x) * 0.04, -_OTTO_SPEED, _OTTO_SPEED
        )
        new_otto_y = state.otto_y + jnp.clip(
            (new_py - state.otto_y) * 0.04, -_OTTO_SPEED, _OTTO_SPEED
        )
        otto_hits_player = (jnp.abs(new_otto_x - new_px) < jnp.float32(8.0)) & (
            jnp.abs(new_otto_y - new_py) < jnp.float32(8.0)
        )

        # Room clear → player exits right (next room)
        room_clear = ~jnp.any(new_robot_active)
        player_exits = room_clear & (new_px >= jnp.float32(_WALL_RIGHT - 12))
        new_room = state.room + jnp.where(player_exits, jnp.int32(1), jnp.int32(0))

        # Life loss
        life_lost = hit_wall | hit_by_robot_bullet | hit_by_robot | otto_hits_player
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return BerzerkState(
            player_x=new_px,
            player_y=new_py,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_dx=new_bdx,
            bullet_dy=new_bdy,
            bullet_active=new_bactive,
            robot_x=new_rx,
            robot_y=new_ry,
            robot_active=new_robot_active,
            robot_bx=new_rbx,
            robot_by=new_rby,
            robot_bdx=new_rbdx,
            robot_bdy=new_rbdy,
            robot_bactive=new_rbactive,
            otto_x=new_otto_x,
            otto_y=new_otto_y,
            fire_timer=new_fire_timer,
            room=new_room,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: BerzerkState, action: jax.Array) -> BerzerkState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BerzerkState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : BerzerkState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BerzerkState) -> BerzerkState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BerzerkState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BerzerkState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Outer walls
        wall_mask = (
            (_ROW_IDX == _WALL_TOP)
            | (_ROW_IDX == _WALL_BOTTOM)
            | (_COL_IDX == _WALL_LEFT)
            | (_COL_IDX == _WALL_RIGHT)
        )
        frame = jnp.where(wall_mask[:, :, None], _COLOR_WALL, frame)

        # Robots
        def draw_robot(frm, i):
            rx = state.robot_x[i].astype(jnp.int32)
            ry = state.robot_y[i].astype(jnp.int32)
            mask = (
                state.robot_active[i]
                & (_ROW_IDX >= ry)
                & (_ROW_IDX < ry + 8)
                & (_COL_IDX >= rx)
                & (_COL_IDX < rx + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_ROBOT, frm), None

        frame, _ = jax.lax.scan(draw_robot, frame, jnp.arange(_N_ROBOTS))

        # Robot bullets
        def draw_rbullet(frm, i):
            rbx = state.robot_bx[i].astype(jnp.int32)
            rby = state.robot_by[i].astype(jnp.int32)
            mask = (
                state.robot_bactive[i]
                & (_ROW_IDX >= rby)
                & (_ROW_IDX < rby + 4)
                & (_COL_IDX >= rbx)
                & (_COL_IDX < rbx + 2)
            )
            return jnp.where(mask[:, :, None], _COLOR_ROBOT_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_rbullet, frame, jnp.arange(_N_ROBOTS))

        # Otto
        ox = state.otto_x.astype(jnp.int32)
        oy = state.otto_y.astype(jnp.int32)
        otto_mask = (
            (_ROW_IDX >= oy)
            & (_ROW_IDX < oy + 10)
            & (_COL_IDX >= ox)
            & (_COL_IDX < ox + 10)
        )
        frame = jnp.where(otto_mask[:, :, None], _COLOR_OTTO, frame)

        # Player bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32))
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 4)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32))
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py)
            & (_ROW_IDX < py + 10)
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
            Mapping of pygame key constants to Berzerk action indices.
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
