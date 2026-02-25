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

"""Ice Hockey — JAX-native game implementation.

Two-player ice hockey game.  The player controls a skater (the one nearest
to the puck) and tries to shoot the puck into the opponent's net.  First
to score 6 goals wins, or the higher score after the time limit.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (shoot puck)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Goal scored   — +1
    Goal conceded — −1
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Rink layout
# ---------------------------------------------------------------------------
_RINK_X0: int = 10
_RINK_X1: int = 150
_RINK_Y0: int = 30
_RINK_Y1: int = 185

_NET_W: int = 4
_NET_H: int = 20
_GOAL_Y0: int = (_RINK_Y0 + _RINK_Y1) // 2 - _NET_H // 2
_GOAL_Y1: int = _GOAL_Y0 + _NET_H

_SKATER_W: int = 8
_SKATER_H: int = 10
_PUCK_W: int = 4
_SPEED: float = 1.5  # skater px per sub-step
_PUCK_SPEED: float = 4.0  # puck px per sub-step when shot
_WIN_SCORE: int = 6
_MAX_STEPS: int = 9600  # sub-steps (~2400 agent steps)

# Start positions
_P1_X0: float = 100.0
_P1_Y0: float = 105.0
_P2_X0: float = 52.0
_P2_Y0: float = 105.0
_PUCK_X0: float = 75.0
_PUCK_Y0: float = 107.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_ICE = jnp.array([200, 220, 255], dtype=jnp.uint8)
_COLOR_BOARD = jnp.array([100, 100, 160], dtype=jnp.uint8)
_COLOR_LINE = jnp.array([255, 0, 0], dtype=jnp.uint8)
_COLOR_NET = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_P1 = jnp.array([255, 255, 255], dtype=jnp.uint8)
_COLOR_P2 = jnp.array([200, 50, 50], dtype=jnp.uint8)
_COLOR_PUCK = jnp.array([0, 0, 0], dtype=jnp.uint8)


@chex.dataclass
class IceHockeyState(AtariState):
    """
    Complete Ice Hockey game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player skater x (left edge).
    player_y : jax.Array
        float32 — Player skater y (top edge).
    cpu_x : jax.Array
        float32 — CPU skater x.
    cpu_y : jax.Array
        float32 — CPU skater y.
    puck_x : jax.Array
        float32 — Puck x (left edge).
    puck_y : jax.Array
        float32 — Puck y (top edge).
    puck_dx : jax.Array
        float32 — Puck x velocity.
    puck_dy : jax.Array
        float32 — Puck y velocity.
    puck_active : jax.Array
        bool — Puck is moving freely (not controlled).
    cpu_score : jax.Array
        int32 — CPU score.
    """

    player_x: jax.Array
    player_y: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    puck_x: jax.Array
    puck_y: jax.Array
    puck_dx: jax.Array
    puck_dy: jax.Array
    puck_active: jax.Array
    cpu_score: jax.Array


class IceHockey(AtariEnv):
    """
    Ice Hockey implemented as a pure JAX function suite.

    First to score 6 goals wins.  Episode also ends after the time limit
    (`max_episode_steps = 9600` emulated frames ≈ 2400 agent steps).
    """

    num_actions: int = 6

    def __init__(self, params: EnvParams | None = None) -> None:
        super().__init__(params or EnvParams(noop_max=0, max_episode_steps=9600))

    def _reset(self, key: jax.Array) -> IceHockeyState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : IceHockeyState
            Both skaters at centre-ice, puck at face-off spot.
        """
        return IceHockeyState(
            player_x=jnp.float32(_P1_X0),
            player_y=jnp.float32(_P1_Y0),
            cpu_x=jnp.float32(_P2_X0),
            cpu_y=jnp.float32(_P2_Y0),
            puck_x=jnp.float32(_PUCK_X0),
            puck_y=jnp.float32(_PUCK_Y0),
            puck_dx=jnp.float32(0.0),
            puck_dy=jnp.float32(0.0),
            puck_active=jnp.bool_(False),
            cpu_score=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: IceHockeyState, action: jax.Array) -> IceHockeyState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : IceHockeyState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=DOWN, 5=LEFT.

        Returns
        -------
        new_state : IceHockeyState
            State after one emulated frame.
        """
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(
            action == jnp.int32(3),
            _SPEED,
            jnp.where(action == jnp.int32(5), -_SPEED, jnp.float32(0.0)),
        )
        dy = jnp.where(
            action == jnp.int32(4),
            _SPEED,
            jnp.where(action == jnp.int32(2), -_SPEED, jnp.float32(0.0)),
        )
        new_px = jnp.clip(
            state.player_x + dx, float(_RINK_X0), float(_RINK_X1 - _SKATER_W)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_RINK_Y0), float(_RINK_Y1 - _SKATER_H)
        )

        # CPU AI: move toward puck, then toward player goal (left net)
        puck_cx = state.puck_x + _PUCK_W / 2
        puck_cy = state.puck_y + _PUCK_W / 2
        cpu_tx = jnp.where(state.puck_active, puck_cx, state.puck_x)
        cpu_ty = jnp.where(state.puck_active, puck_cy, state.puck_y)
        cpu_dx = jnp.clip((cpu_tx - state.cpu_x) * 0.06, -_SPEED, _SPEED)
        cpu_dy = jnp.clip((cpu_ty - state.cpu_y) * 0.06, -_SPEED, _SPEED)
        new_cx = jnp.clip(
            state.cpu_x + cpu_dx, float(_RINK_X0), float(_RINK_X1 - _SKATER_W)
        )
        new_cy = jnp.clip(
            state.cpu_y + cpu_dy, float(_RINK_Y0), float(_RINK_Y1 - _SKATER_H)
        )

        # Puck possession: player picks up puck if nearby
        player_cx = new_px + _SKATER_W / 2
        player_cy = new_py + _SKATER_H / 2
        dist_player_puck = jnp.sqrt(
            (player_cx - puck_cx) ** 2 + (player_cy - puck_cy) ** 2
        )
        player_has_puck = ~state.puck_active & (dist_player_puck < jnp.float32(12.0))

        dist_cpu_puck = jnp.sqrt(
            (new_cx + _SKATER_W / 2 - puck_cx) ** 2
            + (new_cy + _SKATER_H / 2 - puck_cy) ** 2
        )
        cpu_has_puck = (
            ~state.puck_active & ~player_has_puck & (dist_cpu_puck < jnp.float32(12.0))
        )

        # Puck sticks to player if player holds it
        new_puck_x = jnp.where(player_has_puck, new_px + jnp.float32(4), state.puck_x)
        new_puck_y = jnp.where(player_has_puck, new_py - jnp.float32(4), state.puck_y)

        # Player shoots puck toward left net
        fire = (action == jnp.int32(1)) & player_has_puck
        shoot_dx = jnp.where(fire, -_PUCK_SPEED, jnp.float32(0.0))
        shoot_dy = jnp.where(fire, jnp.float32(0.0), jnp.float32(0.0))
        new_puck_active = jnp.where(
            fire, jnp.bool_(True), state.puck_active & ~player_has_puck
        )

        # CPU shoots toward right net (player's goal)
        cpu_shoot = cpu_has_puck
        cpu_puck_x = jnp.where(cpu_has_puck, new_cx + jnp.float32(8), state.puck_x)
        cpu_puck_y = jnp.where(cpu_has_puck, new_cy + jnp.float32(5), state.puck_y)
        new_puck_x = jnp.where(cpu_has_puck, cpu_puck_x, new_puck_x)
        new_puck_y = jnp.where(cpu_has_puck, cpu_puck_y, new_puck_y)
        new_puck_dx = jnp.where(
            fire, shoot_dx, jnp.where(cpu_shoot, _PUCK_SPEED, state.puck_dx)
        )
        new_puck_dy = jnp.where(
            fire, shoot_dy, jnp.where(cpu_shoot, jnp.float32(0.0), state.puck_dy)
        )
        new_puck_active = jnp.where(cpu_shoot, jnp.bool_(True), new_puck_active)

        # Move puck
        moved_puck_x = jnp.where(new_puck_active, new_puck_x + new_puck_dx, new_puck_x)
        moved_puck_y = jnp.where(new_puck_active, new_puck_y + new_puck_dy, new_puck_y)

        # Board bounces (top/bottom)
        hit_top = moved_puck_y < float(_RINK_Y0)
        hit_bot = moved_puck_y + _PUCK_W > float(_RINK_Y1)
        new_puck_dy = jnp.where(hit_top | hit_bot, -new_puck_dy, new_puck_dy)
        moved_puck_y = jnp.clip(
            moved_puck_y, float(_RINK_Y0), float(_RINK_Y1 - _PUCK_W)
        )

        # Goal detection: left net (player shoots into this) and right net (CPU shoots into this)
        in_goal_y = (moved_puck_y + _PUCK_W / 2 >= _GOAL_Y0) & (
            moved_puck_y + _PUCK_W / 2 <= _GOAL_Y1
        )
        player_goal = new_puck_active & (moved_puck_x < float(_RINK_X0)) & in_goal_y
        cpu_goal = (
            new_puck_active & (moved_puck_x + _PUCK_W > float(_RINK_X1)) & in_goal_y
        )

        step_reward = step_reward + jnp.where(
            player_goal, jnp.float32(1.0), jnp.float32(0.0)
        )
        step_reward = step_reward - jnp.where(
            cpu_goal, jnp.float32(1.0), jnp.float32(0.0)
        )
        new_score = state.score + jnp.where(player_goal, jnp.int32(1), jnp.int32(0))
        new_cpu_score = state.cpu_score + jnp.where(
            cpu_goal, jnp.int32(1), jnp.int32(0)
        )

        # Reset puck to centre after goal
        goal = player_goal | cpu_goal
        moved_puck_x = jnp.where(goal, jnp.float32(_PUCK_X0), moved_puck_x)
        moved_puck_y = jnp.where(goal, jnp.float32(_PUCK_Y0), moved_puck_y)
        new_puck_dx = jnp.where(goal, jnp.float32(0.0), new_puck_dx)
        new_puck_dy = jnp.where(goal, jnp.float32(0.0), new_puck_dy)
        new_puck_active = jnp.where(goal, jnp.bool_(False), new_puck_active)

        done = (new_score >= jnp.int32(_WIN_SCORE)) | (
            new_cpu_score >= jnp.int32(_WIN_SCORE)
        )

        return IceHockeyState(
            player_x=new_px,
            player_y=new_py,
            cpu_x=new_cx,
            cpu_y=new_cy,
            puck_x=moved_puck_x,
            puck_y=moved_puck_y,
            puck_dx=new_puck_dx,
            puck_dy=new_puck_dy,
            puck_active=new_puck_active,
            cpu_score=new_cpu_score,
            lives=jnp.int32(0),
            score=new_score,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: IceHockeyState, action: jax.Array) -> IceHockeyState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : IceHockeyState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : IceHockeyState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: IceHockeyState) -> IceHockeyState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: IceHockeyState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : IceHockeyState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)

        # Ice rink
        in_rink = (
            (_ROW_IDX >= _RINK_Y0)
            & (_ROW_IDX < _RINK_Y1)
            & (_COL_IDX >= _RINK_X0)
            & (_COL_IDX < _RINK_X1)
        )
        frame = jnp.where(in_rink[:, :, None], _COLOR_ICE, frame)

        # Boards (perimeter)
        board = (
            ((_ROW_IDX == _RINK_Y0) | (_ROW_IDX == _RINK_Y1 - 1))
            & (_COL_IDX >= _RINK_X0)
            & (_COL_IDX < _RINK_X1)
        ) | (
            ((_COL_IDX == _RINK_X0) | (_COL_IDX == _RINK_X1 - 1))
            & (_ROW_IDX >= _RINK_Y0)
            & (_ROW_IDX < _RINK_Y1)
        )
        frame = jnp.where(board[:, :, None], _COLOR_BOARD, frame)

        # Centre line (red)
        ctr_x = (_RINK_X0 + _RINK_X1) // 2
        centre = (_COL_IDX == ctr_x) & in_rink
        frame = jnp.where(centre[:, :, None], _COLOR_LINE, frame)

        # Nets
        left_net = (
            (_COL_IDX >= _RINK_X0 - _NET_W)
            & (_COL_IDX < _RINK_X0)
            & (_ROW_IDX >= _GOAL_Y0)
            & (_ROW_IDX < _GOAL_Y1)
        )
        right_net = (
            (_COL_IDX >= _RINK_X1)
            & (_COL_IDX < _RINK_X1 + _NET_W)
            & (_ROW_IDX >= _GOAL_Y0)
            & (_ROW_IDX < _GOAL_Y1)
        )
        frame = jnp.where((left_net | right_net)[:, :, None], _COLOR_NET, frame)

        # CPU skater
        cpu_mask = (
            (_ROW_IDX >= jnp.int32(state.cpu_y))
            & (_ROW_IDX < jnp.int32(state.cpu_y) + _SKATER_H)
            & (_COL_IDX >= jnp.int32(state.cpu_x))
            & (_COL_IDX < jnp.int32(state.cpu_x) + _SKATER_W)
        )
        frame = jnp.where(cpu_mask[:, :, None], _COLOR_P2, frame)

        # Player skater
        p_mask = (
            (_ROW_IDX >= jnp.int32(state.player_y))
            & (_ROW_IDX < jnp.int32(state.player_y) + _SKATER_H)
            & (_COL_IDX >= jnp.int32(state.player_x))
            & (_COL_IDX < jnp.int32(state.player_x) + _SKATER_W)
        )
        frame = jnp.where(p_mask[:, :, None], _COLOR_P1, frame)

        # Puck
        puck_mask = (
            (_ROW_IDX >= jnp.int32(state.puck_y))
            & (_ROW_IDX < jnp.int32(state.puck_y) + _PUCK_W)
            & (_COL_IDX >= jnp.int32(state.puck_x))
            & (_COL_IDX < jnp.int32(state.puck_x) + _PUCK_W)
        )
        frame = jnp.where(puck_mask[:, :, None], _COLOR_PUCK, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Ice Hockey action indices.
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
