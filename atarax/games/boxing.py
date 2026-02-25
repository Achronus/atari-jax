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

"""Boxing — JAX-native game implementation.

Trade punches with a CPU opponent in a boxing ring.  Score points for each
punch that lands.  First to 100 points wins by knockout; otherwise the
higher score after the time limit wins.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (punch)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_RING_X0: int = 8
_RING_X1: int = 152
_RING_Y0: int = 50
_RING_Y1: int = 185

_BOXER_W: int = 8
_BOXER_H: int = 12
_SPEED: float = 1.5  # px per sub-step
_PUNCH_RANGE: float = 18.0  # max distance for punch to land
_PUNCH_COOLDOWN: int = 8  # sub-steps between punches
_KO_SCORE: int = 100  # score limit for knockout

# Episode length: 7680 sub-steps = 1920 agent steps (~2 min at 60 Hz / 4× skip)
_MAX_STEPS: int = 7680

# Start positions
_PLAYER_X0: float = 64.0
_PLAYER_Y0: float = 150.0
_CPU_X0: float = 88.0
_CPU_Y0: float = 70.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_RING = jnp.array([180, 120, 60], dtype=jnp.uint8)
_COLOR_ROPE = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 255, 255], dtype=jnp.uint8)
_COLOR_CPU = jnp.array([200, 0, 0], dtype=jnp.uint8)
_COLOR_FLASH = jnp.array([255, 255, 0], dtype=jnp.uint8)


@chex.dataclass
class BoxingState(AtariState):
    """
    Complete Boxing game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player boxer x (left edge).
    player_y : jax.Array
        float32 — Player boxer y (top edge).
    cpu_x : jax.Array
        float32 — CPU boxer x.
    cpu_y : jax.Array
        float32 — CPU boxer y.
    cpu_score : jax.Array
        int32 — CPU's accumulated score.
    punch_timer : jax.Array
        int32 — Sub-steps until player can punch again.
    cpu_punch_timer : jax.Array
        int32 — Sub-steps until CPU can punch again.
    hit_flash : jax.Array
        int32 — Sub-steps to display punch-landed flash.
    """

    player_x: jax.Array
    player_y: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    cpu_score: jax.Array
    punch_timer: jax.Array
    cpu_punch_timer: jax.Array
    hit_flash: jax.Array


class Boxing(AtariEnv):
    """
    Boxing implemented as a pure JAX function suite.

    No lives system.  Episode ends on knockout (either boxer reaches 100 pts)
    or time limit (`max_episode_steps = 7680` emulated frames ≈ 1920 agent
    steps).
    """

    num_actions: int = 6

    def __init__(self, params: EnvParams | None = None) -> None:
        super().__init__(params or EnvParams(noop_max=0, max_episode_steps=7680))

    def _reset(self, key: jax.Array) -> BoxingState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BoxingState
            Both boxers at start positions, scores zeroed.
        """
        return BoxingState(
            player_x=jnp.float32(_PLAYER_X0),
            player_y=jnp.float32(_PLAYER_Y0),
            cpu_x=jnp.float32(_CPU_X0),
            cpu_y=jnp.float32(_CPU_Y0),
            cpu_score=jnp.int32(0),
            punch_timer=jnp.int32(0),
            cpu_punch_timer=jnp.int32(0),
            hit_flash=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: BoxingState, action: jax.Array) -> BoxingState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BoxingState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=DOWN, 5=LEFT.

        Returns
        -------
        new_state : BoxingState
            State after one emulated frame.
        """
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
            state.player_x + dx, float(_RING_X0), float(_RING_X1 - _BOXER_W)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_RING_Y0), float(_RING_Y1 - _BOXER_H)
        )

        # CPU AI: move toward player, maintain slight gap
        dist_x = new_px - state.cpu_x
        dist_y = new_py - state.cpu_y
        cpu_dx = jnp.clip(dist_x * 0.05, -_SPEED, _SPEED)
        cpu_dy = jnp.clip(dist_y * 0.05, -_SPEED, _SPEED)
        new_cx = jnp.clip(
            state.cpu_x + cpu_dx, float(_RING_X0), float(_RING_X1 - _BOXER_W)
        )
        new_cy = jnp.clip(
            state.cpu_y + cpu_dy, float(_RING_Y0), float(_RING_Y1 - _BOXER_H)
        )

        # Distance between boxers
        dist = jnp.sqrt((new_px - new_cx) ** 2 + (new_py - new_cy) ** 2)
        in_range = dist <= _PUNCH_RANGE

        # Player punch
        can_punch = (action == jnp.int32(1)) & (state.punch_timer <= jnp.int32(0))
        player_hit = can_punch & in_range
        new_punch_timer = jnp.where(
            can_punch, jnp.int32(_PUNCH_COOLDOWN), state.punch_timer
        )
        new_punch_timer = jnp.where(
            new_punch_timer > jnp.int32(0), new_punch_timer - jnp.int32(1), jnp.int32(0)
        )
        player_pts = jnp.where(player_hit, jnp.int32(1), jnp.int32(0))

        # CPU punch (auto, with cooldown)
        cpu_can_punch = (state.cpu_punch_timer <= jnp.int32(0)) & in_range
        cpu_hit = cpu_can_punch
        new_cpu_punch_timer = jnp.where(
            cpu_can_punch, jnp.int32(_PUNCH_COOLDOWN * 2), state.cpu_punch_timer
        )
        new_cpu_punch_timer = jnp.where(
            new_cpu_punch_timer > jnp.int32(0),
            new_cpu_punch_timer - jnp.int32(1),
            jnp.int32(0),
        )
        cpu_pts = jnp.where(cpu_hit, jnp.int32(1), jnp.int32(0))

        new_score = state.score + player_pts
        new_cpu_score = state.cpu_score + cpu_pts
        step_reward = jnp.float32(player_pts) - jnp.float32(cpu_pts)

        new_hit_flash = jnp.where(player_hit, jnp.int32(3), state.hit_flash)
        new_hit_flash = jnp.where(
            new_hit_flash > jnp.int32(0), new_hit_flash - jnp.int32(1), jnp.int32(0)
        )

        done = (new_score >= jnp.int32(_KO_SCORE)) | (
            new_cpu_score >= jnp.int32(_KO_SCORE)
        )

        return BoxingState(
            player_x=new_px,
            player_y=new_py,
            cpu_x=new_cx,
            cpu_y=new_cy,
            cpu_score=new_cpu_score,
            punch_timer=new_punch_timer,
            cpu_punch_timer=new_cpu_punch_timer,
            hit_flash=new_hit_flash,
            lives=jnp.int32(0),
            score=new_score,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: BoxingState, action: jax.Array) -> BoxingState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BoxingState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : BoxingState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BoxingState) -> BoxingState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BoxingState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BoxingState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)

        # Ring floor
        in_ring = (
            (_ROW_IDX >= _RING_Y0)
            & (_ROW_IDX < _RING_Y1)
            & (_COL_IDX >= _RING_X0)
            & (_COL_IDX < _RING_X1)
        )
        frame = jnp.where(in_ring[:, :, None], _COLOR_RING, frame)

        # Ropes (perimeter lines)
        rope = (
            ((_ROW_IDX == _RING_Y0) | (_ROW_IDX == _RING_Y1 - 1))
            & (_COL_IDX >= _RING_X0)
            & (_COL_IDX < _RING_X1)
        ) | (
            ((_COL_IDX == _RING_X0) | (_COL_IDX == _RING_X1 - 1))
            & (_ROW_IDX >= _RING_Y0)
            & (_ROW_IDX < _RING_Y1)
        )
        frame = jnp.where(rope[:, :, None], _COLOR_ROPE, frame)

        # CPU boxer (top, red)
        cpu_mask = (
            (_ROW_IDX >= jnp.int32(state.cpu_y))
            & (_ROW_IDX < jnp.int32(state.cpu_y) + _BOXER_H)
            & (_COL_IDX >= jnp.int32(state.cpu_x))
            & (_COL_IDX < jnp.int32(state.cpu_x) + _BOXER_W)
        )
        frame = jnp.where(cpu_mask[:, :, None], _COLOR_CPU, frame)

        # Player boxer (bottom, white)
        p_mask = (
            (_ROW_IDX >= jnp.int32(state.player_y))
            & (_ROW_IDX < jnp.int32(state.player_y) + _BOXER_H)
            & (_COL_IDX >= jnp.int32(state.player_x))
            & (_COL_IDX < jnp.int32(state.player_x) + _BOXER_W)
        )
        p_color = jnp.where(state.hit_flash > jnp.int32(0), _COLOR_FLASH, _COLOR_PLAYER)
        frame = jnp.where(p_mask[:, :, None], p_color, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Boxing action indices.
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
