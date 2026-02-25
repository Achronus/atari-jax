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

"""Skiing — JAX-native game implementation.

Slalom ski down a mountain, passing through gates as fast as possible.
Miss a gate and get a time penalty.  Lowest time wins.

Action space (5 actions):
    0 — NOOP
    1 — RIGHT (lean right)
    2 — LEFT  (lean left)
    3 — HARD RIGHT (sharp turn)
    4 — HARD LEFT  (sharp turn)

Scoring:
    Score is negative elapsed time (lower is better).
    Penalty: +5 s per missed gate.
    Episode ends when all gates are passed.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_GATES: int = 20
_PLAYER_X_INIT: float = 80.0
_PLAYER_Y: int = 50  # fixed screen position; mountain scrolls
_SCROLL_SPEED_BASE: float = 2.0
_PLAYER_SPEED_MAX: float = 3.0

_GATE_W: int = 20  # opening width
_GATE_POST_W: int = 4
_GATE_H: int = 6

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_SNOW = jnp.array([230, 230, 255], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 80, 80], dtype=jnp.uint8)
_COLOR_GATE = jnp.array([255, 50, 50], dtype=jnp.uint8)
_COLOR_POST = jnp.array([180, 40, 40], dtype=jnp.uint8)


@chex.dataclass
class SkiingState(AtariState):
    """
    Complete Skiing game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x position.
    player_vx : jax.Array
        float32 — Player horizontal velocity.
    scroll_y : jax.Array
        float32 — World scroll offset (distance covered).
    gate_y : jax.Array
        float32[20] — Gate y positions (world coords).
    gate_x : jax.Array
        float32[20] — Gate centre x positions.
    gate_passed : jax.Array
        bool[20] — Gate passed (correctly or as penalty).
    elapsed_time : jax.Array
        int32 — Elapsed frames.
    penalties : jax.Array
        int32 — Number of missed gates.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_vx: jax.Array
    scroll_y: jax.Array
    gate_y: jax.Array
    gate_x: jax.Array
    gate_passed: jax.Array
    elapsed_time: jax.Array
    penalties: jax.Array
    key: jax.Array


class Skiing(AtariEnv):
    """
    Skiing implemented as a pure JAX function suite.

    Pass all gates as fast as possible.  No lives — time-attack game.
    """

    num_actions: int = 5

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=10_000)

    def _reset(self, key: jax.Array) -> SkiingState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : SkiingState
            Player at centre top, gates spread down the slope.
        """
        gate_ys = jnp.linspace(300.0, 5800.0, _N_GATES, dtype=jnp.float32)
        gate_xs = jnp.full(_N_GATES, 80.0, dtype=jnp.float32)
        return SkiingState(
            player_x=jnp.float32(_PLAYER_X_INIT),
            player_vx=jnp.float32(0.0),
            scroll_y=jnp.float32(0.0),
            gate_y=gate_ys,
            gate_x=gate_xs,
            gate_passed=jnp.zeros(_N_GATES, dtype=jnp.bool_),
            elapsed_time=jnp.int32(0),
            penalties=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: SkiingState, action: jax.Array) -> SkiingState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : SkiingState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : SkiingState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Horizontal control
        accel = jnp.where(
            action == 1,
            jnp.float32(0.2),
            jnp.where(
                action == 2,
                jnp.float32(-0.2),
                jnp.where(
                    action == 3,
                    jnp.float32(0.5),
                    jnp.where(action == 4, jnp.float32(-0.5), jnp.float32(0.0)),
                ),
            ),
        )
        new_vx = jnp.clip(
            state.player_vx + accel, -_PLAYER_SPEED_MAX, _PLAYER_SPEED_MAX
        )
        # Friction
        new_vx = new_vx * jnp.float32(0.92)
        new_px = jnp.clip(state.player_x + new_vx, jnp.float32(5.0), jnp.float32(155.0))

        # Scroll
        new_scroll = state.scroll_y + _SCROLL_SPEED_BASE

        # Gate screen positions
        gate_screen_y = state.gate_y - new_scroll + jnp.float32(float(_PLAYER_Y))

        # Check if player passes through gate
        at_gate_y = (gate_screen_y >= jnp.float32(_PLAYER_Y - 2)) & (
            gate_screen_y <= jnp.float32(_PLAYER_Y + 8)
        )
        in_gate_x = jnp.abs(new_px - state.gate_x) <= jnp.float32(_GATE_W // 2)
        passes_gate = at_gate_y & in_gate_x & ~state.gate_passed
        misses_gate = at_gate_y & ~in_gate_x & ~state.gate_passed

        new_gate_passed = state.gate_passed | passes_gate | misses_gate
        new_penalties = state.penalties + jnp.sum(misses_gate, dtype=jnp.int32)

        # Reward: -1 per frame, -300 per missed gate
        step_reward = jnp.float32(-1.0) - jnp.sum(misses_gate).astype(
            jnp.float32
        ) * jnp.float32(300.0)

        new_elapsed = state.elapsed_time + jnp.int32(1)

        # Done when all gates passed
        done = jnp.all(new_gate_passed)

        return SkiingState(
            player_x=new_px,
            player_vx=new_vx,
            scroll_y=new_scroll,
            gate_y=state.gate_y,
            gate_x=state.gate_x,
            gate_passed=new_gate_passed,
            elapsed_time=new_elapsed,
            penalties=new_penalties,
            key=key,
            lives=jnp.int32(0),
            score=state.score
            + jnp.int32(jnp.maximum(step_reward, jnp.float32(-300.0))),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: SkiingState, action: jax.Array) -> SkiingState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : SkiingState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : SkiingState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: SkiingState) -> SkiingState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: SkiingState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : SkiingState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_SNOW, dtype=jnp.uint8)

        # Gates
        def draw_gate(frm, i):
            gx = state.gate_x[i].astype(jnp.int32)
            gy = (state.gate_y[i] - state.scroll_y + jnp.float32(_PLAYER_Y)).astype(
                jnp.int32
            )
            left_post = (
                ~state.gate_passed[i]
                & (_ROW_IDX >= gy)
                & (_ROW_IDX < gy + _GATE_H)
                & (_COL_IDX >= gx - _GATE_W // 2 - _GATE_POST_W)
                & (_COL_IDX < gx - _GATE_W // 2)
            )
            right_post = (
                ~state.gate_passed[i]
                & (_ROW_IDX >= gy)
                & (_ROW_IDX < gy + _GATE_H)
                & (_COL_IDX >= gx + _GATE_W // 2)
                & (_COL_IDX < gx + _GATE_W // 2 + _GATE_POST_W)
            )
            frm = jnp.where(left_post[:, :, None], _COLOR_POST, frm)
            frm = jnp.where(right_post[:, :, None], _COLOR_POST, frm)
            return frm, None

        frame, _ = jax.lax.scan(draw_gate, frame, jnp.arange(_N_GATES))

        # Player
        px = state.player_x.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= _PLAYER_Y - 6)
            & (_ROW_IDX < _PLAYER_Y + 8)
            & (_COL_IDX >= px - 4)
            & (_COL_IDX < px + 4)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Skiing action indices.
        """
        import pygame

        return {
            pygame.K_RIGHT: 1,
            pygame.K_d: 1,
            pygame.K_LEFT: 2,
            pygame.K_a: 2,
        }
