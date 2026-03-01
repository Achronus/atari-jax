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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Ring        : x ∈ [8, 152),  y ∈ [50, 185)
    Player      : 8×12 px, starts x=64 y=150
    CPU         : 8×12 px, starts x=88 y=70

Action space (18 actions — ALE minimal set):
    0  NOOP
    1  FIRE  (punch)
    2  UP
    3  RIGHT
    4  LEFT
    5  DOWN
    6  UPRIGHT    7  UPLEFT    8  DOWNRIGHT   9  DOWNLEFT
    10 UPFIRE     11 RIGHTFIRE  12 LEFTFIRE   13 DOWNFIRE
    14 UPRIGHTFIRE 15 UPLEFTFIRE 16 DOWNRIGHTFIRE 17 DOWNLEFTFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Ring geometry
_RING_LEFT: int = 8
_RING_RIGHT: int = 152
_RING_TOP: int = 50
_RING_BOTTOM: int = 185

# Boxer dimensions
_BOXER_W: int = 8
_BOXER_H: int = 12

# Start positions
_PLAYER_START_X: float = 64.0
_PLAYER_START_Y: float = 150.0
_CPU_START_X: float = 88.0
_CPU_START_Y: float = 70.0

# Movement
_PLAYER_SPEED: float = 1.0
_CPU_SPEED: float = 1.0

# Punch mechanics
_PUNCH_RANGE: float = 16.0      # Manhattan distance threshold for a hit
_PUNCH_COOLDOWN: int = 12       # frames until player can punch again
_CPU_PUNCH_COOLDOWN: int = 15   # frames until CPU can punch again
_KO_SCORE: int = 100            # knockout threshold

_FRAME_SKIP: int = 4

# Colours
_RING_COLOR = jnp.array([144, 72, 17], dtype=jnp.uint8)    # brown ring floor
_PLAYER_COLOR = jnp.array([92, 186, 92], dtype=jnp.uint8)  # green player
_CPU_COLOR = jnp.array([213, 130, 74], dtype=jnp.uint8)    # orange CPU
_SCORE_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

# 3×5 bitmap font for digits 0–9.  Shape: [10, 5, 3].
_DIGIT_FONT = jnp.array(
    [
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],  # 0
        [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 1
        [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],  # 2
        [[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],  # 3
        [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],  # 4
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 5
        [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 6
        [[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],  # 7
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],  # 8
        [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]],  # 9
    ],
    dtype=jnp.bool_,
)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]


def _blit_digit(frame: jax.Array, digit: jax.Array, x0: int, y0: int) -> jax.Array:
    """Blit a 3×5 digit glyph onto *frame* at pixel (x0, y0)."""
    glyph = _DIGIT_FONT[digit]
    dr = jnp.clip(_ROW_IDX - y0, 0, 4)
    dc = jnp.clip(_COL_IDX - x0, 0, 2)
    in_box = (
        (_ROW_IDX >= y0) & (_ROW_IDX < y0 + 5) & (_COL_IDX >= x0) & (_COL_IDX < x0 + 3)
    )
    lit = glyph[dr, dc]
    return jnp.where((in_box & lit)[:, :, None], _SCORE_COLOR, frame)


@chex.dataclass
class BoxingState(AtariState):
    """
    Complete Boxing game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `score` tracks player's landed punches; `cpu_hits` tracks CPU's landed punches.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player left-edge x coordinate.
    player_y : jax.Array
        float32 — Player top-edge y coordinate.
    cpu_x : jax.Array
        float32 — CPU left-edge x coordinate.
    cpu_y : jax.Array
        float32 — CPU top-edge y coordinate.
    cpu_hits : jax.Array
        int32 — Punches landed by the CPU.
    punch_timer : jax.Array
        int32 — Frames remaining before player can punch again (0 = ready).
    cpu_punch_timer : jax.Array
        int32 — Frames remaining before CPU can punch again (0 = ready).
    """

    player_x: jax.Array
    player_y: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    cpu_hits: jax.Array
    punch_timer: jax.Array
    cpu_punch_timer: jax.Array


def _decode_action(action: jax.Array):
    """Decode an 18-action ALE index into movement and fire flags."""
    has_fire = (
        (action == 1) | (action == 10) | (action == 11) | (action == 12) | (action == 13)
        | (action == 14) | (action == 15) | (action == 16) | (action == 17)
    )
    move_up = (action == 2) | (action == 6) | (action == 7) | (action == 10) | (action == 14) | (action == 15)
    move_right = (action == 3) | (action == 6) | (action == 8) | (action == 11) | (action == 14) | (action == 16)
    move_left = (action == 4) | (action == 7) | (action == 9) | (action == 12) | (action == 15) | (action == 17)
    move_down = (action == 5) | (action == 8) | (action == 9) | (action == 13) | (action == 16) | (action == 17)
    return move_up, move_right, move_down, move_left, has_fire


class Boxing(AtaraxGame):
    """
    Boxing implemented as a pure-JAX function suite.

    Physics: both boxers can move freely within the ring. A punch lands
    when the opponent is within `_PUNCH_RANGE` (Manhattan distance) and
    the boxer isn't in cooldown. Episode ends on knockout (100 hits) or
    the step limit from `AtaraxParams`.
    """

    num_actions: int = 18

    def _reset(self, key: chex.PRNGKey) -> BoxingState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : BoxingState
            Boxers at start positions, timers zeroed.
        """
        return BoxingState(
            player_x=jnp.float32(_PLAYER_START_X),
            player_y=jnp.float32(_PLAYER_START_Y),
            cpu_x=jnp.float32(_CPU_START_X),
            cpu_y=jnp.float32(_CPU_START_Y),
            cpu_hits=jnp.int32(0),
            punch_timer=jnp.int32(0),
            cpu_punch_timer=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: BoxingState, action: jax.Array) -> BoxingState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : BoxingState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).

        Returns
        -------
        new_state : BoxingState
            State after one emulated frame.
        """
        move_up, move_right, move_down, move_left, has_fire = _decode_action(action)

        # --- Player movement (clamped to ring) ---
        dx = jnp.where(move_right, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0)) - jnp.where(move_left, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0))
        dy = jnp.where(move_down, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0)) - jnp.where(move_up, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0))
        player_x = jnp.clip(state.player_x + dx, jnp.float32(_RING_LEFT), jnp.float32(_RING_RIGHT - _BOXER_W))
        player_y = jnp.clip(state.player_y + dy, jnp.float32(_RING_TOP), jnp.float32(_RING_BOTTOM - _BOXER_H))

        # --- CPU AI: move toward player ---
        cpu_dx = jnp.clip(state.player_x - state.cpu_x, -_CPU_SPEED, _CPU_SPEED)
        cpu_dy = jnp.clip(state.player_y - state.cpu_y, -_CPU_SPEED, _CPU_SPEED)
        cpu_x = jnp.clip(state.cpu_x + cpu_dx, jnp.float32(_RING_LEFT), jnp.float32(_RING_RIGHT - _BOXER_W))
        cpu_y = jnp.clip(state.cpu_y + cpu_dy, jnp.float32(_RING_TOP), jnp.float32(_RING_BOTTOM - _BOXER_H))

        # --- Manhattan distance between boxer centres ---
        player_cx = player_x + jnp.float32(_BOXER_W) / 2.0
        player_cy = player_y + jnp.float32(_BOXER_H) / 2.0
        cpu_cx = cpu_x + jnp.float32(_BOXER_W) / 2.0
        cpu_cy = cpu_y + jnp.float32(_BOXER_H) / 2.0
        manhattan = jnp.abs(player_cx - cpu_cx) + jnp.abs(player_cy - cpu_cy)
        in_range = manhattan < jnp.float32(_PUNCH_RANGE)

        # --- Player punch ---
        punch_timer_ready = state.punch_timer == jnp.int32(0)
        player_lands = has_fire & in_range & punch_timer_ready
        new_score = state.score + jnp.where(player_lands, jnp.int32(1), jnp.int32(0))
        new_punch_timer = jnp.where(
            player_lands,
            jnp.int32(_PUNCH_COOLDOWN),
            jnp.maximum(state.punch_timer - jnp.int32(1), jnp.int32(0)),
        )

        # --- CPU punch ---
        cpu_timer_ready = state.cpu_punch_timer == jnp.int32(0)
        cpu_lands = in_range & cpu_timer_ready
        new_cpu_hits = state.cpu_hits + jnp.where(cpu_lands, jnp.int32(1), jnp.int32(0))
        new_cpu_punch_timer = jnp.where(
            cpu_lands,
            jnp.int32(_CPU_PUNCH_COOLDOWN),
            jnp.maximum(state.cpu_punch_timer - jnp.int32(1), jnp.int32(0)),
        )

        # Reward = net score change this frame
        step_reward = (
            jnp.where(player_lands, jnp.float32(1.0), jnp.float32(0.0))
            - jnp.where(cpu_lands, jnp.float32(1.0), jnp.float32(0.0))
        )

        # Episode end on knockout
        done = (new_score >= jnp.int32(_KO_SCORE)) | (new_cpu_hits >= jnp.int32(_KO_SCORE))

        return BoxingState(
            player_x=player_x,
            player_y=player_y,
            cpu_x=cpu_x,
            cpu_y=cpu_y,
            cpu_hits=new_cpu_hits,
            punch_timer=new_punch_timer,
            cpu_punch_timer=new_cpu_punch_timer,
            lives=jnp.int32(0),
            score=new_score,
            level=state.level,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step,  # incremented once per agent step in _step
            key=state.key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: BoxingState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> BoxingState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : BoxingState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).
        params : AtaraxParams
            Static environment parameters.

        Returns
        -------
        new_state : BoxingState
            State after 4 emulated frames with `episode_step` incremented once.
        """
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

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
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Ring floor
        ring_mask = (
            (_ROW_IDX >= _RING_TOP) & (_ROW_IDX < _RING_BOTTOM)
            & (_COL_IDX >= _RING_LEFT) & (_COL_IDX < _RING_RIGHT)
        )
        frame = jnp.where(ring_mask[:, :, None], _RING_COLOR, frame)

        # CPU boxer
        cpu_mask = (
            (_ROW_IDX >= jnp.int32(state.cpu_y)) & (_ROW_IDX < jnp.int32(state.cpu_y) + _BOXER_H)
            & (_COL_IDX >= jnp.int32(state.cpu_x)) & (_COL_IDX < jnp.int32(state.cpu_x) + _BOXER_W)
        )
        frame = jnp.where(cpu_mask[:, :, None], _CPU_COLOR, frame)

        # Player boxer
        player_mask = (
            (_ROW_IDX >= jnp.int32(state.player_y)) & (_ROW_IDX < jnp.int32(state.player_y) + _BOXER_H)
            & (_COL_IDX >= jnp.int32(state.player_x)) & (_COL_IDX < jnp.int32(state.player_x) + _BOXER_W)
        )
        frame = jnp.where(player_mask[:, :, None], _PLAYER_COLOR, frame)

        # Score: player hits (left) | CPU hits (right)
        score = state.score
        cpu_score = state.cpu_hits
        frame = _blit_digit(frame, (score // 10) % 10, x0=8, y0=4)
        frame = _blit_digit(frame, score % 10, x0=12, y0=4)
        frame = _blit_digit(frame, (cpu_score // 10) % 10, x0=140, y0=4)
        frame = _blit_digit(frame, cpu_score % 10, x0=144, y0=4)

        return frame

    def _key_map(self) -> dict:
        """Return the key-to-action mapping for interactive play."""
        import pygame

        return {
            pygame.K_SPACE: 1,   # FIRE — punch
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 4,
            pygame.K_a: 4,
            pygame.K_DOWN: 5,
            pygame.K_s: 5,
        }
