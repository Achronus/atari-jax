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

"""Fishing Derby — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Dock       : y = 50
    Water      : y ∈ [70, 185]
    Player     : dock x movable ∈ [10, 50], centred at 20
    CPU        : dock x movable ∈ [110, 150], centred at 140
    Shark      : 16×8 px, patrols y=130
    Fish       : 5 fish at various depths (y ∈ [90, 160])

Action space (18 actions — ALE minimal set):
     0  NOOP
     1  FIRE
     2  UP
     3  RIGHT
     4  LEFT
     5  DOWN
     6  UPRIGHT
     7  UPLEFT
     8  DOWNRIGHT
     9  DOWNLEFT
    10  UPFIRE
    11  RIGHTFIRE
    12  LEFTFIRE
    13  DOWNFIRE
    14  UPRIGHTFIRE
    15  UPLEFTFIRE
    16  DOWNRIGHTFIRE
    17  DOWNLEFTFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry
_DOCK_Y: int = 50
_WATER_Y0: int = 70
_WATER_Y1: int = 185

_PLAYER_DOCK_LEFT: float = 10.0
_PLAYER_DOCK_RIGHT: float = 50.0
_PLAYER_DOCK_START: float = 20.0

_CPU_DOCK_LEFT: float = 110.0
_CPU_DOCK_RIGHT: float = 150.0
_CPU_DOCK_START: float = 140.0

_SHARK_Y: int = 130
_SHARK_W: int = 16
_SHARK_H: int = 8
_SHARK_LEFT: float = 0.0
_SHARK_RIGHT: float = 144.0  # 160 - SHARK_W
_SHARK_SPEED: float = 1.5

# Fish
_N_FISH: int = 5

# Initial fish x positions (spread across full water width; CPU side accessible)
_FISH_X0 = jnp.array([20.0, 45.0, 80.0, 115.0, 140.0], dtype=jnp.float32)

# Initial fish y (depths): two shallow, two mid, one deep
_FISH_Y0 = jnp.array([90.0, 95.0, 120.0, 125.0, 150.0], dtype=jnp.float32)

# Point values by fish index (matching depth tier)
_FISH_VALUES = jnp.array([2, 2, 4, 4, 6], dtype=jnp.int32)

# Fish drift speed and bounds
_FISH_SPEED: float = 0.3
_FISH_DRIFT = jnp.array([0.3, -0.3, 0.3, -0.3, 0.3], dtype=jnp.float32)  # alternating
_FISH_X_LEFT: float = 5.0
_FISH_X_RIGHT: float = 155.0

# Catch threshold (pixels)
_CATCH_DX: float = 10.0  # horizontal tolerance
_CATCH_DY: float = 5.0  # vertical tolerance

_LINE_W: int = 1
_LINE_H: int = 4  # hook size

# Player / CPU
_DOCK_SPEED: float = 1.0
_LINE_SPEED: float = 1.0
_CPU_SPEED: float = 1.5  # CPU dock movement speed (faster than player)
_CPU_LINE_SPEED: float = 1.5  # CPU line depth speed

# Episode
_MAX_STEPS: int = 2400  # agent steps (~9600 emulated frames / 4-frame skip)
_SCORE_LIMIT: int = 99
_FRAME_SKIP: int = 4

# Render colours
_WATER_COLOR = jnp.array([0, 40, 120], dtype=jnp.uint8)
_DOCK_COLOR = jnp.array([139, 90, 43], dtype=jnp.uint8)
_SHARK_COLOR = jnp.array([150, 150, 150], dtype=jnp.uint8)
_FISH_SHALLOW_COLOR = jnp.array([255, 220, 0], dtype=jnp.uint8)
_FISH_MID_COLOR = jnp.array([255, 140, 0], dtype=jnp.uint8)
_FISH_DEEP_COLOR = jnp.array([255, 60, 60], dtype=jnp.uint8)
_PLAYER_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_CPU_COLOR = jnp.array([200, 100, 100], dtype=jnp.uint8)
_LINE_COLOR = jnp.array([200, 200, 200], dtype=jnp.uint8)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]


@chex.dataclass
class FishingDerbyState(AtariState):
    """
    Complete Fishing Derby game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `score` = player cumulative score; `cpu_score` = CPU cumulative score.
    `lives` is always 0 (no lives system).

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player dock x ∈ [10, 50].
    player_line_y : jax.Array
        float32 — Player line depth ∈ [70, 185].
    cpu_x : jax.Array
        float32 — CPU dock x ∈ [110, 150].
    cpu_line_y : jax.Array
        float32 — CPU line depth ∈ [70, 185].
    fish_x : jax.Array
        float32[5] — Fish x positions.
    fish_y : jax.Array
        float32[5] — Fish y depths.
    fish_active : jax.Array
        bool[5] — Fish present in water.
    fish_dx : jax.Array
        float32[5] — Fish drift velocities.
    shark_x : jax.Array
        float32 — Shark x at fixed y=130.
    shark_dx : jax.Array
        float32 — Shark direction speed (signed).
    cpu_score : jax.Array
        int32 — CPU cumulative score.
    """

    player_x: chex.Array
    player_line_y: chex.Array
    cpu_x: chex.Array
    cpu_line_y: chex.Array
    fish_x: chex.Array
    fish_y: chex.Array
    fish_active: chex.Array
    fish_dx: chex.Array
    shark_x: chex.Array
    shark_dx: chex.Array
    cpu_score: chex.Array


class FishingDerby(AtaraxGame):
    """
    Fishing Derby implemented as a pure-JAX function suite.

    Player and CPU compete to catch fish worth 2, 4, or 6 points (by depth).
    A shark patrols y=130 and can snap the player's line, resetting its depth.
    Episode ends when either angler reaches 99 points or max_steps is reached.
    Reward = change in (player_score - cpu_score) per step.
    """

    num_actions: int = 18

    def _reset(self, key: chex.PRNGKey) -> FishingDerbyState:
        """Return the canonical initial game state."""
        return FishingDerbyState(
            player_x=jnp.float32(_PLAYER_DOCK_START),
            player_line_y=jnp.float32(_WATER_Y0),
            cpu_x=jnp.float32(_CPU_DOCK_START),
            cpu_line_y=jnp.float32(_WATER_Y0),
            fish_x=_FISH_X0.copy(),
            fish_y=_FISH_Y0.copy(),
            fish_active=jnp.ones(_N_FISH, dtype=jnp.bool_),
            fish_dx=_FISH_DRIFT.copy(),
            shark_x=jnp.float32(80.0),
            shark_dx=jnp.float32(_SHARK_SPEED),
            cpu_score=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: FishingDerbyState, action: jax.Array
    ) -> FishingDerbyState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : FishingDerbyState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).

        Returns
        -------
        new_state : FishingDerbyState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Action decode ---
        move_up = (
            (action == 2)
            | (action == 6)
            | (action == 7)
            | (action == 10)
            | (action == 14)
            | (action == 15)
        )
        move_down = (
            (action == 5)
            | (action == 8)
            | (action == 9)
            | (action == 12)
            | (action == 16)
            | (action == 17)
        )
        move_right = (
            (action == 3)
            | (action == 6)
            | (action == 8)
            | (action == 11)
            | (action == 14)
            | (action == 16)
        )
        move_left = (
            (action == 4)
            | (action == 7)
            | (action == 9)
            | (action == 13)
            | (action == 15)
            | (action == 17)
        )

        # --- Player dock and line movement ---
        dx = jnp.where(
            move_right,
            jnp.float32(_DOCK_SPEED),
            jnp.where(move_left, jnp.float32(-_DOCK_SPEED), jnp.float32(0.0)),
        )
        dy = jnp.where(
            move_down,
            jnp.float32(_LINE_SPEED),
            jnp.where(move_up, jnp.float32(-_LINE_SPEED), jnp.float32(0.0)),
        )

        player_x = jnp.clip(state.player_x + dx, _PLAYER_DOCK_LEFT, _PLAYER_DOCK_RIGHT)
        player_line_y = jnp.clip(
            state.player_line_y + dy, jnp.float32(_WATER_Y0), jnp.float32(_WATER_Y1)
        )

        # --- Fish drift ---
        new_fish_x = state.fish_x + state.fish_dx
        # Bounce off walls
        hit_left = new_fish_x < jnp.float32(_FISH_X_LEFT)
        hit_right = new_fish_x > jnp.float32(_FISH_X_RIGHT)
        new_fish_dx = jnp.where(hit_left | hit_right, -state.fish_dx, state.fish_dx)
        new_fish_x = jnp.clip(
            new_fish_x, jnp.float32(_FISH_X_LEFT), jnp.float32(_FISH_X_RIGHT)
        )

        # --- Shark movement ---
        new_shark_x = state.shark_x + state.shark_dx
        shark_hit_left = new_shark_x < jnp.float32(_SHARK_LEFT)
        shark_hit_right = new_shark_x > jnp.float32(_SHARK_RIGHT)
        new_shark_dx = jnp.where(
            shark_hit_left | shark_hit_right, -state.shark_dx, state.shark_dx
        )
        new_shark_x = jnp.clip(
            new_shark_x, jnp.float32(_SHARK_LEFT), jnp.float32(_SHARK_RIGHT)
        )

        # --- Shark cuts player line ---
        # Shark AABB: (shark_x, SHARK_Y) to (shark_x+SHARK_W, SHARK_Y+SHARK_H)
        # Line point: (player_x, player_line_y)
        shark_cuts_player = (
            (player_x >= new_shark_x)
            & (player_x < new_shark_x + jnp.float32(_SHARK_W))
            & (player_line_y >= jnp.float32(_SHARK_Y))
            & (player_line_y < jnp.float32(_SHARK_Y + _SHARK_H))
        )
        player_line_y = jnp.where(
            shark_cuts_player, jnp.float32(_WATER_Y0), player_line_y
        )

        # --- Player catch check ---
        catch_dist_x = jnp.abs(player_x - new_fish_x)
        catch_dist_y = jnp.abs(player_line_y - state.fish_y)
        caught_mask = (
            state.fish_active
            & (catch_dist_x < jnp.float32(_CATCH_DX))
            & (catch_dist_y < jnp.float32(_CATCH_DY))
        )
        any_caught = jnp.any(caught_mask)
        caught_idx = jnp.argmax(caught_mask)
        player_pts = jnp.where(any_caught, _FISH_VALUES[caught_idx], jnp.int32(0))
        new_score = state.score + player_pts

        # --- CPU AI ---
        # CPU targets the fish with the best value-to-distance ratio so it
        # prioritises reachable high-value fish rather than always chasing the
        # deepest one regardless of position.
        cpu_dist_to_fish = jnp.abs(state.cpu_x - new_fish_x) + jnp.float32(1.0)
        reach_score = (
            _FISH_VALUES.astype(jnp.float32)
            * state.fish_active.astype(jnp.float32)
            / cpu_dist_to_fish
        )
        best_idx = jnp.argmax(reach_score)
        target_fish_x = new_fish_x[best_idx]
        target_fish_y = state.fish_y[best_idx]
        target_active = state.fish_active[best_idx]

        cpu_dx = jnp.where(
            target_active & (state.cpu_x < target_fish_x),
            jnp.float32(_CPU_SPEED),
            jnp.where(
                target_active & (state.cpu_x > target_fish_x),
                jnp.float32(-_CPU_SPEED),
                jnp.float32(0.0),
            ),
        )
        cpu_x = jnp.clip(state.cpu_x + cpu_dx, _CPU_DOCK_LEFT, _CPU_DOCK_RIGHT)

        # CPU lowers line toward target fish depth
        cpu_dy = jnp.where(
            target_active & (state.cpu_line_y < target_fish_y - jnp.float32(_CATCH_DY)),
            jnp.float32(_CPU_LINE_SPEED),
            jnp.where(
                target_active
                & (state.cpu_line_y > target_fish_y + jnp.float32(_CATCH_DY)),
                jnp.float32(-_CPU_LINE_SPEED),
                jnp.float32(0.0),
            ),
        )
        cpu_line_y = jnp.clip(
            state.cpu_line_y + cpu_dy, jnp.float32(_WATER_Y0), jnp.float32(_WATER_Y1)
        )

        # --- CPU catch check ---
        cpu_catch_dist_x = jnp.abs(cpu_x - new_fish_x)
        cpu_catch_dist_y = jnp.abs(cpu_line_y - state.fish_y)
        cpu_caught_mask = (
            state.fish_active
            & ~caught_mask  # player catch takes priority
            & (cpu_catch_dist_x < jnp.float32(_CATCH_DX))
            & (cpu_catch_dist_y < jnp.float32(_CATCH_DY))
        )
        any_cpu_caught = jnp.any(cpu_caught_mask)
        cpu_caught_idx = jnp.argmax(cpu_caught_mask)
        cpu_pts = jnp.where(any_cpu_caught, _FISH_VALUES[cpu_caught_idx], jnp.int32(0))
        new_cpu_score = state.cpu_score + cpu_pts

        # --- Respawn caught fish at random depth ---
        respawn_mask = caught_mask | cpu_caught_mask
        # New y: random within [90, 160]
        rand_y = jax.random.uniform(subkey, (_N_FISH,), minval=90.0, maxval=160.0)
        new_fish_y = jnp.where(respawn_mask, rand_y.astype(jnp.float32), state.fish_y)
        # Deactivate on catch, reactivate (respawn) immediately
        new_fish_active = state.fish_active  # fish always active (respawn instantly)

        # --- Reward ---
        reward = (new_score - state.score).astype(jnp.float32) - (
            new_cpu_score - state.cpu_score
        ).astype(jnp.float32)

        # --- Episode end ---
        done = (new_score >= jnp.int32(_SCORE_LIMIT)) | (
            new_cpu_score >= jnp.int32(_SCORE_LIMIT)
        )

        return state.__replace__(
            player_x=player_x,
            player_line_y=player_line_y,
            cpu_x=cpu_x,
            cpu_line_y=cpu_line_y,
            fish_x=new_fish_x,
            fish_y=new_fish_y,
            fish_active=new_fish_active,
            fish_dx=new_fish_dx,
            shark_x=new_shark_x,
            shark_dx=new_shark_dx,
            cpu_score=new_cpu_score,
            score=new_score,
            reward=state.reward + reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: FishingDerbyState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> FishingDerbyState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        new_ep = state.episode_step + jnp.int32(1)
        # Episode ends at max_steps (time limit)
        done = new_state.done | (new_ep >= jnp.int32(_MAX_STEPS))
        return new_state.__replace__(episode_step=new_ep, done=done)

    def render(self, state: FishingDerbyState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Water ---
        water_mask = (_ROW_IDX >= _WATER_Y0) & (_ROW_IDX < _WATER_Y1)
        frame = jnp.where(water_mask[:, :, None], _WATER_COLOR[None, None, :], frame)

        # --- Dock ---
        dock_mask = (_ROW_IDX == _DOCK_Y) & (_COL_IDX < 160)
        frame = jnp.where(dock_mask[:, :, None], _DOCK_COLOR[None, None, :], frame)

        # --- Shark ---
        sx = jnp.int32(state.shark_x)
        shark_mask = (
            (_ROW_IDX >= _SHARK_Y)
            & (_ROW_IDX < _SHARK_Y + _SHARK_H)
            & (_COL_IDX >= sx)
            & (_COL_IDX < sx + _SHARK_W)
        )
        frame = jnp.where(shark_mask[:, :, None], _SHARK_COLOR[None, None, :], frame)

        # --- Fish ---
        _FISH_SIZE = 6
        for i in range(_N_FISH):
            fx = jnp.int32(state.fish_x[i])
            fy = jnp.int32(state.fish_y[i])
            color = jnp.where(
                i < 2,
                _FISH_SHALLOW_COLOR,
                jnp.where(i < 4, _FISH_MID_COLOR, _FISH_DEEP_COLOR),
            )
            fish_mask = (
                state.fish_active[i]
                & (_ROW_IDX >= fy)
                & (_ROW_IDX < fy + _FISH_SIZE)
                & (_COL_IDX >= fx)
                & (_COL_IDX < fx + _FISH_SIZE)
            )
            frame = jnp.where(fish_mask[:, :, None], color[None, None, :], frame)

        # --- Player angler ---
        px = jnp.int32(state.player_x)
        player_mask = (
            (_ROW_IDX >= _DOCK_Y - 6)
            & (_ROW_IDX < _DOCK_Y)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 4)
        )
        frame = jnp.where(player_mask[:, :, None], _PLAYER_COLOR[None, None, :], frame)

        # --- Player fishing line ---
        ply = jnp.int32(state.player_line_y)
        line_mask = (_ROW_IDX >= _DOCK_Y) & (_ROW_IDX <= ply) & (_COL_IDX == px + 2)
        frame = jnp.where(line_mask[:, :, None], _LINE_COLOR[None, None, :], frame)

        # --- CPU angler ---
        cx = jnp.int32(state.cpu_x)
        cpu_mask = (
            (_ROW_IDX >= _DOCK_Y - 6)
            & (_ROW_IDX < _DOCK_Y)
            & (_COL_IDX >= cx)
            & (_COL_IDX < cx + 4)
        )
        frame = jnp.where(cpu_mask[:, :, None], _CPU_COLOR[None, None, :], frame)

        # --- CPU fishing line ---
        cly = jnp.int32(state.cpu_line_y)
        cpu_line_mask = (_ROW_IDX >= _DOCK_Y) & (_ROW_IDX <= cly) & (_COL_IDX == cx + 2)
        frame = jnp.where(cpu_line_mask[:, :, None], _LINE_COLOR[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,  # FIRE
                pygame.K_UP: 2,
                pygame.K_w: 2,
                pygame.K_RIGHT: 3,
                pygame.K_d: 3,
                pygame.K_LEFT: 4,
                pygame.K_a: 4,
                pygame.K_DOWN: 5,
                pygame.K_s: 5,
            }
        except ImportError:
            return {}
