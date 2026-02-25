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

Compete against a CPU angler to catch the most valuable fish before either
player reaches 99 points.  A shark patrols the middle depths and can snap
the line, forcing a recast.

Action space (6 actions):
    0 — NOOP
    1 — FIRE  (reel in / cast)
    2 — UP    (raise line / retract)
    3 — RIGHT (move along dock)
    4 — DOWN  (lower line)
    5 — LEFT  (move along dock)

Scoring:
    Catch fish at depth 1 (shallow) — +2
    Catch fish at depth 2           — +4
    Catch fish at depth 3 (deep)    — +6
    Opponent catches fish           — opponent gains same
    Episode ends when either score ≥ 99.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_DOCK_Y: int = 50  # y of the dock / angler row
_WATER_TOP: int = 70  # y where water begins
_WATER_BOT: int = 185  # y where water ends
_WATER_H: int = _WATER_BOT - _WATER_TOP

_N_FISH: int = 8  # fish per player (total 16 in pond)
_FISH_W: int = 8
_FISH_H: int = 4

# Fish swim at discrete depth bands
_DEPTH_YS = jnp.array(
    [90.0, 120.0, 150.0, 90.0, 110.0, 140.0, 160.0, 130.0],
    dtype=jnp.float32,
)  # y positions of each fish (8 fish)
_FISH_POINTS = jnp.array(
    [2, 4, 6, 2, 4, 6, 4, 2], dtype=jnp.int32
)  # points per fish by depth

_SHARK_Y: float = 130.0
_SHARK_SPEED: float = 1.0
_SHARK_W: int = 16
_SHARK_H: int = 8

_LINE_SPEED: float = 1.5
_LINE_X_P: int = 20  # player dock x
_LINE_X_CPU: int = 140  # CPU dock x

_WIN_SCORE: int = 99
_MAX_STEPS: int = 9600  # ~2400 agent steps

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([20, 60, 20], dtype=jnp.uint8)
_COLOR_WATER = jnp.array([0, 60, 140], dtype=jnp.uint8)
_COLOR_DOCK = jnp.array([120, 80, 40], dtype=jnp.uint8)
_COLOR_LINE = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_FISH = jnp.array([255, 140, 0], dtype=jnp.uint8)
_COLOR_SHARK = jnp.array([80, 80, 80], dtype=jnp.uint8)
_COLOR_ANGLER_P = jnp.array([255, 255, 100], dtype=jnp.uint8)
_COLOR_ANGLER_CPU = jnp.array([255, 80, 80], dtype=jnp.uint8)


@chex.dataclass
class FishingDerbyState(AtariState):
    """
    Complete Fishing Derby game state — a JAX pytree.

    Parameters
    ----------
    line_y : jax.Array
        float32 — Player line hook y (increases as line drops).
    cpu_line_y : jax.Array
        float32 — CPU line hook y.
    fish_x : jax.Array
        float32[8] — Fish x positions.
    fish_dir : jax.Array
        float32[8] — Fish x velocities.
    fish_alive : jax.Array
        bool[8] — Fish still in pond.
    shark_x : jax.Array
        float32 — Shark x position.
    shark_dx : jax.Array
        float32 — Shark direction.
    cpu_score : jax.Array
        int32 — CPU score.
    """

    line_y: jax.Array
    cpu_line_y: jax.Array
    fish_x: jax.Array
    fish_dir: jax.Array
    fish_alive: jax.Array
    shark_x: jax.Array
    shark_dx: jax.Array
    cpu_score: jax.Array


class FishingDerby(AtariEnv):
    """
    Fishing Derby implemented as a pure JAX function suite.

    First to score 99 wins.  The episode also ends after the time limit
    (`max_episode_steps = 9600` emulated frames ≈ 2400 agent steps).
    No lives system.
    """

    num_actions: int = 6

    def __init__(self, params: EnvParams | None = None) -> None:
        super().__init__(params or EnvParams(noop_max=0, max_episode_steps=9600))

    def _reset(self, key: jax.Array) -> FishingDerbyState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : FishingDerbyState
            Lines above water, fish spread across pond, shark at centre.
        """
        fish_x_init = jnp.array(
            [30.0, 60.0, 90.0, 120.0, 40.0, 70.0, 100.0, 80.0], dtype=jnp.float32
        )
        fish_dir_init = jnp.array(
            [0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8], dtype=jnp.float32
        )
        return FishingDerbyState(
            line_y=jnp.float32(float(_DOCK_Y)),
            cpu_line_y=jnp.float32(float(_DOCK_Y)),
            fish_x=fish_x_init,
            fish_dir=fish_dir_init,
            fish_alive=jnp.ones(_N_FISH, dtype=jnp.bool_),
            shark_x=jnp.float32(70.0),
            shark_dx=jnp.float32(_SHARK_SPEED),
            cpu_score=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
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
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : FishingDerbyState
            Current game state.
        action : jax.Array
            int32 — 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=DOWN, 5=LEFT.

        Returns
        -------
        new_state : FishingDerbyState
            State after one emulated frame.
        """
        step_reward = jnp.float32(0.0)

        # Player line movement
        lower = action == jnp.int32(4)
        raise_ = (action == jnp.int32(2)) | (action == jnp.int32(1))
        new_line_y = jnp.where(
            lower,
            jnp.clip(
                state.line_y + _LINE_SPEED, float(_DOCK_Y), float(_WATER_BOT - _FISH_H)
            ),
            jnp.where(
                raise_,
                jnp.clip(
                    state.line_y - _LINE_SPEED * 2, float(_DOCK_Y), float(_WATER_BOT)
                ),
                state.line_y,
            ),
        )

        # CPU AI: lower line slowly
        new_cpu_line_y = jnp.clip(
            state.cpu_line_y + _LINE_SPEED * 0.5,
            float(_DOCK_Y),
            float(_WATER_BOT - _FISH_H),
        )

        # Fish movement
        new_fish_x = state.fish_x + state.fish_dir
        hit_wall_r = new_fish_x + _FISH_W >= jnp.float32(152.0)
        hit_wall_l = new_fish_x <= jnp.float32(8.0)
        new_fish_dir = jnp.where(
            hit_wall_r | hit_wall_l, -state.fish_dir, state.fish_dir
        )
        new_fish_x = jnp.clip(
            new_fish_x, jnp.float32(8.0), jnp.float32(152.0 - _FISH_W)
        )

        # Player catches fish: hook at (_LINE_X_P, line_y) hits a fish
        fish_cx = new_fish_x + _FISH_W / 2
        fish_cy = _DEPTH_YS
        hook_dist_x = jnp.abs(fish_cx - jnp.float32(_LINE_X_P))
        hook_dist_y = jnp.abs(fish_cy - new_line_y)
        caught = (
            (hook_dist_x < jnp.float32(8.0))
            & (hook_dist_y < jnp.float32(6.0))
            & state.fish_alive
        )
        n_caught = jnp.sum(caught).astype(jnp.int32)
        pts = jnp.sum(
            jnp.where(caught, _FISH_POINTS, jnp.zeros(_N_FISH, dtype=jnp.int32))
        )
        step_reward = step_reward + jnp.float32(pts)

        # Reset line on catch
        new_line_y = jnp.where(
            n_caught > jnp.int32(0), jnp.float32(float(_DOCK_Y)), new_line_y
        )

        # CPU catches fish
        cpu_hook_dist_x = jnp.abs(fish_cx - jnp.float32(_LINE_X_CPU))
        cpu_hook_dist_y = jnp.abs(fish_cy - new_cpu_line_y)
        cpu_caught = (
            (cpu_hook_dist_x < jnp.float32(8.0))
            & (cpu_hook_dist_y < jnp.float32(6.0))
            & state.fish_alive
            & ~caught
        )
        n_cpu_caught = jnp.sum(cpu_caught).astype(jnp.int32)
        cpu_pts = jnp.sum(
            jnp.where(cpu_caught, _FISH_POINTS, jnp.zeros(_N_FISH, dtype=jnp.int32))
        )
        new_cpu_line_y = jnp.where(
            n_cpu_caught > jnp.int32(0), jnp.float32(float(_DOCK_Y)), new_cpu_line_y
        )

        # Remove caught fish
        new_fish_alive = state.fish_alive & ~caught & ~cpu_caught

        # Shark movement
        new_shark_x = state.shark_x + state.shark_dx
        hit_r = new_shark_x + _SHARK_W >= jnp.float32(152.0)
        hit_l = new_shark_x <= jnp.float32(8.0)
        new_shark_dx = jnp.where(hit_r | hit_l, -state.shark_dx, state.shark_dx)
        new_shark_x = jnp.clip(
            new_shark_x, jnp.float32(8.0), jnp.float32(152.0 - _SHARK_W)
        )

        # Shark cuts player line if nearby
        shark_cx = new_shark_x + _SHARK_W / 2
        shark_cy = jnp.float32(_SHARK_Y)
        shark_cuts = (jnp.abs(jnp.float32(_LINE_X_P) - shark_cx) < jnp.float32(10)) & (
            jnp.abs(new_line_y - shark_cy) < jnp.float32(10)
        )
        new_line_y = jnp.where(shark_cuts, jnp.float32(float(_DOCK_Y)), new_line_y)

        # Respawn caught fish at random x (use simple deterministic respawn)
        respawn_x = jnp.where(
            ~new_fish_alive,
            jnp.mod(new_fish_x + jnp.float32(37.0), jnp.float32(140.0))
            + jnp.float32(10.0),
            new_fish_x,
        )
        new_fish_alive = jnp.ones(_N_FISH, dtype=jnp.bool_)  # always respawn
        new_fish_x = respawn_x

        new_score = state.score + jnp.int32(pts)
        new_cpu_score = state.cpu_score + jnp.int32(cpu_pts)
        step_reward = step_reward - jnp.float32(cpu_pts)

        done = (new_score >= jnp.int32(_WIN_SCORE)) | (
            new_cpu_score >= jnp.int32(_WIN_SCORE)
        )

        return FishingDerbyState(
            line_y=new_line_y,
            cpu_line_y=new_cpu_line_y,
            fish_x=new_fish_x,
            fish_dir=new_fish_dir,
            fish_alive=new_fish_alive,
            shark_x=new_shark_x,
            shark_dx=new_shark_dx,
            cpu_score=new_cpu_score,
            lives=jnp.int32(0),
            score=new_score,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=state.key,
        )

    def _step(self, state: FishingDerbyState, action: jax.Array) -> FishingDerbyState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : FishingDerbyState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : FishingDerbyState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: FishingDerbyState) -> FishingDerbyState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: FishingDerbyState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : FishingDerbyState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), 0, dtype=jnp.uint8)
        frame = jnp.where(jnp.ones((210, 160, 1), dtype=jnp.bool_), _COLOR_BG, frame)

        # Water
        in_water = (_ROW_IDX >= _WATER_TOP) & (_ROW_IDX < _WATER_BOT)
        frame = jnp.where(in_water[:, :, None], _COLOR_WATER, frame)

        # Dock
        dock = (_ROW_IDX >= _DOCK_Y) & (_ROW_IDX < _DOCK_Y + 8)
        frame = jnp.where(dock[:, :, None], _COLOR_DOCK, frame)

        # Player angler
        p_mask = (
            (_ROW_IDX >= _DOCK_Y - 12)
            & (_ROW_IDX < _DOCK_Y)
            & (_COL_IDX >= _LINE_X_P - 4)
            & (_COL_IDX < _LINE_X_P + 4)
        )
        frame = jnp.where(p_mask[:, :, None], _COLOR_ANGLER_P, frame)

        # CPU angler
        cpu_mask = (
            (_ROW_IDX >= _DOCK_Y - 12)
            & (_ROW_IDX < _DOCK_Y)
            & (_COL_IDX >= _LINE_X_CPU - 4)
            & (_COL_IDX < _LINE_X_CPU + 4)
        )
        frame = jnp.where(cpu_mask[:, :, None], _COLOR_ANGLER_CPU, frame)

        # Player line
        line_mask = (
            (_ROW_IDX >= _DOCK_Y)
            & (_ROW_IDX <= jnp.int32(state.line_y))
            & (_COL_IDX == _LINE_X_P)
        )
        frame = jnp.where(line_mask[:, :, None], _COLOR_LINE, frame)

        # CPU line
        cpu_line_mask = (
            (_ROW_IDX >= _DOCK_Y)
            & (_ROW_IDX <= jnp.int32(state.cpu_line_y))
            & (_COL_IDX == _LINE_X_CPU)
        )
        frame = jnp.where(cpu_line_mask[:, :, None], _COLOR_LINE, frame)

        # Fish
        def draw_fish(frm, i):
            fx = state.fish_x[i]
            fy = _DEPTH_YS[i]
            alive = state.fish_alive[i]
            mask = (
                alive
                & (_ROW_IDX >= jnp.int32(fy))
                & (_ROW_IDX < jnp.int32(fy) + _FISH_H)
                & (_COL_IDX >= jnp.int32(fx))
                & (_COL_IDX < jnp.int32(fx) + _FISH_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_FISH, frm), None

        frame, _ = jax.lax.scan(draw_fish, frame, jnp.arange(_N_FISH))

        # Shark
        shark_mask = (
            (_ROW_IDX >= jnp.int32(_SHARK_Y))
            & (_ROW_IDX < jnp.int32(_SHARK_Y) + _SHARK_H)
            & (_COL_IDX >= jnp.int32(state.shark_x))
            & (_COL_IDX < jnp.int32(state.shark_x) + _SHARK_W)
        )
        frame = jnp.where(shark_mask[:, :, None], _COLOR_SHARK, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Fishing Derby action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
        }
