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

"""Crazy Climber — JAX-native game implementation.

Climb a skyscraper by grabbing window ledges while avoiding obstacles
thrown from windows and a giant condor that drops droppings.

Action space (8 actions):
    0 — NOOP
    1 — UP
    2 — UP + RIGHT
    3 — RIGHT
    4 — DOWN + RIGHT
    5 — DOWN
    6 — DOWN + LEFT
    7 — LEFT

Scoring:
    Reach a bonus flag — +1000
    Reach the top      — +3000
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
_SCREEN_W: int = 160
_SCREEN_H: int = 210

_CLIMBER_X_MIN: int = 10
_CLIMBER_X_MAX: int = 150
_CLIMBER_Y_MIN: int = 20  # top of building (goal)
_CLIMBER_Y_MAX: int = 185  # ground level

_CLIMBER_SPEED_X: float = 3.0
_CLIMBER_SPEED_Y: float = 2.0

_N_WINDOWS: int = 10  # window obstacles (opening/closing)
_N_FLOWERPOTS: int = 4  # thrown obstacles
_CONDOR_Y: int = 60
_DROPPINGS_SPEED: float = 3.0
_FLOWERPOT_SPEED: float = 2.5
_WINDOW_OPEN_FRAMES: int = 30
_MILESTONE_Y = jnp.array([150, 110, 70, 30], dtype=jnp.int32)  # bonus flag positions

_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([60, 60, 60], dtype=jnp.uint8)
_COLOR_BUILDING = jnp.array([140, 140, 160], dtype=jnp.uint8)
_COLOR_WINDOW = jnp.array([200, 220, 255], dtype=jnp.uint8)
_COLOR_WINDOW_OPEN = jnp.array([30, 30, 50], dtype=jnp.uint8)
_COLOR_CLIMBER = jnp.array([255, 220, 80], dtype=jnp.uint8)
_COLOR_CONDOR = jnp.array([80, 60, 40], dtype=jnp.uint8)
_COLOR_HAZARD = jnp.array([220, 120, 40], dtype=jnp.uint8)


@chex.dataclass
class CrazyClimberState(AtariState):
    """
    Complete Crazy Climber game state — a JAX pytree.

    Parameters
    ----------
    climber_x : jax.Array
        float32 — Climber x position.
    climber_y : jax.Array
        float32 — Climber y position (small = high on building).
    window_open : jax.Array
        bool[10] — Window currently open (hazard to grab).
    window_timer : jax.Array
        int32[10] — Frames until window state toggles.
    pot_x : jax.Array
        float32[4] — Flowerpot x positions.
    pot_y : jax.Array
        float32[4] — Flowerpot y positions.
    pot_active : jax.Array
        bool[4] — Flowerpot falling.
    condor_x : jax.Array
        float32 — Condor x position.
    condor_dir : jax.Array
        int32 — Condor direction.
    dropping_y : jax.Array
        float32 — Bird dropping y (−1 = inactive).
    dropping_active : jax.Array
        bool — Dropping falling.
    spawn_timer : jax.Array
        int32 — Frames until next pot drop.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    climber_x: jax.Array
    climber_y: jax.Array
    window_open: jax.Array
    window_timer: jax.Array
    pot_x: jax.Array
    pot_y: jax.Array
    pot_active: jax.Array
    condor_x: jax.Array
    condor_dir: jax.Array
    dropping_y: jax.Array
    dropping_active: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


# Precomputed window positions (x centre, y centre)
_WINDOW_X = jnp.array([40, 80, 120, 40, 80, 120, 40, 80, 120, 40], dtype=jnp.int32)
_WINDOW_Y = jnp.array([150, 150, 150, 110, 110, 110, 70, 70, 70, 30], dtype=jnp.int32)


class CrazyClimber(AtariEnv):
    """
    Crazy Climber implemented as a pure JAX function suite.

    Climb to the top avoiding hazards.  Lives: 3.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> CrazyClimberState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : CrazyClimberState
            Climber at bottom, all windows closed, 3 lives.
        """
        return CrazyClimberState(
            climber_x=jnp.float32(80.0),
            climber_y=jnp.float32(float(_CLIMBER_Y_MAX)),
            window_open=jnp.zeros(_N_WINDOWS, dtype=jnp.bool_),
            window_timer=jnp.full(_N_WINDOWS, _WINDOW_OPEN_FRAMES, dtype=jnp.int32),
            pot_x=jnp.full(_N_FLOWERPOTS, -10.0, dtype=jnp.float32),
            pot_y=jnp.full(_N_FLOWERPOTS, -10.0, dtype=jnp.float32),
            pot_active=jnp.zeros(_N_FLOWERPOTS, dtype=jnp.bool_),
            condor_x=jnp.float32(10.0),
            condor_dir=jnp.int32(1),
            dropping_y=jnp.float32(-10.0),
            dropping_active=jnp.bool_(False),
            spawn_timer=jnp.int32(60),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(
        self, state: CrazyClimberState, action: jax.Array
    ) -> CrazyClimberState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : CrazyClimberState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : CrazyClimberState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Climber movement
        # Actions: 0=NOOP, 1=U, 2=UR, 3=R, 4=DR, 5=D, 6=DL, 7=L
        _dy = (
            jnp.array([0, -1, -1, 0, 1, 1, 1, 0], dtype=jnp.float32) * _CLIMBER_SPEED_Y
        )
        _dx = (
            jnp.array([0, 0, 1, 1, 1, 0, -1, -1], dtype=jnp.float32) * _CLIMBER_SPEED_X
        )
        new_cx = jnp.clip(state.climber_x + _dx[action], _CLIMBER_X_MIN, _CLIMBER_X_MAX)
        new_cy = jnp.clip(state.climber_y + _dy[action], _CLIMBER_Y_MIN, _CLIMBER_Y_MAX)

        # Milestone bonuses
        for ms_y in [150, 110, 70, 30]:
            reached = (new_cy <= jnp.float32(ms_y)) & (
                state.climber_y > jnp.float32(ms_y)
            )
            step_reward = step_reward + jnp.where(
                reached, jnp.float32(1000.0), jnp.float32(0.0)
            )

        # Reached the top
        at_top = new_cy <= jnp.float32(_CLIMBER_Y_MIN + 2)
        step_reward = step_reward + jnp.where(
            at_top, jnp.float32(3000.0), jnp.float32(0.0)
        )
        # Reset to bottom on reaching top
        new_cy = jnp.where(at_top, jnp.float32(_CLIMBER_Y_MAX), new_cy)

        # Window hazards (toggle)
        new_window_timer = state.window_timer - jnp.int32(1)
        toggle = new_window_timer <= jnp.int32(0)
        new_window_open = jnp.where(toggle, ~state.window_open, state.window_open)
        new_window_timer = jnp.where(
            toggle, jnp.int32(_WINDOW_OPEN_FRAMES), new_window_timer
        )

        # Climber grabbed open window → fall
        window_hit = jnp.any(
            new_window_open
            & (jnp.abs(jnp.float32(_WINDOW_X) - new_cx) < 8.0)
            & (jnp.abs(jnp.float32(_WINDOW_Y) - new_cy) < 6.0)
        )

        # Flowerpot physics
        new_pot_y = state.pot_y + jnp.where(state.pot_active, _FLOWERPOT_SPEED, 0.0)
        pot_off = state.pot_active & (new_pot_y > _CLIMBER_Y_MAX)
        new_pot_active = state.pot_active & ~pot_off

        # Pot hits climber
        pot_hits = (
            new_pot_active
            & (jnp.abs(state.pot_x - new_cx) < 8.0)
            & (jnp.abs(new_pot_y - new_cy) < 8.0)
        )
        hit_by_pot = jnp.any(pot_hits)
        new_pot_active = new_pot_active & ~pot_hits

        # Spawn flowerpot
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(60), new_spawn_timer)
        spawn_x = jax.random.uniform(sk1) * 120.0 + 20.0
        free_pot = jnp.argmin(new_pot_active.astype(jnp.int32))
        new_pot_x = jnp.where(
            do_spawn, state.pot_x.at[free_pot].set(spawn_x), state.pot_x
        )
        new_pot_y = jnp.where(
            do_spawn, new_pot_y.at[free_pot].set(jnp.float32(20.0)), new_pot_y
        )
        new_pot_active = jnp.where(
            do_spawn, new_pot_active.at[free_pot].set(True), new_pot_active
        )

        # Condor movement
        new_condor_x = state.condor_x + state.condor_dir.astype(jnp.float32) * 1.5
        at_edge = (new_condor_x < 10.0) | (new_condor_x > 150.0)
        new_condor_dir = jnp.where(at_edge, -state.condor_dir, state.condor_dir)
        new_condor_x = jnp.clip(new_condor_x, 10.0, 150.0)

        # Dropping
        new_dropping_y = state.dropping_y + jnp.where(
            state.dropping_active, _DROPPINGS_SPEED, 0.0
        )
        dropping_off = state.dropping_active & (new_dropping_y > _CLIMBER_Y_MAX)
        new_dropping_active = state.dropping_active & ~dropping_off
        # Spawn dropping when condor above climber
        near_condor = jnp.abs(new_condor_x - new_cx) < 20.0
        do_drop = near_condor & ~new_dropping_active
        new_dropping_y = jnp.where(do_drop, jnp.float32(_CONDOR_Y), new_dropping_y)
        new_dropping_active = jnp.where(do_drop, jnp.bool_(True), new_dropping_active)

        # Dropping hits climber
        dropping_hit = (
            new_dropping_active
            & (jnp.abs(new_condor_x - new_cx) < 6.0)
            & (jnp.abs(new_dropping_y - new_cy) < 6.0)
        )
        new_dropping_active = new_dropping_active & ~dropping_hit

        life_lost = window_hit | hit_by_pot | dropping_hit
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        # Respawn climber at bottom on death
        new_cy = jnp.where(life_lost, jnp.float32(_CLIMBER_Y_MAX), new_cy)
        new_cx = jnp.where(life_lost, jnp.float32(80.0), new_cx)

        done = new_lives <= jnp.int32(0)

        return CrazyClimberState(
            climber_x=new_cx,
            climber_y=new_cy,
            window_open=new_window_open,
            window_timer=new_window_timer,
            pot_x=new_pot_x,
            pot_y=new_pot_y,
            pot_active=new_pot_active,
            condor_x=new_condor_x,
            condor_dir=new_condor_dir,
            dropping_y=new_dropping_y,
            dropping_active=new_dropping_active,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: CrazyClimberState, action: jax.Array) -> CrazyClimberState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : CrazyClimberState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : CrazyClimberState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: CrazyClimberState) -> CrazyClimberState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: CrazyClimberState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : CrazyClimberState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Building
        building_mask = (_COL_IDX >= 10) & (_COL_IDX <= 150)
        frame = jnp.where(building_mask[:, :, None], _COLOR_BUILDING, frame)

        # Windows
        def draw_window(frm, i):
            wx = _WINDOW_X[i]
            wy = _WINDOW_Y[i]
            color = jnp.where(state.window_open[i], _COLOR_WINDOW_OPEN, _COLOR_WINDOW)
            mask = (
                (_ROW_IDX >= wy - 6)
                & (_ROW_IDX <= wy + 6)
                & (_COL_IDX >= wx - 8)
                & (_COL_IDX <= wx + 8)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_window, frame, jnp.arange(_N_WINDOWS))

        # Flowerpots
        def draw_pot(frm, i):
            px = state.pot_x[i].astype(jnp.int32)
            py = state.pot_y[i].astype(jnp.int32)
            mask = (
                state.pot_active[i]
                & (_ROW_IDX >= py - 4)
                & (_ROW_IDX <= py + 4)
                & (_COL_IDX >= px - 4)
                & (_COL_IDX <= px + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_HAZARD, frm), None

        frame, _ = jax.lax.scan(draw_pot, frame, jnp.arange(_N_FLOWERPOTS))

        # Condor
        cx = state.condor_x.astype(jnp.int32)
        condor_mask = (
            (_ROW_IDX >= _CONDOR_Y - 6)
            & (_ROW_IDX <= _CONDOR_Y + 6)
            & (_COL_IDX >= cx - 10)
            & (_COL_IDX <= cx + 10)
        )
        frame = jnp.where(condor_mask[:, :, None], _COLOR_CONDOR, frame)

        # Climber
        climx = state.climber_x.astype(jnp.int32)
        cliiy = state.climber_y.astype(jnp.int32)
        climber_mask = (
            (_ROW_IDX >= cliiy - 6)
            & (_ROW_IDX <= cliiy + 6)
            & (_COL_IDX >= climx - 4)
            & (_COL_IDX <= climx + 4)
        )
        frame = jnp.where(climber_mask[:, :, None], _COLOR_CLIMBER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Crazy Climber action indices.
        """
        import pygame

        return {
            pygame.K_UP: 1,
            pygame.K_w: 1,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_DOWN: 5,
            pygame.K_s: 5,
            pygame.K_LEFT: 7,
            pygame.K_a: 7,
        }
