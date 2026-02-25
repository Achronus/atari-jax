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

"""Pitfall! — JAX-native game implementation.

Harry must collect 32 treasures scattered across 255 jungle screens
within 20 minutes while avoiding hazards: rolling logs, scorpions,
quicksand, crocodiles, and underground tar pits.

Action space (6 actions):
    0 — NOOP
    1 — FIRE  (unused; kept for ALE compatibility)
    2 — UP    (climb/jump)
    3 — RIGHT (run right)
    4 — DOWN  (descend underground)
    5 — LEFT  (run left)

Scoring:
    Treasure collected — +2000 to +5000
    Falling in hazard  — −100 (score penalty; no life cost for most)
    Episode ends when all lives are lost or time runs out; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_TREASURES: int = 32
_N_HAZARDS: int = 6  # per-screen hazard objects

_SCREEN_W: int = 160
_GROUND_Y: int = 160
_UNDERGROUND_Y: int = 195
_PLAYER_SPEED: float = 2.5
_JUMP_VELOCITY: float = -5.0
_GRAVITY: float = 0.4
_LOG_SPEED: float = 1.5
_CROC_SPEED: float = 1.0

_INIT_LIVES: int = 3
_TOTAL_SCREENS: int = 32  # simplified to 32 screens with treasures

# Treasure values (cycling pattern)
_TREASURE_VALUES = jnp.array(
    [2000, 3000, 4000, 5000, 2000, 3000, 4000, 5000] * 4, dtype=jnp.int32
)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_SKY = jnp.array([100, 160, 60], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([60, 100, 20], dtype=jnp.uint8)
_COLOR_UNDERGROUND = jnp.array([80, 50, 20], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([220, 100, 40], dtype=jnp.uint8)
_COLOR_LOG = jnp.array([120, 80, 30], dtype=jnp.uint8)
_COLOR_CROC = jnp.array([40, 140, 40], dtype=jnp.uint8)
_COLOR_SCORPION = jnp.array([200, 200, 40], dtype=jnp.uint8)
_COLOR_TREASURE = jnp.array([255, 215, 0], dtype=jnp.uint8)


@chex.dataclass
class PitfallState(AtariState):
    """
    Complete Pitfall! game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    player_vy : jax.Array
        float32 — Vertical velocity.
    underground : jax.Array
        bool — Player is in underground tunnel.
    screen : jax.Array
        int32 — Current screen index (0–31).
    treasures : jax.Array
        bool[32] — Whether each screen's treasure has been collected.
    log_x : jax.Array
        float32[3] — Rolling log x positions.
    croc_x : jax.Array
        float32[2] — Crocodile x positions.
    croc_mouth : jax.Array
        bool[2] — Crocodile mouth open (dangerous if open).
    croc_timer : jax.Array
        int32[2] — Frames until croc mouth toggles.
    time_left : jax.Array
        int32 — Frames remaining (20 min = 72000 frames).
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    underground: jax.Array
    screen: jax.Array
    treasures: jax.Array
    log_x: jax.Array
    croc_x: jax.Array
    croc_mouth: jax.Array
    croc_timer: jax.Array
    time_left: jax.Array
    key: jax.Array


class Pitfall(AtariEnv):
    """
    Pitfall! implemented as a pure JAX function suite.

    Collect all 32 treasures within 20 minutes.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=72_000)

    def _reset(self, key: jax.Array) -> PitfallState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : PitfallState
            Harry at start, all treasures uncollected, 3 lives.
        """
        return PitfallState(
            player_x=jnp.float32(20.0),
            player_y=jnp.float32(float(_GROUND_Y) - 12.0),
            player_vy=jnp.float32(0.0),
            underground=jnp.bool_(False),
            screen=jnp.int32(0),
            treasures=jnp.zeros(_N_TREASURES, dtype=jnp.bool_),
            log_x=jnp.array([30.0, 80.0, 130.0], dtype=jnp.float32),
            croc_x=jnp.array([60.0, 110.0], dtype=jnp.float32),
            croc_mouth=jnp.array([False, True], dtype=jnp.bool_),
            croc_timer=jnp.array([20, 20], dtype=jnp.int32),
            time_left=jnp.int32(72_000),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: PitfallState, action: jax.Array) -> PitfallState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : PitfallState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : PitfallState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Time
        new_time = state.time_left - jnp.int32(1)

        # Horizontal movement / screen scroll
        move_r = action == jnp.int32(3)
        move_l = action == jnp.int32(5)
        new_px = (
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l, -_PLAYER_SPEED, 0.0)
        )

        # Screen transition
        went_right = new_px >= jnp.float32(_SCREEN_W - 10)
        went_left = new_px < jnp.float32(10)
        new_screen = jnp.clip(
            state.screen
            + jnp.where(went_right, jnp.int32(1), jnp.int32(0))
            + jnp.where(went_left, jnp.int32(-1), jnp.int32(0)),
            0,
            _TOTAL_SCREENS - 1,
        )
        new_px = jnp.where(went_right, jnp.float32(10.0), new_px)
        new_px = jnp.where(went_left, jnp.float32(float(_SCREEN_W) - 10.0), new_px)

        # Underground entry/exit
        go_underground = (action == jnp.int32(4)) & ~state.underground
        go_surface = (action == jnp.int32(2)) & state.underground
        new_underground = jnp.where(go_underground, jnp.bool_(True), state.underground)
        new_underground = jnp.where(go_surface, jnp.bool_(False), new_underground)
        ground_y = jnp.where(
            new_underground,
            jnp.float32(_UNDERGROUND_Y) - 12.0,
            jnp.float32(_GROUND_Y) - 12.0,
        )

        # Jump / gravity
        on_ground = state.player_y >= ground_y - 2.0
        do_jump = (action == jnp.int32(2)) & on_ground & ~state.underground
        new_vy = jnp.where(
            do_jump, jnp.float32(_JUMP_VELOCITY), state.player_vy + _GRAVITY
        )
        new_py = state.player_y + new_vy
        landed = new_py >= ground_y
        new_py = jnp.where(landed, ground_y, new_py)
        new_vy = jnp.where(landed, jnp.float32(0.0), new_vy)

        # Reset position when changing underground state
        new_py = jnp.where(go_underground | go_surface, ground_y, new_py)
        new_vy = jnp.where(go_underground | go_surface, jnp.float32(0.0), new_vy)

        # Logs roll
        new_log_x = state.log_x + _LOG_SPEED
        new_log_x = jnp.where(new_log_x > 160.0, jnp.float32(0.0), new_log_x)

        # Croc mouth toggle
        new_croc_timer = state.croc_timer - jnp.int32(1)
        toggle_croc = new_croc_timer <= jnp.int32(0)
        new_croc_mouth = jnp.where(toggle_croc, ~state.croc_mouth, state.croc_mouth)
        new_croc_timer = jnp.where(toggle_croc, jnp.int32(20), new_croc_timer)

        # Treasure collection
        treasure_y = jnp.float32(_GROUND_Y - 12.0)
        treasure_x = jnp.float32(80.0)
        on_treasure_screen = new_screen == jnp.arange(_N_TREASURES, dtype=jnp.int32)
        near_treasure = jnp.abs(new_px - treasure_x) < 10.0
        at_treasure_y = ~new_underground
        can_collect = (
            on_treasure_screen & near_treasure & at_treasure_y & ~state.treasures
        )
        n_collected = jnp.sum(can_collect).astype(jnp.int32)
        collect_value = jnp.sum(
            jnp.where(can_collect, _TREASURE_VALUES, jnp.int32(0))
        ).astype(jnp.float32)
        step_reward = step_reward + collect_value
        new_treasures = state.treasures | can_collect

        # Log collision
        log_hit = jnp.any(
            (jnp.abs(new_log_x - new_px) < 10.0)
            & (jnp.abs(jnp.float32(_GROUND_Y - 8) - new_py) < 12.0)
            & ~new_underground
        )
        step_reward = step_reward + jnp.where(
            log_hit, jnp.float32(-100.0), jnp.float32(0.0)
        )
        new_px = jnp.where(log_hit, jnp.float32(20.0), new_px)

        # Croc collision (only dangerous when mouth open)
        croc_hit = jnp.any(
            (jnp.abs(state.croc_x - new_px) < 10.0) & new_croc_mouth & ~new_underground
        )
        new_lives = state.lives - jnp.where(croc_hit, jnp.int32(1), jnp.int32(0))
        new_px = jnp.where(croc_hit, jnp.float32(20.0), new_px)

        time_out = new_time <= jnp.int32(0)
        done = (new_lives <= jnp.int32(0)) | time_out | jnp.all(new_treasures)

        return PitfallState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            underground=new_underground,
            screen=new_screen,
            treasures=new_treasures,
            log_x=new_log_x,
            croc_x=state.croc_x,
            croc_mouth=new_croc_mouth,
            croc_timer=new_croc_timer,
            time_left=new_time,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: PitfallState, action: jax.Array) -> PitfallState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : PitfallState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : PitfallState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: PitfallState) -> PitfallState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: PitfallState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : PitfallState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        # Sky / jungle background
        frame = jnp.where(
            (_ROW_IDX < _GROUND_Y)[:, :, None],
            jnp.full((210, 160, 3), _COLOR_SKY, dtype=jnp.uint8),
            jnp.full((210, 160, 3), _COLOR_GROUND, dtype=jnp.uint8),
        )

        # Underground tunnel
        frame = jnp.where(
            (state.underground & (_ROW_IDX >= _UNDERGROUND_Y - 20))[:, :, None],
            _COLOR_UNDERGROUND,
            frame,
        )

        # Logs
        def draw_log(frm, i):
            lx = state.log_x[i].astype(jnp.int32)
            mask = (
                ~state.underground
                & (_ROW_IDX >= _GROUND_Y - 16)
                & (_ROW_IDX <= _GROUND_Y - 4)
                & (_COL_IDX >= lx - 8)
                & (_COL_IDX <= lx + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_LOG, frm), None

        frame, _ = jax.lax.scan(draw_log, frame, jnp.arange(3))

        # Crocs
        def draw_croc(frm, i):
            cx = state.croc_x[i].astype(jnp.int32)
            mask = (
                ~state.underground
                & (_ROW_IDX >= _GROUND_Y - 14)
                & (_ROW_IDX <= _GROUND_Y - 2)
                & (_COL_IDX >= cx - 10)
                & (_COL_IDX <= cx + 10)
            )
            return jnp.where(mask[:, :, None], _COLOR_CROC, frm), None

        frame, _ = jax.lax.scan(draw_croc, frame, jnp.arange(2))

        # Treasure (on current screen)
        treasure_present = ~state.treasures[state.screen]
        t_mask = (
            treasure_present
            & ~state.underground
            & (_ROW_IDX >= _GROUND_Y - 20)
            & (_ROW_IDX <= _GROUND_Y - 8)
            & (_COL_IDX >= 74)
            & (_COL_IDX <= 86)
        )
        frame = jnp.where(t_mask[:, :, None], _COLOR_TREASURE, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 8)
            & (_ROW_IDX <= py + 8)
            & (_COL_IDX >= px - 4)
            & (_COL_IDX <= px + 4)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Pitfall action indices.
        """
        import pygame

        return {
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
