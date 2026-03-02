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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Harry   : feet at y=160, width=8, height=16; head at y=144
    Logs    : 3 logs, top at y=152, bottom at y=160; patrol full screen width
    Crocs   : 2 crocs at x=60, 110; top at y=150, bottom at y=160; period=120 frames
    Treasure: left edge at x=75, top at y=150, bottom at y=160; every 8th screen
    Underground: feet at y=185, passage rows 168–190

Action space (18 actions — ALE minimal set):
    0   NOOP
    1   FIRE (unused)
    2   UP        — jump / exit underground
    3   RIGHT     — move right
    4   LEFT      — move left
    5   DOWN      — enter underground
    6   UPRIGHT   — jump right
    7   UPLEFT    — jump left
    8   DOWNRIGHT — enter underground + right
    9   DOWNLEFT  — enter underground + left
    10  UPFIRE    — jump
    11  RIGHTFIRE — move right
    12  LEFTFIRE  — move left
    13  DOWNFIRE  — enter underground
    14  UPRIGHTFIRE  — jump right
    15  UPLEFTFIRE   — jump left
    16  DOWNRIGHTFIRE — enter underground + right
    17  DOWNLEFTFIRE  — enter underground + left
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# --- Harry geometry ---
_HARRY_W: int = 8
_HARRY_H: int = 16
_HARRY_INIT_X: float = 80.0  # starting left-edge x
_HARRY_SPEED: float = 1.5  # px/frame horizontal

# --- Jump physics ---
_JUMP_VY: float = -4.0  # initial upward velocity (negative = up)
_GRAVITY: float = 0.5  # downward acceleration px/frame²

# --- Vertical layout ---
_GROUND_Y: float = 160.0  # feet y when standing on surface
_UNDERGROUND_Y: float = 185.0  # feet y when in underground tunnel

# --- Screen / scroll ---
_SCREEN_LEFT: float = 10.0  # left transition boundary (Harry left-edge)
_SCREEN_RIGHT: float = 150.0  # right transition boundary (Harry left-edge)
_N_SCREENS: int = 255  # screens 0–254

# --- Logs ---
_LOG_W: int = 16
_LOG_H: int = 8
_LOG_TOP: float = _GROUND_Y - _LOG_H  # 152.0 — top edge
# Velocities include direction: positive = rightward, negative = leftward
_LOG_VX: tuple = (1.0, -1.5, 2.0)
_LOG_INIT_X: tuple = (20.0, 90.0, 140.0)
_LOG_COOLDOWN_FRAMES: int = 55  # frames of invincibility after log hit

# --- Crocodiles ---
_CROC_W: int = 20
_CROC_H: int = 10
_CROC_TOP: float = _GROUND_Y - _CROC_H  # 150.0
_CROC_X: tuple = (60.0, 110.0)  # left edges (fixed positions)
_CROC_PERIOD: int = 120  # frames per open/close cycle
_CROC_OPEN_AT: int = 60  # croc is dangerous when timer > this

# --- Treasure ---
_TREASURE_W: int = 10
_TREASURE_H: int = 10
_TREASURE_X: float = 75.0  # left edge (centred ~80)
_TREASURE_TOP: float = _GROUND_Y - _TREASURE_H  # 150.0
_TREASURE_REWARD: int = 2000

# --- Log penalty ---
_LOG_PENALTY: int = -100

# --- Lives ---
_INIT_LIVES: int = 3

# --- Underground entry holes ---
_HOLE_HALF: float = 8.0  # half-width of hole activation zone
_HOLE_XS: tuple = (40.0, 80.0, 120.0)  # hole centre x positions

# --- Frame skip ---
_FRAME_SKIP: int = 4

# --- Precomputed arrays (module-level for render only — NOT indexed inside JIT loops) ---
_LOG_VX_ARR = jnp.array(_LOG_VX, dtype=jnp.float32)

_ROW_IDX_R = jnp.arange(210)[:, None]
_COL_IDX_R = jnp.arange(160)[None, :]

# --- Colours ---
_BG_TOP_COLOR = jnp.array([20, 90, 20], dtype=jnp.uint8)  # jungle canopy
_BG_MID_COLOR = jnp.array([30, 130, 30], dtype=jnp.uint8)  # jungle floor
_GROUND_COLOR = jnp.array([139, 90, 43], dtype=jnp.uint8)  # brown earth
_UG_COLOR = jnp.array([15, 15, 15], dtype=jnp.uint8)  # underground dark
_HARRY_COLOR = jnp.array([220, 160, 60], dtype=jnp.uint8)  # Harry
_LOG_COLOR = jnp.array([101, 67, 33], dtype=jnp.uint8)  # log brown
_CROC_SAFE_COLOR = jnp.array([34, 139, 34], dtype=jnp.uint8)  # croc closed
_CROC_OPEN_COLOR = jnp.array([220, 30, 30], dtype=jnp.uint8)  # croc open (danger)
_TREASURE_COLOR = jnp.array([255, 215, 0], dtype=jnp.uint8)  # gold


@chex.dataclass
class PitfallState(AtariState):
    """
    Complete Pitfall! game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `level` stores the current screen index (0–254).

    Parameters
    ----------
    harry_x : chex.Array
        float32 — Harry's left-edge x.
    harry_y : chex.Array
        float32 — Harry's feet y (bottom edge). _GROUND_Y when standing.
    harry_vy : chex.Array
        float32 — Vertical velocity; negative = moving upward.
    is_jumping : chex.Array
        bool — True while Harry is airborne.
    is_underground : chex.Array
        bool — True while Harry is in the underground tunnel.
    log_x : chex.Array
        float32[3] — Left edges of the three rolling logs.
    croc_timer : chex.Array
        int32 — Phase counter within the 120-frame croc open/close cycle.
    log_cooldown : chex.Array
        int32 — Frames remaining before the next log collision penalty applies.
    treasure_active : chex.Array
        bool — Treasure is present and uncollected on the current screen.
    """

    harry_x: chex.Array
    harry_y: chex.Array
    harry_vy: chex.Array
    is_jumping: chex.Array
    is_underground: chex.Array
    log_x: chex.Array
    croc_timer: chex.Array
    log_cooldown: chex.Array
    treasure_active: chex.Array


class Pitfall(AtaraxGame):
    """
    Pitfall! implemented as a pure-JAX function suite.

    Harry navigates 255 scrolling jungle screens collecting up to 32 treasures
    while jumping over rolling logs and timing moves around open-mouthed crocs.
    """

    num_actions: int = 18

    def _reset(self, key: chex.PRNGKey) -> PitfallState:
        """Return the canonical initial game state."""
        return PitfallState(
            harry_x=jnp.float32(_HARRY_INIT_X),
            harry_y=jnp.float32(_GROUND_Y),
            harry_vy=jnp.float32(0.0),
            is_jumping=jnp.bool_(False),
            is_underground=jnp.bool_(False),
            log_x=jnp.array(_LOG_INIT_X, dtype=jnp.float32),
            croc_timer=jnp.int32(0),
            log_cooldown=jnp.int32(0),
            treasure_active=jnp.bool_(False),  # screen 0 has no treasure
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),  # screen index
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: PitfallState, action: jax.Array) -> PitfallState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : PitfallState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).

        Returns
        -------
        new_state : PitfallState
            State after one emulated frame.
        """
        # --- Action decode ---
        has_up = (
            (action == 2)
            | (action == 6)
            | (action == 7)
            | (action == 10)
            | (action == 14)
            | (action == 15)
        )
        has_right = (
            (action == 3)
            | (action == 6)
            | (action == 8)
            | (action == 11)
            | (action == 14)
            | (action == 16)
        )
        has_left = (
            (action == 4)
            | (action == 7)
            | (action == 9)
            | (action == 12)
            | (action == 15)
            | (action == 17)
        )
        has_down = (
            (action == 5)
            | (action == 8)
            | (action == 9)
            | (action == 13)
            | (action == 16)
            | (action == 17)
        )

        # --- Update croc timer ---
        croc_timer = (state.croc_timer + jnp.int32(1)) % jnp.int32(_CROC_PERIOD)
        croc_open = croc_timer > jnp.int32(_CROC_OPEN_AT)

        # --- Move logs (vectorised) ---
        new_log_x = state.log_x + _LOG_VX_ARR
        # Wrap logs off the right edge back to the left (and vice versa)
        new_log_x = jnp.where(
            new_log_x > jnp.float32(160.0),
            new_log_x - jnp.float32(160.0 + _LOG_W),
            new_log_x,
        )
        new_log_x = jnp.where(
            new_log_x < jnp.float32(-_LOG_W),
            new_log_x + jnp.float32(160.0 + _LOG_W),
            new_log_x,
        )

        # --- Underground exit (takes priority so UP can still trigger jump next frame) ---
        should_exit_ug = has_up & state.is_underground & ~state.is_jumping

        # --- Horizontal movement ---
        dx = jnp.where(
            has_right,
            jnp.float32(_HARRY_SPEED),
            jnp.where(has_left, jnp.float32(-_HARRY_SPEED), jnp.float32(0.0)),
        )
        harry_x = state.harry_x + dx

        # --- Jump initiation (only above ground and not already airborne) ---
        at_surface = state.harry_y >= jnp.float32(_GROUND_Y - 0.5)
        can_jump = ~state.is_underground & ~state.is_jumping & at_surface
        do_jump = has_up & can_jump

        harry_vy = jnp.where(do_jump, jnp.float32(_JUMP_VY), state.harry_vy)
        is_jumping = state.is_jumping | do_jump

        # --- Apply gravity and update vertical position ---
        harry_vy = harry_vy + jnp.float32(_GRAVITY)
        harry_y = state.harry_y + harry_vy

        # --- Landing ---
        landed = is_jumping & (harry_y >= jnp.float32(_GROUND_Y))
        harry_y = jnp.where(landed, jnp.float32(_GROUND_Y), harry_y)
        harry_vy = jnp.where(landed, jnp.float32(0.0), harry_vy)
        is_jumping = is_jumping & ~landed

        # --- Apply underground exit (teleport Harry back to surface) ---
        harry_y = jnp.where(should_exit_ug, jnp.float32(_GROUND_Y), harry_y)
        harry_vy = jnp.where(should_exit_ug, jnp.float32(0.0), harry_vy)

        # --- Underground entry ---
        on_surface_now = ~state.is_underground & at_surface
        close_to_hole = (
            (
                (harry_x > jnp.float32(_HOLE_XS[0] - _HOLE_HALF))
                & (harry_x < jnp.float32(_HOLE_XS[0] + _HOLE_HALF))
            )
            | (
                (harry_x > jnp.float32(_HOLE_XS[1] - _HOLE_HALF))
                & (harry_x < jnp.float32(_HOLE_XS[1] + _HOLE_HALF))
            )
            | (
                (harry_x > jnp.float32(_HOLE_XS[2] - _HOLE_HALF))
                & (harry_x < jnp.float32(_HOLE_XS[2] + _HOLE_HALF))
            )
        )
        should_enter_ug = has_down & on_surface_now & close_to_hole & ~state.is_jumping

        is_underground = (state.is_underground | should_enter_ug) & ~should_exit_ug
        harry_y = jnp.where(should_enter_ug, jnp.float32(_UNDERGROUND_Y), harry_y)

        # --- Screen transitions (above ground only) ---
        at_right = ~is_underground & (harry_x > jnp.float32(_SCREEN_RIGHT))
        at_left = ~is_underground & (harry_x < jnp.float32(_SCREEN_LEFT))

        screen_num = state.level + jnp.where(
            at_right, jnp.int32(1), jnp.where(at_left, jnp.int32(-1), jnp.int32(0))
        )
        screen_num = screen_num % jnp.int32(_N_SCREENS)

        # Wrap Harry to opposite side of new screen on transition
        harry_x = jnp.where(
            at_right,
            jnp.float32(_SCREEN_LEFT + 2.0),
            jnp.where(at_left, jnp.float32(_SCREEN_RIGHT - _HARRY_W - 2.0), harry_x),
        )

        # Treasure activates on every 8th screen, resets on non-treasure screens
        new_treasure_active = jnp.where(
            at_right | at_left,
            (screen_num % jnp.int32(8)) == jnp.int32(0),
            state.treasure_active,
        )

        # --- Log collision (surface only, with cooldown) ---
        harry_top = harry_y - jnp.float32(_HARRY_H)
        harry_right = harry_x + jnp.float32(_HARRY_W)
        log_right = new_log_x + jnp.float32(_LOG_W)

        log_top_f = jnp.float32(_LOG_TOP)
        log_bottom_f = jnp.float32(_GROUND_Y)

        def _log_hit(lx: jax.Array, lxr: jax.Array) -> jax.Array:
            return (
                ~is_underground
                & (harry_right > lx)
                & (harry_x < lxr)
                & (harry_y > log_top_f)
                & (harry_top < log_bottom_f)
            )

        hit_any_log = (
            _log_hit(new_log_x[0], log_right[0])
            | _log_hit(new_log_x[1], log_right[1])
            | _log_hit(new_log_x[2], log_right[2])
        )

        log_cooldown = jnp.maximum(state.log_cooldown - jnp.int32(1), jnp.int32(0))
        apply_log_penalty = hit_any_log & (log_cooldown <= jnp.int32(0))
        log_reward = jnp.where(
            apply_log_penalty, jnp.float32(_LOG_PENALTY), jnp.float32(0.0)
        )
        log_cooldown = jnp.where(
            apply_log_penalty, jnp.int32(_LOG_COOLDOWN_FRAMES), log_cooldown
        )

        # --- Crocodile collision (surface only, croc mouth open) ---
        croc_top_f = jnp.float32(_CROC_TOP)

        def _croc_hit(cx: float) -> jax.Array:
            return (
                ~is_underground
                & croc_open
                & (harry_right > jnp.float32(cx))
                & (harry_x < jnp.float32(cx + _CROC_W))
                & (harry_y > croc_top_f)
                & (harry_top < jnp.float32(_GROUND_Y))
            )

        hit_any_croc = _croc_hit(_CROC_X[0]) | _croc_hit(_CROC_X[1])

        new_lives = state.lives - jnp.where(hit_any_croc, jnp.int32(1), jnp.int32(0))
        # Reset Harry to centre of screen on croc hit
        harry_x = jnp.where(hit_any_croc, jnp.float32(_HARRY_INIT_X), harry_x)
        harry_y = jnp.where(hit_any_croc, jnp.float32(_GROUND_Y), harry_y)
        harry_vy = jnp.where(hit_any_croc, jnp.float32(0.0), harry_vy)
        is_jumping = is_jumping & ~hit_any_croc
        is_underground = is_underground & ~hit_any_croc

        # --- Treasure collection ---
        treasure_right_f = jnp.float32(_TREASURE_X + _TREASURE_W)
        treasure_top_f = jnp.float32(_TREASURE_TOP)

        hit_treasure = (
            new_treasure_active
            & ~is_underground
            & (harry_right > jnp.float32(_TREASURE_X))
            & (harry_x < treasure_right_f)
            & (harry_y > treasure_top_f)
            & (harry_top < jnp.float32(_GROUND_Y))
        )
        treasure_reward = jnp.where(
            hit_treasure, jnp.float32(_TREASURE_REWARD), jnp.float32(0.0)
        )
        new_treasure_active = new_treasure_active & ~hit_treasure

        # --- Step reward and accumulated score ---
        step_reward = log_reward + treasure_reward
        new_score = state.score + jnp.int32(step_reward)

        # --- Terminal ---
        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            harry_x=harry_x,
            harry_y=harry_y,
            harry_vy=harry_vy,
            is_jumping=is_jumping,
            is_underground=is_underground,
            log_x=new_log_x,
            croc_timer=croc_timer,
            log_cooldown=log_cooldown,
            treasure_active=new_treasure_active,
            lives=new_lives,
            score=new_score,
            level=screen_num,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=state.key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: PitfallState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> PitfallState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: PitfallState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Background: jungle canopy (top), floor strip, ground, underground ---
        canopy_mask = _ROW_IDX_R < 140
        frame = jnp.where(canopy_mask[:, :, None], _BG_TOP_COLOR[None, None, :], frame)

        floor_mask = (_ROW_IDX_R >= 140) & (_ROW_IDX_R < 148)
        frame = jnp.where(floor_mask[:, :, None], _BG_MID_COLOR[None, None, :], frame)

        ground_mask = (_ROW_IDX_R >= 148) & (_ROW_IDX_R < 167)
        frame = jnp.where(ground_mask[:, :, None], _GROUND_COLOR[None, None, :], frame)

        ug_mask = (_ROW_IDX_R >= 167) & (_ROW_IDX_R < 192)
        frame = jnp.where(ug_mask[:, :, None], _UG_COLOR[None, None, :], frame)

        # --- Logs ---
        for i in range(3):
            lx = jnp.int32(state.log_x[i])
            log_mask = (
                (_ROW_IDX_R >= jnp.int32(_LOG_TOP))
                & (_ROW_IDX_R < jnp.int32(_GROUND_Y))
                & (_COL_IDX_R >= lx)
                & (_COL_IDX_R < lx + _LOG_W)
            )
            frame = jnp.where(log_mask[:, :, None], _LOG_COLOR[None, None, :], frame)

        # --- Crocodiles ---
        croc_open = state.croc_timer > jnp.int32(_CROC_OPEN_AT)
        croc_color = jnp.where(croc_open, _CROC_OPEN_COLOR, _CROC_SAFE_COLOR)
        for cx in _CROC_X:
            croc_mask = (
                (_ROW_IDX_R >= jnp.int32(_CROC_TOP))
                & (_ROW_IDX_R < jnp.int32(_GROUND_Y))
                & (_COL_IDX_R >= jnp.int32(cx))
                & (_COL_IDX_R < jnp.int32(cx + _CROC_W))
            )
            frame = jnp.where(croc_mask[:, :, None], croc_color[None, None, :], frame)

        # --- Treasure ---
        treas_mask = (
            state.treasure_active
            & (_ROW_IDX_R >= jnp.int32(_TREASURE_TOP))
            & (_ROW_IDX_R < jnp.int32(_GROUND_Y))
            & (_COL_IDX_R >= jnp.int32(_TREASURE_X))
            & (_COL_IDX_R < jnp.int32(_TREASURE_X + _TREASURE_W))
        )
        frame = jnp.where(treas_mask[:, :, None], _TREASURE_COLOR[None, None, :], frame)

        # --- Harry ---
        hx = jnp.int32(state.harry_x)
        hy_top = jnp.int32(state.harry_y) - _HARRY_H
        hy_bot = jnp.int32(state.harry_y)
        harry_mask = (
            (_ROW_IDX_R >= hy_top)
            & (_ROW_IDX_R < hy_bot)
            & (_COL_IDX_R >= hx)
            & (_COL_IDX_R < hx + _HARRY_W)
        )
        frame = jnp.where(harry_mask[:, :, None], _HARRY_COLOR[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
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
