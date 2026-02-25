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

"""Space Invaders — JAX-native game implementation.

Mechanics implemented directly in JAX with no hardware emulation.
All conditionals use `jnp.where`; the step loop uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Playfield    : x ∈ [8, 152),  y ∈ [30, 185)
    Cannon       : y ∈ [177, 185), w=13 px; x ∈ [8, 139)
    Alien grid   : 5 rows × 11 cols, 8×8 px each, 10 px col step, 16 px row step
    Player bullet: 1×4 px, moves up 4 px/frame
    Alien bullet : 1×4 px, moves down 2 px/frame
    Ground line  : y = 185

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — RIGHT + FIRE
    5 — LEFT + FIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
_PLAY_LEFT: int = 8
_PLAY_RIGHT: int = 152
_PLAY_TOP: int = 30
_GROUND_Y: int = 185

_CANNON_W: int = 13
_CANNON_H: int = 8
_CANNON_Y: int = 177
_CANNON_SPEED: float = 2.0
_CANNON_X_MIN: float = 8.0
_CANNON_X_MAX: float = 139.0  # _PLAY_RIGHT - _CANNON_W

_ALIEN_ROWS: int = 5
_ALIEN_COLS: int = 11
_ALIEN_W: int = 8
_ALIEN_H: int = 8
_COL_STEP: int = 10  # horizontal spacing between alien column origins
_ROW_STEP: int = 16  # vertical spacing between alien row origins
_ALIEN_INIT_X: float = 26.0  # initial left edge of leftmost column
_ALIEN_INIT_Y: float = 50.0  # initial top edge of top row
_ALIEN_STEP_X: float = 8.0  # horizontal displacement per move event
_ALIEN_DROP_Y: float = 8.0  # vertical drop on edge reversal

_BULLET_W: int = 1
_BULLET_H: int = 4
_PLAYER_BULLET_SPEED: float = 4.0
_ALIEN_BULLET_SPEED: float = 2.0

_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4
_ALIEN_MOVE_INITIAL: int = 6  # sub-steps between moves at full count
_ALIEN_FIRE_INTERVAL: int = 24  # sub-steps between alien shots

# Row scores (row 0 = top → highest points)
_ROW_SCORES = jnp.array([30, 20, 20, 10, 10], dtype=jnp.float32)

# Alien colours by row (top → bottom)
_ALIEN_COLORS = jnp.array(
    [
        [255, 255, 255],  # row 0: white
        [255, 220, 80],  # row 1: yellow
        [255, 220, 80],  # row 2: yellow
        [80, 200, 80],  # row 3: green
        [80, 200, 80],  # row 4: green
    ],
    dtype=jnp.uint8,
)

# Precomputed index arrays for branch-free rendering
_ROW_IDX = jnp.arange(210)[:, None]  # [210, 1]
_COL_IDX = jnp.arange(160)[None, :]  # [1, 160]

# Precomputed alien grid offsets (float32 for arithmetic)
_ALIEN_COL_OFFSETS = jnp.arange(_ALIEN_COLS, dtype=jnp.float32) * _COL_STEP  # [11]
_ALIEN_ROW_OFFSETS = jnp.arange(_ALIEN_ROWS, dtype=jnp.float32) * _ROW_STEP  # [5]


@chex.dataclass
class SpaceInvadersState(AtariState):
    """
    Complete Space Invaders game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score` from `AtariState`.

    Parameters
    ----------
    aliens : jax.Array
        bool[5, 11] — Active aliens.  `True` = alien present.
    alien_x : jax.Array
        float32 — Left edge of column 0 of the alien formation.
    alien_y : jax.Array
        float32 — Top edge of row 0 of the alien formation.
    alien_dx : jax.Array
        float32 — Formation step: +`_ALIEN_STEP_X` (right) or −`_ALIEN_STEP_X` (left).
    player_x : jax.Array
        float32 — Cannon left-edge x coordinate.
    player_bullet_x : jax.Array
        float32 — Player bullet left-edge x.
    player_bullet_y : jax.Array
        float32 — Player bullet top-edge y.
    player_bullet_active : jax.Array
        bool — `True` while bullet is in-flight.
    alien_bullet_x : jax.Array
        float32 — Alien bullet left-edge x.
    alien_bullet_y : jax.Array
        float32 — Alien bullet top-edge y.
    alien_bullet_active : jax.Array
        bool — `True` while alien bullet is in-flight.
    move_timer : jax.Array
        int32 — Sub-steps until the next alien formation move.
    fire_timer : jax.Array
        int32 — Sub-steps until the aliens fire next.
    key : jax.Array
        uint32[2] — PRNG key evolved each frame for stochastic alien AI.
    """

    aliens: jax.Array
    alien_x: jax.Array
    alien_y: jax.Array
    alien_dx: jax.Array
    player_x: jax.Array
    player_bullet_x: jax.Array
    player_bullet_y: jax.Array
    player_bullet_active: jax.Array
    alien_bullet_x: jax.Array
    alien_bullet_y: jax.Array
    alien_bullet_active: jax.Array
    move_timer: jax.Array
    fire_timer: jax.Array
    key: jax.Array


class SpaceInvaders(AtariEnv):
    """
    Space Invaders implemented as a pure JAX function suite.

    No hardware emulation — game physics are computed directly using
    `jnp.where` for all conditionals and `jax.lax.fori_loop` for the
    4-frame skip inside `_step`.

    The alien formation accelerates as invaders are destroyed, matching
    the feel of the original arcade game.
    """

    num_actions: int = 6

    def _reset(self, key: jax.Array) -> SpaceInvadersState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : SpaceInvadersState
            Full 5×11 alien grid, cannon centred, no bullets in flight,
            3 lives, scores zero.
        """
        cannon_x = jnp.float32((_PLAY_LEFT + _PLAY_RIGHT - _CANNON_W) / 2)

        return SpaceInvadersState(
            aliens=jnp.ones((_ALIEN_ROWS, _ALIEN_COLS), dtype=jnp.bool_),
            alien_x=jnp.float32(_ALIEN_INIT_X),
            alien_y=jnp.float32(_ALIEN_INIT_Y),
            alien_dx=jnp.float32(_ALIEN_STEP_X),
            player_x=cannon_x,
            player_bullet_x=jnp.float32(0.0),
            player_bullet_y=jnp.float32(0.0),
            player_bullet_active=jnp.bool_(False),
            alien_bullet_x=jnp.float32(0.0),
            alien_bullet_y=jnp.float32(0.0),
            alien_bullet_active=jnp.bool_(False),
            move_timer=jnp.int32(_ALIEN_MOVE_INITIAL),
            fire_timer=jnp.int32(_ALIEN_FIRE_INTERVAL),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: SpaceInvadersState, action: jax.Array
    ) -> SpaceInvadersState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : SpaceInvadersState
            Current game state.
        action : jax.Array
            int32 — Action index (0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT,
            4=RIGHT+FIRE, 5=LEFT+FIRE).

        Returns
        -------
        new_state : SpaceInvadersState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Player cannon movement ---
        move_right = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_left = (action == jnp.int32(3)) | (action == jnp.int32(5))
        delta = jnp.where(
            move_right,
            jnp.float32(_CANNON_SPEED),
            jnp.where(move_left, jnp.float32(-_CANNON_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(
            state.player_x + delta,
            jnp.float32(_CANNON_X_MIN),
            jnp.float32(_CANNON_X_MAX),
        )

        # --- Player fires ---
        wants_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(4))
            | (action == jnp.int32(5))
        )
        fire = wants_fire & ~state.player_bullet_active
        new_pbx = jnp.where(
            fire,
            player_x + jnp.float32((_CANNON_W - _BULLET_W) / 2),
            state.player_bullet_x,
        )
        new_pby = jnp.where(
            fire,
            jnp.float32(_CANNON_Y - _BULLET_H),
            state.player_bullet_y,
        )
        player_bullet_active = state.player_bullet_active | fire

        # --- Move player bullet (upward) ---
        new_pby = jnp.where(
            player_bullet_active,
            new_pby - jnp.float32(_PLAYER_BULLET_SPEED),
            new_pby,
        )
        pb_oob = player_bullet_active & (new_pby < jnp.float32(_PLAY_TOP))
        player_bullet_active = player_bullet_active & ~pb_oob

        # --- Move alien bullet (downward) ---
        new_aby = jnp.where(
            state.alien_bullet_active,
            state.alien_bullet_y + jnp.float32(_ALIEN_BULLET_SPEED),
            state.alien_bullet_y,
        )
        alien_bullet_active = state.alien_bullet_active
        ab_oob = alien_bullet_active & (
            new_aby + jnp.float32(_BULLET_H) > jnp.float32(_GROUND_Y)
        )
        alien_bullet_active = alien_bullet_active & ~ab_oob

        # --- Player bullet vs alien collision ---
        alien_lefts = state.alien_x + _ALIEN_COL_OFFSETS[None, :]  # [1, 11]
        alien_tops = state.alien_y + _ALIEN_ROW_OFFSETS[:, None]  # [5, 1]
        hit_mask = (
            (new_pbx + jnp.float32(_BULLET_W) > alien_lefts)
            & (new_pbx < alien_lefts + jnp.float32(_ALIEN_W))
            & (new_pby + jnp.float32(_BULLET_H) > alien_tops)
            & (new_pby < alien_tops + jnp.float32(_ALIEN_H))
            & state.aliens
            & player_bullet_active
        )  # bool[5, 11]
        any_alien_hit = jnp.any(hit_mask)
        new_aliens = state.aliens & ~hit_mask
        step_reward = jnp.sum(hit_mask * _ROW_SCORES[:, None])
        player_bullet_active = player_bullet_active & ~any_alien_hit

        # --- Alien bullet vs cannon collision ---
        abx = state.alien_bullet_x
        ab_hit_cannon = (
            alien_bullet_active
            & (abx + jnp.float32(_BULLET_W) > player_x)
            & (abx < player_x + jnp.float32(_CANNON_W))
            & (new_aby + jnp.float32(_BULLET_H) > jnp.float32(_CANNON_Y))
            & (new_aby < jnp.float32(_CANNON_Y + _CANNON_H))
        )
        new_lives = state.lives - jnp.where(ab_hit_cannon, jnp.int32(1), jnp.int32(0))
        alien_bullet_active = alien_bullet_active & ~ab_hit_cannon

        # --- Alien formation movement ---
        should_move = state.move_timer <= jnp.int32(0)
        proposed_x = state.alien_x + state.alien_dx
        formation_right = proposed_x + jnp.float32(
            (_ALIEN_COLS - 1) * _COL_STEP + _ALIEN_W
        )
        edge_hit = (formation_right > jnp.float32(_PLAY_RIGHT)) | (
            proposed_x < jnp.float32(_PLAY_LEFT)
        )
        new_alien_dx = jnp.where(
            should_move & edge_hit, -state.alien_dx, state.alien_dx
        )
        new_alien_y = jnp.where(
            should_move & edge_hit,
            state.alien_y + jnp.float32(_ALIEN_DROP_Y),
            state.alien_y,
        )
        new_alien_x = jnp.where(
            should_move & ~edge_hit,
            state.alien_x + state.alien_dx,
            state.alien_x,
        )
        n_alive = jnp.sum(new_aliens).astype(jnp.int32)
        new_interval = jnp.maximum(
            jnp.int32(1),
            n_alive
            * jnp.int32(_ALIEN_MOVE_INITIAL)
            // jnp.int32(_ALIEN_ROWS * _ALIEN_COLS),
        )
        new_move_timer = jnp.where(
            should_move, new_interval, state.move_timer - jnp.int32(1)
        )

        # --- Alien firing ---
        col_idx = jax.random.randint(subkey, (), 0, _ALIEN_COLS)
        col_aliens = jnp.take(new_aliens, col_idx, axis=1)  # bool[5]
        has_col_alien = jnp.any(col_aliens)
        bot_from_bottom = jnp.argmax(col_aliens[::-1])
        spawn_row = jnp.int32(_ALIEN_ROWS - 1) - bot_from_bottom
        spawn_abx = (
            new_alien_x
            + jnp.float32(col_idx) * jnp.float32(_COL_STEP)
            + jnp.float32((_ALIEN_W - _BULLET_W) / 2)
        )
        spawn_aby = (
            new_alien_y
            + jnp.float32(spawn_row) * jnp.float32(_ROW_STEP)
            + jnp.float32(_ALIEN_H)
        )
        should_alien_fire = (
            (state.fire_timer <= jnp.int32(0))
            & ~state.alien_bullet_active
            & has_col_alien
            & (n_alive > jnp.int32(0))
        )
        new_abx = jnp.where(should_alien_fire, spawn_abx, abx)
        new_aby_final = jnp.where(should_alien_fire, spawn_aby, new_aby)
        alien_bullet_active = jnp.where(
            should_alien_fire, jnp.bool_(True), alien_bullet_active
        )
        new_fire_timer = jnp.where(
            state.fire_timer <= jnp.int32(0),
            jnp.int32(_ALIEN_FIRE_INTERVAL),
            state.fire_timer - jnp.int32(1),
        )

        # --- Episode end conditions ---
        all_dead = ~jnp.any(new_aliens)
        formation_bottom = new_alien_y + jnp.float32(
            (_ALIEN_ROWS - 1) * _ROW_STEP + _ALIEN_H
        )
        aliens_at_ground = formation_bottom >= jnp.float32(_GROUND_Y)
        done = all_dead | aliens_at_ground | (new_lives <= jnp.int32(0))

        return SpaceInvadersState(
            aliens=new_aliens,
            alien_x=new_alien_x,
            alien_y=new_alien_y,
            alien_dx=new_alien_dx,
            player_x=player_x,
            player_bullet_x=new_pbx,
            player_bullet_y=new_pby,
            player_bullet_active=player_bullet_active,
            alien_bullet_x=new_abx,
            alien_bullet_y=new_aby_final,
            alien_bullet_active=alien_bullet_active,
            move_timer=new_move_timer,
            fire_timer=new_fire_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
            key=key,
        )

    def _step(self, state: SpaceInvadersState, action: jax.Array) -> SpaceInvadersState:
        """
        Advance the game by one agent step (4 emulated frames).

        The reward is accumulated across all 4 frames, matching the ALE
        frame-skip convention.

        Parameters
        ----------
        state : SpaceInvadersState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : SpaceInvadersState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: SpaceInvadersState) -> SpaceInvadersState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, _FRAME_SKIP, body, state)

    def render(self, state: SpaceInvadersState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : SpaceInvadersState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Ground line (dim white)
        ground_mask = (_ROW_IDX == _GROUND_Y) & (
            (_COL_IDX >= _PLAY_LEFT) & (_COL_IDX < _PLAY_RIGHT)
        )
        frame = jnp.where(ground_mask[:, :, None], jnp.uint8(100), frame)

        # --- Alien grid ---
        # For each pixel, determine which alien cell it falls into and
        # whether that alien is alive.  This avoids loops entirely.
        ax = jnp.int32(state.alien_x)
        ay = jnp.int32(state.alien_y)
        dy = _ROW_IDX - ay  # [210, 1]
        dx = _COL_IDX - ax  # [1, 160]

        in_bounds_y = (dy >= 0) & (dy < jnp.int32(_ALIEN_ROWS * _ROW_STEP))
        in_bounds_x = (dx >= 0) & (dx < jnp.int32(_ALIEN_COLS * _COL_STEP))

        alien_row_idx = jnp.clip(dy // _ROW_STEP, 0, _ALIEN_ROWS - 1)  # [210, 1]
        alien_col_idx = jnp.clip(dx // _COL_STEP, 0, _ALIEN_COLS - 1)  # [1, 160]

        in_cell_y = (dy % _ROW_STEP) < _ALIEN_H  # [210, 1]
        in_cell_x = (dx % _COL_STEP) < _ALIEN_W  # [1, 160]

        # Advanced gather: aliens[[210,1], [1,160]] → [210, 160]
        alien_alive = state.aliens[alien_row_idx, alien_col_idx]
        alien_pixel_mask = (
            in_bounds_y & in_bounds_x & in_cell_y & in_cell_x & alien_alive
        )

        # Per-row colour lookup
        alien_row_color = _ALIEN_COLORS[alien_row_idx[:, 0], :]  # [210, 3]
        frame = jnp.where(
            alien_pixel_mask[:, :, None], alien_row_color[:, None, :], frame
        )

        # --- Cannon ---
        px = jnp.int32(state.player_x)
        cannon_mask = (
            (_ROW_IDX >= _CANNON_Y)
            & (_ROW_IDX < _CANNON_Y + _CANNON_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _CANNON_W)
        )
        frame = jnp.where(cannon_mask[:, :, None], jnp.uint8(120), frame)

        # --- Player bullet (white) ---
        pbx = jnp.int32(state.player_bullet_x)
        pby = jnp.int32(state.player_bullet_y)
        pb_mask = (
            state.player_bullet_active
            & (_ROW_IDX >= pby)
            & (_ROW_IDX < pby + _BULLET_H)
            & (_COL_IDX >= pbx)
            & (_COL_IDX < pbx + _BULLET_W)
        )
        frame = jnp.where(pb_mask[:, :, None], jnp.uint8(255), frame)

        # --- Alien bullet (red-tinted) ---
        abx_int = jnp.int32(state.alien_bullet_x)
        aby_int = jnp.int32(state.alien_bullet_y)
        ab_mask = (
            state.alien_bullet_active
            & (_ROW_IDX >= aby_int)
            & (_ROW_IDX < aby_int + _BULLET_H)
            & (_COL_IDX >= abx_int)
            & (_COL_IDX < abx_int + _BULLET_W)
        )
        alien_bullet_color = jnp.array([255, 80, 80], dtype=jnp.uint8)
        frame = jnp.where(ab_mask[:, :, None], alien_bullet_color, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Space Invaders action indices.
            Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHT+FIRE, 5=LEFT+FIRE.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
        }
