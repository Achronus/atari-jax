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

"""Bank Heist — JAX-native game implementation.

Drive a getaway car through a scrolling city robbing banks and evading
police.  Lay dynamite bombs to destroy pursuing police cars.

Action space (18 actions, minimal set used):
    0 — NOOP
    1 — FIRE  (drop dynamite)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — UP + FIRE
    7 — RIGHT + FIRE
    8 — DOWN + FIRE
    9 — LEFT + FIRE

Scoring:
    Bank robbed  — +50
    Police blown up — +25
    Episode ends when all lives (fuel / getaway cars) are lost; lives: 4.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_BANKS: int = 8
_N_POLICE: int = 4
_N_DYNOS: int = 3  # max dynamite bombs

_SCREEN_W: int = 160
_SCREEN_H: int = 210
_PLAYER_SPEED: float = 2.0
_POLICE_SPEED: float = 1.5
_DYNO_SIZE: int = 8

_INIT_LIVES: int = 4  # getaway cars

_BANK_POSITIONS_X = jnp.array(
    [20.0, 60.0, 100.0, 140.0, 20.0, 60.0, 100.0, 140.0], dtype=jnp.float32
)
_BANK_POSITIONS_Y = jnp.array(
    [50.0, 50.0, 50.0, 50.0, 120.0, 120.0, 120.0, 120.0], dtype=jnp.float32
)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([30, 30, 50], dtype=jnp.uint8)
_COLOR_ROAD = jnp.array([60, 60, 60], dtype=jnp.uint8)
_COLOR_BANK = jnp.array([180, 160, 40], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([80, 200, 80], dtype=jnp.uint8)
_COLOR_POLICE = jnp.array([60, 60, 200], dtype=jnp.uint8)
_COLOR_DYNO = jnp.array([220, 100, 20], dtype=jnp.uint8)


@chex.dataclass
class BankHeistState(AtariState):
    """
    Complete Bank Heist game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Getaway car x.
    player_y : jax.Array
        float32 — Getaway car y.
    player_dir : jax.Array
        int32 — Direction (0=up, 1=right, 2=down, 3=left).
    bank_robbed : jax.Array
        bool[8] — Banks already robbed.
    police_x : jax.Array
        float32[4] — Police car x positions.
    police_y : jax.Array
        float32[4] — Police car y positions.
    police_active : jax.Array
        bool[4] — Police cars pursuing.
    dyno_x : jax.Array
        float32[3] — Dynamite x positions.
    dyno_y : jax.Array
        float32[3] — Dynamite y positions.
    dyno_active : jax.Array
        bool[3] — Dynamite placed.
    spawn_timer : jax.Array
        int32 — Frames until next police spawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_dir: jax.Array
    bank_robbed: jax.Array
    police_x: jax.Array
    police_y: jax.Array
    police_active: jax.Array
    dyno_x: jax.Array
    dyno_y: jax.Array
    dyno_active: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


class BankHeist(AtariEnv):
    """
    Bank Heist implemented as a pure JAX function suite.

    Rob banks and evade police.  Lives (getaway cars): 4.
    """

    num_actions: int = 10

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> BankHeistState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : BankHeistState
            Car at start, all banks unrobbed, 4 lives.
        """
        return BankHeistState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(170.0),
            player_dir=jnp.int32(0),
            bank_robbed=jnp.zeros(_N_BANKS, dtype=jnp.bool_),
            police_x=jnp.full(_N_POLICE, -20.0, dtype=jnp.float32),
            police_y=jnp.full(_N_POLICE, -20.0, dtype=jnp.float32),
            police_active=jnp.zeros(_N_POLICE, dtype=jnp.bool_),
            dyno_x=jnp.full(_N_DYNOS, -10.0, dtype=jnp.float32),
            dyno_y=jnp.full(_N_DYNOS, -10.0, dtype=jnp.float32),
            dyno_active=jnp.zeros(_N_DYNOS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(80),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: BankHeistState, action: jax.Array) -> BankHeistState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : BankHeistState
            Current game state.
        action : jax.Array
            int32 — Action index (0–9).

        Returns
        -------
        new_state : BankHeistState
            State after one emulated frame.
        """
        key, sk = jax.random.split(state.key)
        step_reward = jnp.float32(0.0)

        # Player movement
        _dy = jnp.array(
            [
                0,
                -_PLAYER_SPEED,
                0,
                _PLAYER_SPEED,
                0,
                _PLAYER_SPEED,
                0,
                _PLAYER_SPEED,
                _PLAYER_SPEED,
                _PLAYER_SPEED,
            ],
            dtype=jnp.float32,
        )
        _dx = jnp.array(
            [
                0,
                0,
                _PLAYER_SPEED,
                0,
                -_PLAYER_SPEED,
                0,
                _PLAYER_SPEED,
                0,
                -_PLAYER_SPEED,
                0,
            ],
            dtype=jnp.float32,
        )

        # Simplified: map actions to dx/dy
        dx = (
            jnp.where(action == jnp.int32(3), _PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(5), -_PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(7), _PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(9), -_PLAYER_SPEED, 0.0)
        )
        dy = (
            jnp.where(action == jnp.int32(2), -_PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(4), _PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(6), -_PLAYER_SPEED, 0.0)
            + jnp.where(action == jnp.int32(8), _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(state.player_x + dx, 5.0, 155.0)
        new_py = jnp.clip(state.player_y + dy, 20.0, 195.0)

        # Drop dynamite
        do_drop = (
            (action == jnp.int32(1))
            | (action == jnp.int32(6))
            | (action == jnp.int32(7))
            | (action == jnp.int32(8))
            | (action == jnp.int32(9))
        )
        free_dyno = jnp.argmin(state.dyno_active.astype(jnp.int32))
        has_free = ~jnp.all(state.dyno_active)
        new_dyno_x = jnp.where(
            do_drop & has_free,
            state.dyno_x.at[free_dyno].set(state.player_x),
            state.dyno_x,
        )
        new_dyno_y = jnp.where(
            do_drop & has_free,
            state.dyno_y.at[free_dyno].set(state.player_y),
            state.dyno_y,
        )
        new_dyno_active = jnp.where(
            do_drop & has_free,
            state.dyno_active.at[free_dyno].set(True),
            state.dyno_active,
        )

        # Police movement (chase player)
        pdx = jnp.sign(state.player_x - state.police_x) * _POLICE_SPEED
        pdy = jnp.sign(state.player_y - state.police_y) * _POLICE_SPEED
        new_police_x = state.police_x + jnp.where(state.police_active, pdx, 0.0)
        new_police_y = state.police_y + jnp.where(state.police_active, pdy, 0.0)

        # Dynamite explodes when police touches it
        dyno_hits_police = (
            new_dyno_active[:, None]
            & state.police_active[None, :]
            & (jnp.abs(new_dyno_x[:, None] - new_police_x[None, :]) < _DYNO_SIZE)
            & (jnp.abs(new_dyno_y[:, None] - new_police_y[None, :]) < _DYNO_SIZE)
        )
        police_killed = jnp.any(dyno_hits_police, axis=0)
        dyno_used = jnp.any(dyno_hits_police, axis=1)
        n_killed = jnp.sum(police_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_killed * 25)
        new_police_active = state.police_active & ~police_killed
        new_dyno_active = new_dyno_active & ~dyno_used

        # Rob a bank (player touches bank)
        near_bank = (
            (jnp.abs(_BANK_POSITIONS_X - new_px) < 12.0)
            & (jnp.abs(_BANK_POSITIONS_Y - new_py) < 12.0)
            & ~state.bank_robbed
        )
        n_robbed = jnp.sum(near_bank).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_robbed * 50)
        new_bank_robbed = state.bank_robbed | near_bank

        # Spawn police
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(do_spawn, jnp.int32(80), new_spawn_timer)
        free_police = jnp.argmin(new_police_active.astype(jnp.int32))
        spawn_x = jax.random.uniform(sk) * 120.0 + 20.0
        new_police_x = jnp.where(
            do_spawn, new_police_x.at[free_police].set(spawn_x), new_police_x
        )
        new_police_y = jnp.where(
            do_spawn, new_police_y.at[free_police].set(jnp.float32(20.0)), new_police_y
        )
        new_police_active = jnp.where(
            do_spawn, new_police_active.at[free_police].set(True), new_police_active
        )

        # Police catches player
        police_catch = (
            new_police_active
            & (jnp.abs(new_police_x - new_px) < 10.0)
            & (jnp.abs(new_police_y - new_py) < 10.0)
        )
        caught = jnp.any(police_catch)
        new_lives = state.lives - jnp.where(caught, jnp.int32(1), jnp.int32(0))

        # All banks robbed: reset
        all_robbed = jnp.all(new_bank_robbed)
        new_bank_robbed = jnp.where(
            all_robbed, jnp.zeros(_N_BANKS, dtype=jnp.bool_), new_bank_robbed
        )

        done = new_lives <= jnp.int32(0)

        return BankHeistState(
            player_x=new_px,
            player_y=new_py,
            player_dir=state.player_dir,
            bank_robbed=new_bank_robbed,
            police_x=new_police_x,
            police_y=new_police_y,
            police_active=new_police_active,
            dyno_x=new_dyno_x,
            dyno_y=new_dyno_y,
            dyno_active=new_dyno_active,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: BankHeistState, action: jax.Array) -> BankHeistState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : BankHeistState
            Current game state.
        action : jax.Array
            int32 — Action index (0–9).

        Returns
        -------
        new_state : BankHeistState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: BankHeistState) -> BankHeistState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: BankHeistState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : BankHeistState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Roads
        h_road = (_ROW_IDX == 80) | (_ROW_IDX == 150) | (_ROW_IDX == 170)
        v_road = (_COL_IDX == 20) | (_COL_IDX == 80) | (_COL_IDX == 140)
        frame = jnp.where((h_road | v_road)[:, :, None], _COLOR_ROAD, frame)

        # Banks
        def draw_bank(frm, i):
            bx = _BANK_POSITIONS_X[i].astype(jnp.int32)
            by = _BANK_POSITIONS_Y[i].astype(jnp.int32)
            color = jnp.where(state.bank_robbed[i], _COLOR_ROAD, _COLOR_BANK)
            mask = (
                (_ROW_IDX >= by - 8)
                & (_ROW_IDX <= by + 8)
                & (_COL_IDX >= bx - 8)
                & (_COL_IDX <= bx + 8)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_bank, frame, jnp.arange(_N_BANKS))

        # Police
        def draw_police(frm, i):
            px = state.police_x[i].astype(jnp.int32)
            py = state.police_y[i].astype(jnp.int32)
            mask = (
                state.police_active[i]
                & (_ROW_IDX >= py - 5)
                & (_ROW_IDX <= py + 5)
                & (_COL_IDX >= px - 6)
                & (_COL_IDX <= px + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_POLICE, frm), None

        frame, _ = jax.lax.scan(draw_police, frame, jnp.arange(_N_POLICE))

        # Dynamite
        def draw_dyno(frm, i):
            dx = state.dyno_x[i].astype(jnp.int32)
            dy = state.dyno_y[i].astype(jnp.int32)
            mask = (
                state.dyno_active[i]
                & (_ROW_IDX >= dy - 3)
                & (_ROW_IDX <= dy + 3)
                & (_COL_IDX >= dx - 3)
                & (_COL_IDX <= dx + 3)
            )
            return jnp.where(mask[:, :, None], _COLOR_DYNO, frm), None

        frame, _ = jax.lax.scan(draw_dyno, frame, jnp.arange(_N_DYNOS))

        # Player car
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 6)
            & (_ROW_IDX <= py + 6)
            & (_COL_IDX >= px - 6)
            & (_COL_IDX <= px + 6)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Bank Heist action indices.
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
