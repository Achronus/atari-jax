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

"""Yars' Revenge — JAX-native game implementation.

As a giant fly, nibble through the Qotile shield to create a gap, then
fire the Zorlon Cannon through the gap to destroy the Qotile.  The Quotile
shoots a swirl that must be dodged; there's also a neutral zone where
nothing can fire.

Action space (7 actions):
    0 — NOOP
    1 — FIRE (nibble shield / fire cannon)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — FIRE+UP

Scoring:
    Shield block nibbled — +69
    Qotile destroyed (cannon hit) — +6000+
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_SHIELD_X: int = 120  # shield column x
_QOTILE_X: int = 148  # Qotile x (right side)
_NEUTRAL_X_LEFT: int = 60
_NEUTRAL_X_RIGHT: int = 80
_N_SHIELD: int = 16  # shield blocks (vertical)

_PLAYER_SPEED: float = 2.0
_CANNON_SPEED: float = 6.0
_SWIRL_SPEED: float = 2.0

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([100, 200, 80], dtype=jnp.uint8)
_COLOR_SHIELD = jnp.array([0, 120, 200], dtype=jnp.uint8)
_COLOR_QOTILE = jnp.array([200, 50, 200], dtype=jnp.uint8)
_COLOR_CANNON = jnp.array([255, 255, 0], dtype=jnp.uint8)
_COLOR_SWIRL = jnp.array([255, 100, 0], dtype=jnp.uint8)
_COLOR_NEUTRAL = jnp.array([20, 20, 60], dtype=jnp.uint8)

_SHIELD_YS = jnp.linspace(30.0, 185.0, _N_SHIELD, dtype=jnp.float32)
_SHIELD_SPACING: float = (
    (_SHIELD_YS[1] - _SHIELD_YS[0]).item() if _N_SHIELD > 1 else 10.0
)


@chex.dataclass
class YarsRevengeState(AtariState):
    """
    Complete Yars' Revenge game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    shield : jax.Array
        bool[16] — Shield blocks remaining.
    cannon_x : jax.Array
        float32 — Zorlon Cannon x.
    cannon_y : jax.Array
        float32 — Zorlon Cannon y.
    cannon_active : jax.Array
        bool — Cannon in flight.
    qotile_y : jax.Array
        float32 — Qotile y (tracks player).
    swirl_x : jax.Array
        float32 — Swirl x.
    swirl_y : jax.Array
        float32 — Swirl y.
    swirl_active : jax.Array
        bool — Swirl in flight.
    cannon_charged : jax.Array
        bool — Cannon charged (enough shield nibbled).
    nibble_count : jax.Array
        int32 — Shield blocks nibbled.
    fire_timer : jax.Array
        int32 — Qotile fire delay.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    shield: jax.Array
    cannon_x: jax.Array
    cannon_y: jax.Array
    cannon_active: jax.Array
    qotile_y: jax.Array
    swirl_x: jax.Array
    swirl_y: jax.Array
    swirl_active: jax.Array
    cannon_charged: jax.Array
    nibble_count: jax.Array
    fire_timer: jax.Array
    key: jax.Array


class YarsRevenge(AtariEnv):
    """
    Yars' Revenge implemented as a pure JAX function suite.

    Nibble the shield, then destroy the Qotile.  Lives: 3.
    """

    num_actions: int = 7

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> YarsRevengeState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : YarsRevengeState
            Player at left, full shield, Qotile at right.
        """
        return YarsRevengeState(
            player_x=jnp.float32(10.0),
            player_y=jnp.float32(105.0),
            shield=jnp.ones(_N_SHIELD, dtype=jnp.bool_),
            cannon_x=jnp.float32(10.0),
            cannon_y=jnp.float32(105.0),
            cannon_active=jnp.bool_(False),
            qotile_y=jnp.float32(105.0),
            swirl_x=jnp.float32(float(_QOTILE_X)),
            swirl_y=jnp.float32(105.0),
            swirl_active=jnp.bool_(False),
            cannon_charged=jnp.bool_(False),
            nibble_count=jnp.int32(0),
            fire_timer=jnp.int32(120),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: YarsRevengeState, action: jax.Array
    ) -> YarsRevengeState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : YarsRevengeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : YarsRevengeState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        dy = jnp.where(
            action == 2, -_PLAYER_SPEED, jnp.where(action == 4, _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(
            state.player_x + dx, jnp.float32(5.0), jnp.float32(float(_SHIELD_X - 4))
        )
        new_py = jnp.clip(state.player_y + dy, jnp.float32(25.0), jnp.float32(185.0))

        # Check if player is in neutral zone (cannot fire, but protected)
        in_neutral = (new_px >= jnp.float32(_NEUTRAL_X_LEFT)) & (
            new_px <= jnp.float32(_NEUTRAL_X_RIGHT)
        )

        # Nibble shield block if player touches it
        at_shield = jnp.abs(new_px - jnp.float32(_SHIELD_X - 4)) < jnp.float32(6.0)
        shield_idx = jnp.argmin(jnp.abs(_SHIELD_YS - new_py))
        nibble = (action == jnp.int32(1)) & at_shield & state.shield[shield_idx]
        new_shield = jnp.where(
            nibble, state.shield.at[shield_idx].set(False), state.shield
        )
        new_nibble = state.nibble_count + jnp.where(nibble, jnp.int32(1), jnp.int32(0))
        step_reward = step_reward + jnp.where(
            nibble, jnp.float32(69.0), jnp.float32(0.0)
        )

        # Cannon charges when enough blocks nibbled
        new_charged = state.cannon_charged | (new_nibble >= jnp.int32(4))

        # Fire cannon (only when charged and not in neutral zone)
        fire_cannon = (
            (action == jnp.int32(1))
            & new_charged
            & ~at_shield
            & ~state.cannon_active
            & ~in_neutral
        )
        new_cx = jnp.where(fire_cannon, new_px, state.cannon_x)
        new_cy = jnp.where(fire_cannon, new_py, state.cannon_y)
        new_cactive = state.cannon_active | fire_cannon
        new_cx = jnp.where(new_cactive, new_cx + _CANNON_SPEED, new_cx)
        new_cactive = new_cactive & (new_cx < jnp.float32(160.0))

        # Cannon hits Qotile
        cannon_hits_qotile = (
            new_cactive
            & (new_cx >= jnp.float32(_QOTILE_X - 6))
            & (jnp.abs(new_cy - state.qotile_y) < jnp.float32(12.0))
        )
        step_reward = step_reward + jnp.where(
            cannon_hits_qotile, jnp.float32(6000.0), jnp.float32(0.0)
        )
        # Reset on kill
        new_cactive = new_cactive & ~cannon_hits_qotile
        new_charged2 = new_charged & ~cannon_hits_qotile
        new_nibble2 = jnp.where(cannon_hits_qotile, jnp.int32(0), new_nibble)
        new_shield2 = jnp.where(
            cannon_hits_qotile, jnp.ones(_N_SHIELD, dtype=jnp.bool_), new_shield
        )

        # Qotile tracks player vertically
        new_qy = state.qotile_y + jnp.clip(new_py - state.qotile_y, -1.0, 1.0)

        # Qotile fires swirl
        new_fire_timer = state.fire_timer - jnp.int32(1)
        can_fire = (new_fire_timer <= jnp.int32(0)) & ~state.swirl_active
        new_sx = jnp.where(can_fire, jnp.float32(_QOTILE_X - 8), state.swirl_x)
        new_sy = jnp.where(can_fire, state.qotile_y, state.swirl_y)
        new_sactive = jnp.where(can_fire, jnp.bool_(True), state.swirl_active)
        new_fire_timer = jnp.where(can_fire, jnp.int32(90), new_fire_timer)

        new_sx = jnp.where(new_sactive, new_sx - _SWIRL_SPEED, new_sx)
        # Swirl oscillates vertically
        new_sy = new_sy + jnp.where(
            new_sactive,
            jnp.sin(new_sx * jnp.float32(0.2)) * jnp.float32(1.5),
            jnp.float32(0.0),
        )
        new_sactive = new_sactive & (new_sx > jnp.float32(5.0))

        # Swirl hits player (not in neutral zone)
        swirl_hits = (
            new_sactive
            & ~in_neutral
            & (jnp.abs(new_sx - new_px) < jnp.float32(8.0))
            & (jnp.abs(new_sy - new_py) < jnp.float32(8.0))
        )

        life_lost = swirl_hits
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return YarsRevengeState(
            player_x=new_px,
            player_y=new_py,
            shield=new_shield2,
            cannon_x=new_cx,
            cannon_y=new_cy,
            cannon_active=new_cactive,
            qotile_y=new_qy,
            swirl_x=new_sx,
            swirl_y=new_sy,
            swirl_active=new_sactive,
            cannon_charged=new_charged2,
            nibble_count=new_nibble2,
            fire_timer=new_fire_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: YarsRevengeState, action: jax.Array) -> YarsRevengeState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : YarsRevengeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : YarsRevengeState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: YarsRevengeState) -> YarsRevengeState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: YarsRevengeState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : YarsRevengeState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Neutral zone
        nz_mask = (_COL_IDX >= _NEUTRAL_X_LEFT) & (_COL_IDX <= _NEUTRAL_X_RIGHT)
        frame = jnp.where(nz_mask[:, :, None], _COLOR_NEUTRAL, frame)

        # Shield blocks
        def draw_shield(frm, i):
            sy = _SHIELD_YS[i].astype(jnp.int32)
            mask = (
                state.shield[i]
                & (_ROW_IDX >= sy - 4)
                & (_ROW_IDX < sy + 4)
                & (_COL_IDX >= _SHIELD_X - 4)
                & (_COL_IDX < _SHIELD_X + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_SHIELD, frm), None

        frame, _ = jax.lax.scan(draw_shield, frame, jnp.arange(_N_SHIELD))

        # Qotile
        qy = state.qotile_y.astype(jnp.int32)
        qm = (
            (_ROW_IDX >= qy - 8)
            & (_ROW_IDX < qy + 8)
            & (_COL_IDX >= _QOTILE_X - 6)
            & (_COL_IDX < _QOTILE_X + 6)
        )
        frame = jnp.where(qm[:, :, None], _COLOR_QOTILE, frame)

        # Swirl
        sm = (
            state.swirl_active
            & (_ROW_IDX >= state.swirl_y.astype(jnp.int32) - 4)
            & (_ROW_IDX < state.swirl_y.astype(jnp.int32) + 4)
            & (_COL_IDX >= state.swirl_x.astype(jnp.int32) - 4)
            & (_COL_IDX < state.swirl_x.astype(jnp.int32) + 4)
        )
        frame = jnp.where(sm[:, :, None], _COLOR_SWIRL, frame)

        # Cannon
        cm = (
            state.cannon_active
            & (_ROW_IDX >= state.cannon_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.cannon_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.cannon_x.astype(jnp.int32))
            & (_COL_IDX < state.cannon_x.astype(jnp.int32) + 8)
        )
        frame = jnp.where(cm[:, :, None], _COLOR_CANNON, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py - 5)
            & (_ROW_IDX < py + 5)
            & (_COL_IDX >= px - 5)
            & (_COL_IDX < px + 5)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Yars' Revenge action indices.
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
