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

"""Tennis — JAX-native game implementation.

One-on-one tennis against a CPU opponent.  Win points by hitting the ball
past the opponent.  First to win 6 games wins the match; standard deuce rules.

Action space (6 actions):
    0 — NOOP
    1 — FIRE (swing racket)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT

Scoring:
    Points follow tennis scoring: 0, 15, 30, 40, game.
    Player score in AtariState tracks number of games won.
    Episode ends when one player wins 6 games; lives: 0.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_COURT_LEFT: int = 10
_COURT_RIGHT: int = 150
_COURT_TOP: int = 30
_COURT_BOTTOM: int = 185
_NET_Y: int = 107
_PLAYER_Y_INIT: int = 160
_CPU_Y_INIT: int = 55
_PLAYER_SPEED: float = 2.0
_BALL_SPEED: float = 3.0
_GRAVITY: float = 0.1

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([50, 150, 50], dtype=jnp.uint8)
_COLOR_COURT = jnp.array([100, 180, 80], dtype=jnp.uint8)
_COLOR_LINES = jnp.array([220, 220, 220], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 200, 100], dtype=jnp.uint8)
_COLOR_CPU = jnp.array([100, 100, 255], dtype=jnp.uint8)
_COLOR_BALL = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class TennisState(AtariState):
    """
    Complete Tennis game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    cpu_x : jax.Array
        float32 — CPU x.
    cpu_y : jax.Array
        float32 — CPU y.
    ball_x : jax.Array
        float32 — Ball x.
    ball_y : jax.Array
        float32 — Ball y.
    ball_vx : jax.Array
        float32 — Ball x velocity.
    ball_vy : jax.Array
        float32 — Ball y velocity.
    ball_z : jax.Array
        float32 — Ball height (bounce simulation).
    ball_vz : jax.Array
        float32 — Ball vertical velocity.
    player_points : jax.Array
        int32 — Player point index (0=0, 1=15, 2=30, 3=40, 4=game).
    cpu_points : jax.Array
        int32 — CPU point index.
    player_games : jax.Array
        int32 — Player games won.
    cpu_games : jax.Array
        int32 — CPU games won.
    serving : jax.Array
        int32 — 0=player, 1=cpu.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_vx: jax.Array
    ball_vy: jax.Array
    ball_z: jax.Array
    ball_vz: jax.Array
    player_points: jax.Array
    cpu_points: jax.Array
    player_games: jax.Array
    cpu_games: jax.Array
    serving: jax.Array
    key: jax.Array


class Tennis(AtariEnv):
    """
    Tennis implemented as a pure JAX function suite.

    Play against a CPU opponent.  First to 6 games wins.  Lives: 0.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=50_000)

    def _reset(self, key: jax.Array) -> TennisState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : TennisState
            Both players at their starting positions, ball in play.
        """
        return TennisState(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(float(_PLAYER_Y_INIT)),
            cpu_x=jnp.float32(80.0),
            cpu_y=jnp.float32(float(_CPU_Y_INIT)),
            ball_x=jnp.float32(80.0),
            ball_y=jnp.float32(float(_NET_Y)),
            ball_vx=jnp.float32(2.0),
            ball_vy=jnp.float32(2.0),
            ball_z=jnp.float32(0.0),
            ball_vz=jnp.float32(1.0),
            player_points=jnp.int32(0),
            cpu_points=jnp.int32(0),
            player_games=jnp.int32(0),
            cpu_games=jnp.int32(0),
            serving=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: TennisState, action: jax.Array) -> TennisState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : TennisState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : TennisState
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
            state.player_x + dx, float(_COURT_LEFT), float(_COURT_RIGHT - 8)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_NET_Y + 5), float(_COURT_BOTTOM - 12)
        )

        # CPU AI: track ball horizontally
        cpu_dx = jnp.clip((state.ball_x - state.cpu_x) * 0.05, -1.5, 1.5)
        new_cx = jnp.clip(
            state.cpu_x + cpu_dx, float(_COURT_LEFT), float(_COURT_RIGHT - 8)
        )
        new_cy = state.cpu_y

        # Ball physics
        new_bx = state.ball_x + state.ball_vx
        new_by = state.ball_y + state.ball_vy
        new_bz = state.ball_z + state.ball_vz
        new_bvz = state.ball_vz - _GRAVITY

        # Bounce on ground (z=0)
        bounced = new_bz < jnp.float32(0.0)
        new_bz = jnp.where(bounced, jnp.float32(0.0), new_bz)
        new_bvz = jnp.where(bounced, -new_bvz * jnp.float32(0.7), new_bvz)

        # Side walls
        at_left = new_bx < float(_COURT_LEFT)
        at_right = new_bx > float(_COURT_RIGHT)
        new_bvx = jnp.where(at_left | at_right, -state.ball_vx, state.ball_vx)
        new_bx = jnp.clip(new_bx, float(_COURT_LEFT), float(_COURT_RIGHT))

        # Player hits ball (swing)
        swing = action == jnp.int32(1)
        player_hits = (
            swing
            & (jnp.abs(new_bx - new_px) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_py) < jnp.float32(10.0))
        )
        new_bvy = jnp.where(player_hits, jnp.float32(-3.0), state.ball_vy)
        new_bvz = jnp.where(player_hits, jnp.float32(1.5), new_bvz)

        # CPU hits ball when in range
        cpu_hits = (
            (jnp.abs(new_bx - new_cx) < jnp.float32(10.0))
            & (jnp.abs(new_by - new_cy) < jnp.float32(10.0))
            & (state.ball_vy < jnp.float32(0.0))
        )
        new_bvy = jnp.where(cpu_hits, jnp.float32(3.0), new_bvy)

        # Ball out of bounds (past baseline)
        ball_out_top = new_by < float(_COURT_TOP)
        ball_out_bot = new_by > float(_COURT_BOTTOM)

        # Point scoring
        player_scores = ball_out_top  # ball went past CPU
        cpu_scores = ball_out_bot  # ball went past player

        new_pp = state.player_points + jnp.where(
            player_scores, jnp.int32(1), jnp.int32(0)
        )
        new_cp = state.cpu_points + jnp.where(cpu_scores, jnp.int32(1), jnp.int32(0))

        # Game won
        player_wins_game = new_pp >= jnp.int32(4)
        cpu_wins_game = new_cp >= jnp.int32(4)

        step_reward = step_reward + jnp.where(
            player_scores, jnp.float32(1.0), jnp.float32(0.0)
        )
        step_reward = step_reward - jnp.where(
            cpu_scores, jnp.float32(1.0), jnp.float32(0.0)
        )

        new_pg = state.player_games + jnp.where(
            player_wins_game, jnp.int32(1), jnp.int32(0)
        )
        new_cg = state.cpu_games + jnp.where(cpu_wins_game, jnp.int32(1), jnp.int32(0))
        new_pp = jnp.where(player_wins_game | cpu_wins_game, jnp.int32(0), new_pp)
        new_cp = jnp.where(player_wins_game | cpu_wins_game, jnp.int32(0), new_cp)

        # Reset ball after point
        point_scored = player_scores | cpu_scores
        new_bx2 = jnp.where(point_scored, jnp.float32(80.0), new_bx)
        new_by2 = jnp.where(point_scored, jnp.float32(float(_NET_Y)), new_by)
        new_bvy2 = jnp.where(
            point_scored,
            jnp.where(player_scores, jnp.float32(-2.0), jnp.float32(2.0)),
            new_bvy,
        )
        new_bvx2 = jnp.where(point_scored, jnp.float32(1.5), new_bvx)

        done = (new_pg >= jnp.int32(6)) | (new_cg >= jnp.int32(6))

        return TennisState(
            player_x=new_px,
            player_y=new_py,
            cpu_x=new_cx,
            cpu_y=new_cy,
            ball_x=new_bx2,
            ball_y=new_by2,
            ball_vx=new_bvx2,
            ball_vy=new_bvy2,
            ball_z=new_bz,
            ball_vz=new_bvz,
            player_points=new_pp,
            cpu_points=new_cp,
            player_games=new_pg,
            cpu_games=new_cg,
            serving=state.serving,
            key=key,
            lives=jnp.int32(0),
            score=state.score + jnp.int32(jnp.maximum(step_reward, jnp.float32(0.0))),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: TennisState, action: jax.Array) -> TennisState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : TennisState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : TennisState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: TennisState) -> TennisState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: TennisState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : TennisState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_COURT, dtype=jnp.uint8)

        # Net
        net_mask = (
            (_ROW_IDX == _NET_Y)
            & (_COL_IDX >= _COURT_LEFT)
            & (_COL_IDX <= _COURT_RIGHT)
        )
        frame = jnp.where(net_mask[:, :, None], _COLOR_LINES, frame)

        # CPU
        cx = state.cpu_x.astype(jnp.int32)
        cy = state.cpu_y.astype(jnp.int32)
        cm = (
            (_ROW_IDX >= cy)
            & (_ROW_IDX < cy + 10)
            & (_COL_IDX >= cx)
            & (_COL_IDX < cx + 8)
        )
        frame = jnp.where(cm[:, :, None], _COLOR_CPU, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py)
            & (_ROW_IDX < py + 10)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        # Ball (size based on height z)
        bx = state.ball_x.astype(jnp.int32)
        by = state.ball_y.astype(jnp.int32)
        bm = (
            (_ROW_IDX >= by - 3)
            & (_ROW_IDX < by + 3)
            & (_COL_IDX >= bx - 3)
            & (_COL_IDX < bx + 3)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BALL, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Tennis action indices.
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
