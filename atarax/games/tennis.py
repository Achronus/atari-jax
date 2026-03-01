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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Court      : x ∈ [10, 150],  y ∈ [30, 185]
    Net        : y = 107
    Player     : 6×12 px,  starts x=77 y=160  (bottom half)
    CPU        : 6×12 px,  starts x=77 y=55   (top half)
    Ball       : 3×3 px
    Player half: y ∈ [107, 185]
    CPU half   : y ∈ [30, 107]

Action space (18 actions — ALE minimal set):
    0  NOOP
    1  FIRE  (swing racket)
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

# Court geometry
_COURT_LEFT: int = 10
_COURT_RIGHT: int = 150
_COURT_TOP: int = 30
_COURT_BOTTOM: int = 185
_NET_Y: int = 107

# Player (bottom half)
_PLAYER_START_X: float = 77.0
_PLAYER_START_Y: float = 160.0
_PLAYER_W: int = 6
_PLAYER_H: int = 12
_PLAYER_SPEED: float = 2.0
_PLAYER_YMIN: float = float(_NET_Y)
_PLAYER_YMAX: float = float(_COURT_BOTTOM - _PLAYER_H)

# CPU (top half)
_CPU_START_X: float = 77.0
_CPU_START_Y: float = 55.0
_CPU_SPEED: float = 1.5
_CPU_YMIN: float = float(_COURT_TOP)
_CPU_YMAX: float = float(_NET_Y - _PLAYER_H)

# Ball
_BALL_W: int = 3
_BALL_H: int = 3
_BALL_SPEED: float = 3.0
_BALL_SERVE_X: float = 80.0
_BALL_SERVE_Y_PLAYER: float = 150.0   # served toward CPU (up)
_BALL_SERVE_Y_CPU: float = 65.0       # served by CPU (down)

# Racket hit radius (half-widths of player + ball)
_HIT_MARGIN: float = 8.0

# Tennis scoring: pts 0–4 (0=love, 1=15, 2=30, 3=40, 4=advantage)
_WIN_GAMES: int = 6

_FRAME_SKIP: int = 4

# Colours
_COURT_COLOR = jnp.array([144, 144, 144], dtype=jnp.uint8)    # gray court
_NET_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)      # white net
_PLAYER_COLOR = jnp.array([92, 186, 92], dtype=jnp.uint8)     # green player
_CPU_COLOR = jnp.array([213, 130, 74], dtype=jnp.uint8)       # orange CPU
_BALL_COLOR = jnp.array([252, 252, 84], dtype=jnp.uint8)      # yellow ball
_SCORE_COLOR = jnp.array([236, 236, 236], dtype=jnp.uint8)

# 3×5 bitmap font
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
class TennisState(AtariState):
    """
    Complete Tennis game state — a JAX pytree.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `score` tracks games won by the player.

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
    ball_x : jax.Array
        float32 — Ball left-edge x coordinate.
    ball_y : jax.Array
        float32 — Ball top-edge y coordinate.
    ball_dx : jax.Array
        float32 — Ball x velocity (px/frame; positive = right).
    ball_dy : jax.Array
        float32 — Ball y velocity (px/frame; negative = up).
    ball_active : jax.Array
        bool — True once the ball is in play.
    player_pts : jax.Array
        int32 — Player's current point level (0–4: love/15/30/40/advantage).
    cpu_pts : jax.Array
        int32 — CPU's current point level (0–4).
    cpu_games : jax.Array
        int32 — Games won by the CPU.
    """

    player_x: jax.Array
    player_y: jax.Array
    cpu_x: jax.Array
    cpu_y: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    ball_dx: jax.Array
    ball_dy: jax.Array
    ball_active: jax.Array
    player_pts: jax.Array
    cpu_pts: jax.Array
    cpu_games: jax.Array


def _decode_action(action: jax.Array):
    """Decode an 18-action ALE index into (move_up, move_right, move_down, move_left, has_fire)."""
    has_fire = (
        (action == 1) | (action == 10) | (action == 11) | (action == 12) | (action == 13)
        | (action == 14) | (action == 15) | (action == 16) | (action == 17)
    )
    move_up = (action == 2) | (action == 6) | (action == 7) | (action == 10) | (action == 14) | (action == 15)
    move_right = (action == 3) | (action == 6) | (action == 8) | (action == 11) | (action == 14) | (action == 16)
    move_left = (action == 4) | (action == 7) | (action == 9) | (action == 12) | (action == 15) | (action == 17)
    move_down = (action == 5) | (action == 8) | (action == 9) | (action == 13) | (action == 16) | (action == 17)
    return move_up, move_right, move_down, move_left, has_fire


def _serve_ball(key: chex.PRNGKey, player_serves: jax.Array):
    """Return (ball_x, ball_y, ball_dx, ball_dy, new_key) for a fresh serve."""
    key, subkey = jax.random.split(key)
    angle = jax.random.uniform(subkey, minval=-0.4, maxval=0.4)
    ball_x = jnp.float32(_BALL_SERVE_X)
    # Player serves upward (toward CPU), CPU serves downward
    ball_y = jnp.where(player_serves, jnp.float32(_BALL_SERVE_Y_PLAYER), jnp.float32(_BALL_SERVE_Y_CPU))
    ball_dy = jnp.where(player_serves, jnp.float32(-_BALL_SPEED), jnp.float32(_BALL_SPEED))
    ball_dx = jnp.float32(_BALL_SPEED) * angle
    return ball_x, ball_y, ball_dx, ball_dy, key


def _advance_pts(scorer_pts: jax.Array, other_pts: jax.Array):
    """
    Advance scoring (tennis deuce-aware) when the scorer wins a point.

    Returns (new_scorer_pts, new_other_pts, game_won).
    """
    # At 40-40 (pts==3 both), entering "deuce" zone: pts escalates to 4=advantage
    at_deuce = (scorer_pts == 3) & (other_pts == 3)
    at_advantage = scorer_pts == 4        # scorer has advantage
    other_advantage = other_pts == 4      # opponent has advantage

    # Win the game: already at advantage (4) OR at 40 (3) and opponent < 40
    game_won = at_advantage | ((scorer_pts == 3) & (other_pts < 3))

    # Deuce: both at 40 → give advantage (pts 4)
    new_pts = jnp.where(
        game_won,
        jnp.int32(0),
        jnp.where(at_deuce, jnp.int32(4), scorer_pts + jnp.int32(1)),
    )
    # If other had advantage (4), scoring returns both to deuce (3)
    new_other = jnp.where(
        game_won,
        jnp.int32(0),
        jnp.where(other_advantage, jnp.int32(3), other_pts),
    )
    return new_pts, new_other, game_won


class Tennis(AtaraxGame):
    """
    Tennis implemented as a pure-JAX function suite.

    Physics: player and CPU move freely in their respective court halves.
    The ball bounces off walls and the net. A swing (FIRE action) near the ball
    returns it. Standard tennis scoring with deuce/advantage. First to 6 games wins.
    """

    num_actions: int = 18

    def _reset(self, key: chex.PRNGKey) -> TennisState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : TennisState
            Players at start positions, ball inactive.
        """
        ball_x, ball_y, ball_dx, ball_dy, key = _serve_ball(key, player_serves=jnp.bool_(True))
        return TennisState(
            player_x=jnp.float32(_PLAYER_START_X),
            player_y=jnp.float32(_PLAYER_START_Y),
            cpu_x=jnp.float32(_CPU_START_X),
            cpu_y=jnp.float32(_CPU_START_Y),
            ball_x=ball_x,
            ball_y=ball_y,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            ball_active=jnp.bool_(False),
            player_pts=jnp.int32(0),
            cpu_pts=jnp.int32(0),
            cpu_games=jnp.int32(0),
            lives=jnp.int32(0),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: TennisState, action: jax.Array) -> TennisState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : TennisState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).

        Returns
        -------
        new_state : TennisState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        move_up, move_right, move_down, move_left, has_fire = _decode_action(action)

        # --- Player movement (clamped to bottom half) ---
        dx = jnp.where(move_right, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0)) - jnp.where(move_left, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0))
        dy = jnp.where(move_down, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0)) - jnp.where(move_up, jnp.float32(_PLAYER_SPEED), jnp.float32(0.0))
        player_x = jnp.clip(state.player_x + dx, jnp.float32(_COURT_LEFT), jnp.float32(_COURT_RIGHT - _PLAYER_W))
        player_y = jnp.clip(state.player_y + dy, jnp.float32(_PLAYER_YMIN), jnp.float32(_PLAYER_YMAX))

        # --- CPU AI (tracks ball, stays in top half) ---
        cpu_dx = jnp.clip(state.ball_x - state.cpu_x, -_CPU_SPEED, _CPU_SPEED)
        cpu_dy = jnp.clip(state.ball_y - state.cpu_y, -_CPU_SPEED, _CPU_SPEED)
        cpu_x = jnp.clip(state.cpu_x + cpu_dx, jnp.float32(_COURT_LEFT), jnp.float32(_COURT_RIGHT - _PLAYER_W))
        cpu_y = jnp.clip(state.cpu_y + cpu_dy, jnp.float32(_CPU_YMIN), jnp.float32(_CPU_YMAX))

        # --- Ball activation: player serves on FIRE when not active ---
        fire_serve = has_fire & ~state.ball_active
        sx, sy, sdx, sdy, key = _serve_ball(key, player_serves=jnp.bool_(True))
        ball_x = jnp.where(fire_serve, sx, state.ball_x)
        ball_y = jnp.where(fire_serve, sy, state.ball_y)
        ball_dx = jnp.where(fire_serve, sdx, state.ball_dx)
        ball_dy = jnp.where(fire_serve, sdy, state.ball_dy)
        ball_active = state.ball_active | fire_serve

        # --- Ball movement ---
        new_bx = jnp.where(ball_active, ball_x + ball_dx, ball_x)
        new_by = jnp.where(ball_active, ball_y + ball_dy, ball_y)

        # Left/right wall bounce
        hit_left = new_bx < jnp.float32(_COURT_LEFT)
        new_bx = jnp.where(hit_left, 2.0 * _COURT_LEFT - new_bx, new_bx)
        ball_dx = jnp.where(hit_left, -ball_dx, ball_dx)

        hit_right = new_bx + _BALL_W > jnp.float32(_COURT_RIGHT)
        new_bx = jnp.where(hit_right, 2.0 * (_COURT_RIGHT - _BALL_W) - new_bx, new_bx)
        ball_dx = jnp.where(hit_right, -ball_dx, ball_dx)

        # Net bounce (dy flip when ball crosses net)
        net_hit = (
            ball_active
            & (jnp.sign(ball_dy) * (new_by - _NET_Y) > 0.0)  # crossed net
            & (jnp.sign(ball_dy) * (ball_y - _NET_Y) <= 0.0)  # was on other side
        )
        new_by = jnp.where(net_hit, 2.0 * _NET_Y - new_by, new_by)
        ball_dy = jnp.where(net_hit, -ball_dy, ball_dy)

        # --- Player racket contact ---
        player_cx = player_x + jnp.float32(_PLAYER_W) / 2.0
        player_cy = player_y + jnp.float32(_PLAYER_H) / 2.0
        ball_cx = new_bx + jnp.float32(_BALL_W) / 2.0
        ball_cy = new_by + jnp.float32(_BALL_H) / 2.0

        player_near = (
            (jnp.abs(ball_cx - player_cx) < _HIT_MARGIN)
            & (jnp.abs(ball_cy - player_cy) < _HIT_MARGIN)
        )
        player_hit = ball_active & has_fire & player_near & (ball_dy > 0.0)
        angle_p = (ball_cx - player_cx) / _HIT_MARGIN * 0.5  # small x-angle
        ball_dx = jnp.where(player_hit, ball_dx + angle_p * _BALL_SPEED, ball_dx)
        ball_dy = jnp.where(player_hit, -jnp.abs(ball_dy), ball_dy)

        # --- CPU auto-return ---
        cpu_cx = cpu_x + jnp.float32(_PLAYER_W) / 2.0
        cpu_cy = cpu_y + jnp.float32(_PLAYER_H) / 2.0
        cpu_near = (
            (jnp.abs(ball_cx - cpu_cx) < _HIT_MARGIN)
            & (jnp.abs(ball_cy - cpu_cy) < _HIT_MARGIN)
        )
        cpu_hit = ball_active & cpu_near & (ball_dy < 0.0)
        angle_c = (ball_cx - cpu_cx) / _HIT_MARGIN * 0.5
        ball_dx = jnp.where(cpu_hit, ball_dx + angle_c * _BALL_SPEED, ball_dx)
        ball_dy = jnp.where(cpu_hit, jnp.abs(ball_dy), ball_dy)

        # --- Scoring: ball exits court ---
        player_scores = ball_active & (new_by < jnp.float32(_COURT_TOP))   # ball past CPU baseline
        cpu_scores = ball_active & (new_by > jnp.float32(_COURT_BOTTOM))   # ball past player baseline

        step_reward = jnp.where(player_scores, jnp.float32(1.0), jnp.where(cpu_scores, jnp.float32(-1.0), jnp.float32(0.0)))

        # Advance points
        new_player_pts, new_cpu_pts_after_pscore, player_game_won = _advance_pts(state.player_pts, state.cpu_pts)
        new_cpu_pts, new_player_pts_after_cscore, cpu_game_won = _advance_pts(state.cpu_pts, state.player_pts)

        # Apply scoring: player_scores takes priority over cpu_scores (both can't happen same frame)
        final_player_pts = jnp.where(player_scores, new_player_pts, jnp.where(cpu_scores, new_player_pts_after_cscore, state.player_pts))
        final_cpu_pts = jnp.where(player_scores, new_cpu_pts_after_pscore, jnp.where(cpu_scores, new_cpu_pts, state.cpu_pts))

        game_won_by_player = player_scores & player_game_won
        game_won_by_cpu = cpu_scores & cpu_game_won

        new_player_games = state.score + jnp.where(game_won_by_player, jnp.int32(1), jnp.int32(0))
        new_cpu_games = state.cpu_games + jnp.where(game_won_by_cpu, jnp.int32(1), jnp.int32(0))

        # Reset ball after point
        any_scored = player_scores | cpu_scores
        sx2, sy2, sdx2, sdy2, key = _serve_ball(key, player_serves=cpu_scores)  # CPU just scored → player serves next... but keep it simple: always player serves
        ball_active = jnp.where(any_scored, jnp.bool_(False), ball_active)
        new_bx = jnp.where(any_scored, sx2, new_bx)
        new_by = jnp.where(any_scored, sy2, new_by)
        ball_dx = jnp.where(any_scored, sdx2, ball_dx)
        ball_dy = jnp.where(any_scored, sdy2, ball_dy)

        # Episode done: either player reaches _WIN_GAMES
        done = (new_player_games >= jnp.int32(_WIN_GAMES)) | (new_cpu_games >= jnp.int32(_WIN_GAMES))

        return TennisState(
            player_x=player_x,
            player_y=player_y,
            cpu_x=cpu_x,
            cpu_y=cpu_y,
            ball_x=new_bx,
            ball_y=new_by,
            ball_dx=ball_dx,
            ball_dy=ball_dy,
            ball_active=ball_active,
            player_pts=final_player_pts,
            cpu_pts=final_cpu_pts,
            cpu_games=new_cpu_games,
            lives=jnp.int32(0),
            score=new_player_games,
            level=state.level,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step,
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: TennisState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> TennisState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : TennisState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).
        params : AtaraxParams
            Static environment parameters.

        Returns
        -------
        new_state : TennisState
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
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # Court surface
        court_mask = (
            (_ROW_IDX >= _COURT_TOP) & (_ROW_IDX < _COURT_BOTTOM)
            & (_COL_IDX >= _COURT_LEFT) & (_COL_IDX < _COURT_RIGHT)
        )
        frame = jnp.where(court_mask[:, :, None], _COURT_COLOR, frame)

        # Net
        net_mask = (
            (_ROW_IDX == _NET_Y)
            & (_COL_IDX >= _COURT_LEFT) & (_COL_IDX < _COURT_RIGHT)
        )
        frame = jnp.where(net_mask[:, :, None], _NET_COLOR, frame)

        # CPU player
        cpu_mask = (
            (_ROW_IDX >= jnp.int32(state.cpu_y)) & (_ROW_IDX < jnp.int32(state.cpu_y) + _PLAYER_H)
            & (_COL_IDX >= jnp.int32(state.cpu_x)) & (_COL_IDX < jnp.int32(state.cpu_x) + _PLAYER_W)
        )
        frame = jnp.where(cpu_mask[:, :, None], _CPU_COLOR, frame)

        # Player
        player_mask = (
            (_ROW_IDX >= jnp.int32(state.player_y)) & (_ROW_IDX < jnp.int32(state.player_y) + _PLAYER_H)
            & (_COL_IDX >= jnp.int32(state.player_x)) & (_COL_IDX < jnp.int32(state.player_x) + _PLAYER_W)
        )
        frame = jnp.where(player_mask[:, :, None], _PLAYER_COLOR, frame)

        # Ball (only when active)
        bx = jnp.int32(state.ball_x)
        by = jnp.int32(state.ball_y)
        ball_mask = (
            (_ROW_IDX >= by) & (_ROW_IDX < by + _BALL_H)
            & (_COL_IDX >= bx) & (_COL_IDX < bx + _BALL_W)
            & state.ball_active
        )
        frame = jnp.where(ball_mask[:, :, None], _BALL_COLOR, frame)

        # Score: player games (left) | cpu games (right)
        frame = _blit_digit(frame, state.score % 10, x0=8, y0=4)
        frame = _blit_digit(frame, state.cpu_games % 10, x0=20, y0=4)

        return frame

    def _key_map(self) -> dict:
        """Return the key-to-action mapping for interactive play."""
        import pygame

        return {
            pygame.K_SPACE: 1,   # FIRE — swing
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 4,
            pygame.K_a: 4,
            pygame.K_DOWN: 5,
            pygame.K_s: 5,
        }
