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

"""Tennis — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Tennis.cpp
# Reward: tracks changes in both game differential and point differential.
GAME_SCORE_P1 = 0xC5  # player 1 games won
GAME_SCORE_P2 = 0xC6  # player 2 games won
POINT_SCORE_P1 = 0xC7  # player 1 points in current game
POINT_SCORE_P2 = 0xC8  # player 2 points in current game


class Tennis(AtariGame):
    """Tennis game logic: reward and terminal extraction, reset, and step.

    Reward reflects changes in point differential first, then game differential
    (matching ALE behaviour).  Terminal when either player wins the set: player
    must have ≥6 games with a 2+ game lead, or 7 games (tiebreak).  No lives
    counter.
    """

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Tennis has no lives counter; always returns 0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Always 0 (no lives counter in this game).
        """
        return jnp.int32(0)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step.

        Reward is the change in point differential if points changed, otherwise
        the change in game differential, matching ALE behaviour.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Reward signal for this step.
        """
        dp_prev = ram_prev[POINT_SCORE_P1].astype(jnp.int32) - ram_prev[
            POINT_SCORE_P2
        ].astype(jnp.int32)
        dp_curr = ram_curr[POINT_SCORE_P1].astype(jnp.int32) - ram_curr[
            POINT_SCORE_P2
        ].astype(jnp.int32)
        ds_prev = ram_prev[GAME_SCORE_P1].astype(jnp.int32) - ram_prev[
            GAME_SCORE_P2
        ].astype(jnp.int32)
        ds_curr = ram_curr[GAME_SCORE_P1].astype(jnp.int32) - ram_curr[
            GAME_SCORE_P2
        ].astype(jnp.int32)

        points_changed = dp_curr != dp_prev
        score_changed = ds_curr != ds_prev

        reward = jnp.where(
            points_changed,
            dp_curr - dp_prev,
            jnp.where(score_changed, ds_curr - ds_prev, jnp.int32(0)),
        )
        return reward.astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when either player wins the set (≥6 games with 2+ lead, or 7 games).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; score-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        my_pts = ram[POINT_SCORE_P1].astype(jnp.int32)
        opp_pts = ram[POINT_SCORE_P2].astype(jnp.int32)
        dp = my_pts - opp_pts

        p1_wins = (my_pts >= jnp.int32(6)) & (dp >= jnp.int32(2))
        p2_wins = (opp_pts >= jnp.int32(6)) & (-dp >= jnp.int32(2))
        tiebreak = (my_pts == jnp.int32(7)) | (opp_pts == jnp.int32(7))

        return p1_wins | p2_wins | tiebreak
