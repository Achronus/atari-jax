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

import chex


@chex.dataclass
class GameState:
    """
    Base dataclass for all JAX game state structs.

    Every game's concrete state class must inherit from `GameState` and declare
    these four fields (plus any game-specific fields).

    Parameters
    ----------
    reward : chex.Array
        float32 — Reward earned on the last `step` call.
    done : chex.Array
        bool — `True` when the episode has ended (game over or truncated).
    step : chex.Array
        int32 — Total emulated steps since power-on.
    episode_step : chex.Array
        int32 — Agent steps elapsed in the current episode.
    """

    reward: chex.Array
    done: chex.Array
    step: chex.Array
    episode_step: chex.Array


@chex.dataclass
class AtariState(GameState):
    """
    Atari-specific extension of `GameState`.

    Adds fields common to all Atari 2600 games. Game-specific state
    dataclasses (e.g. `BreakoutState`) inherit from `AtariState` and add
    their own fields.

    Parameters
    ----------
    lives : chex.Array
        int32 — Remaining lives (use `0` for games without lives).
    score : chex.Array
        int32 — Cumulative episode score.
    level : chex.Array
        int32 — Board / wave / level counter; `0` = first board.
    key : chex.PRNGKey
        JAX PRNG key, evolved each physics sub-step.
    """

    lives: chex.Array
    score: chex.Array
    level: chex.Array
    key: chex.PRNGKey
