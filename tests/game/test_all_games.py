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

"""Parametrized smoke tests covering all 57 supported game ROMs.

Verifies that every game class satisfies the `AtariGame` contract:
  - `get_lives`    returns a rank-0 `int32` array
  - `get_reward`   returns a rank-0 `float32` array
  - `is_terminal`  returns a rank-0 `bool` array

Also verifies the base class structure: `get_lives`, `get_reward`, and
`is_terminal` are abstract; `reset` and `step` are concrete.

Run with:
    pytest tests/game/test_all_games.py -v
"""

import chex
import jax.numpy as jnp
import pytest

from atari_jax.games.base import AtariGame
from atari_jax.games.registry import _GAMES

_GAME_PARAMS = [(spec.ale_name, spec.game) for spec in _GAMES]
_GAME_IDS = [spec.ale_name for spec in _GAMES]


def _zeros_ram() -> chex.Array:
    return jnp.zeros(128, dtype=jnp.uint8)


def test_abstract_methods():
    """get_lives, get_reward, is_terminal must be abstract."""
    assert getattr(AtariGame.get_lives, "__isabstractmethod__", False)
    assert getattr(AtariGame.get_reward, "__isabstractmethod__", False)
    assert getattr(AtariGame.is_terminal, "__isabstractmethod__", False)


def test_concrete_methods():
    """reset and step must be concrete (not abstract)."""
    assert not getattr(AtariGame.reset, "__isabstractmethod__", False)
    assert not getattr(AtariGame.step, "__isabstractmethod__", False)


def test_warmup_frames_default():
    """`_WARMUP_FRAMES` must be a positive int on the base class."""
    assert isinstance(AtariGame._WARMUP_FRAMES, int)
    assert AtariGame._WARMUP_FRAMES > 0


@pytest.mark.parametrize("ale_name,game", _GAME_PARAMS, ids=_GAME_IDS)
def test_get_lives_dtype(ale_name: str, game: AtariGame):
    """get_lives must return a rank-0 int32 scalar."""
    lives = game.get_lives(_zeros_ram())
    chex.assert_rank(lives, 0)
    chex.assert_type(lives, jnp.int32)


@pytest.mark.parametrize("ale_name,game", _GAME_PARAMS, ids=_GAME_IDS)
def test_get_reward_dtype(ale_name: str, game: AtariGame):
    """get_reward must return a rank-0 float32 scalar."""
    ram = _zeros_ram()
    reward = game.get_reward(ram, ram)
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)


@pytest.mark.parametrize("ale_name,game", _GAME_PARAMS, ids=_GAME_IDS)
def test_is_terminal_dtype(ale_name: str, game: AtariGame):
    """is_terminal must return a rank-0 bool scalar."""
    terminal = game.is_terminal(_zeros_ram(), jnp.int32(0))
    chex.assert_rank(terminal, 0)
    chex.assert_type(terminal, bool)
