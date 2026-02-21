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

"""Unit tests for atarax.games.roms.breakout — reward and terminal logic.

Run with:
    pytest tests/game/test_breakout.py -v
"""

import chex
import jax.numpy as jnp

from atarax.games.roms.breakout import LIVES_ADDR, SCORE_X, SCORE_Y, Breakout

game = Breakout()


def _ram(overrides=None):
    """Return a zeroed 128-byte RAM array with optional ``{index: value}`` overrides."""
    data = jnp.zeros(128, dtype=jnp.uint8)
    if overrides:
        for idx, val in overrides.items():
            data = data.at[idx].set(jnp.uint8(val))
    return data


def test_get_reward_zero():
    ram = _ram()
    reward = game.get_reward(ram, ram)
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)
    assert float(reward) == 0.0


def test_get_reward_ones():
    # SCORE_X ones nibble: 0x01 → 1 point
    ram_prev = _ram()
    ram_curr = _ram({SCORE_X: 0x01})
    reward = game.get_reward(ram_prev, ram_curr)
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)
    assert float(reward) == 1.0


def test_get_reward_ones_tens():
    # SCORE_X = 0x10 → tens=1, ones=0 → 10 points
    ram_prev = _ram()
    ram_curr = _ram({SCORE_X: 0x10})
    assert float(game.get_reward(ram_prev, ram_curr)) == 10.0


def test_get_reward_hundreds():
    # SCORE_Y = 0x01 → hundreds=1 → 100 points
    ram_prev = _ram()
    ram_curr = _ram({SCORE_Y: 0x01})
    assert float(game.get_reward(ram_prev, ram_curr)) == 100.0


def test_get_reward_combined():
    # SCORE_Y=0x01, SCORE_X=0x23 → 100 + 20 + 3 = 123 points
    ram_prev = _ram()
    ram_curr = _ram({SCORE_Y: 0x01, SCORE_X: 0x23})
    assert float(game.get_reward(ram_prev, ram_curr)) == 123.0


def test_is_terminal_false_before_start():
    # lives_prev = 0 → game never started → not terminal even if ram lives = 0
    ram = _ram({LIVES_ADDR: 0})
    terminal = game.is_terminal(ram, jnp.int32(0))
    chex.assert_rank(terminal, 0)
    chex.assert_type(terminal, bool)
    assert not bool(terminal)


def test_is_terminal_true():
    # lives_prev = 5, current lives = 0 → terminal
    ram = _ram({LIVES_ADDR: 0})
    assert bool(game.is_terminal(ram, jnp.int32(5)))


def test_is_terminal_false_mid_game():
    # lives_prev = 3, current lives = 2 → still playing
    ram = _ram({LIVES_ADDR: 2})
    assert not bool(game.is_terminal(ram, jnp.int32(3)))


def test_is_terminal_false_when_lives_equal():
    # lives_prev = 5, current lives = 5 → no change, not terminal
    ram = _ram({LIVES_ADDR: 5})
    assert not bool(game.is_terminal(ram, jnp.int32(5)))
