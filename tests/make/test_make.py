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

"""Tests for the make() factory function.

Run with:
    pytest tests/make/test_make.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax import make
from atarax.env.registry import GAMES
from atarax.game import AtaraxGame
from atarax.wrappers import EpisodeStatisticsState, GrayscaleObservation

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_returns_tuple():
    env, _ = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, AtaraxGame)


def test_make_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid environment ID"):
        make("breakout")


def test_make_unknown_game_raises():
    with pytest.raises(ValueError, match="Unknown game"):
        make("atari/not_a_game-v0")


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_wrappers_list():
    env, _ = make(_BREAKOUT, wrappers=[GrayscaleObservation], jit_compile=False)
    assert isinstance(env, GrayscaleObservation)


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_preset_dqn_obs_shape():
    env, params = make(_BREAKOUT, preset=True, jit_compile=False)
    obs, _ = env.reset(_key, params)
    assert obs.shape == (84, 84, 4)


def test_make_preset_and_wrappers_raises():
    with pytest.raises(ValueError, match="not both"):
        make(_BREAKOUT, wrappers=[GrayscaleObservation], preset=True)


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_reset_shape():
    env, params = make(_BREAKOUT, jit_compile=False)
    obs, _ = env.reset(_key, params)
    chex.assert_shape(obs, (210, 160, 3))


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_step_shape():
    env, params = make(_BREAKOUT, jit_compile=False)
    _, state = env.reset(_key, params)
    obs, _, reward, done, _ = env.step(_key, state, jnp.int32(0), params)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_step_info_keys():
    env, params = make(_BREAKOUT, jit_compile=False)
    _, state = env.reset(_key, params)
    _, _, _, _, info = env.step(_key, state, jnp.int32(0), params)
    assert "lives" in info
    assert "score" in info
    assert "episode_step" in info


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_case_insensitive():
    env1, _ = make("atari/breakout-v0", jit_compile=False)
    env2, _ = make("atari/Breakout-v0", jit_compile=False)
    assert type(env1) is type(env2)


@pytest.mark.skipif(not GAMES, reason="No games registered yet")
def test_make_preset_episode_statistics_state():
    env, params = make(_BREAKOUT, preset=True, jit_compile=False)
    _, state = env.reset(_key, params)
    assert isinstance(state, EpisodeStatisticsState)
