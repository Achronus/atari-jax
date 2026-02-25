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

from atarax.env import Env, make
from atarax.env.atari_env import AtariEnv
from atarax.env.wrappers import EpisodeStatisticsState, GrayscaleObservation

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"


def test_make_returns_env():
    env = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, Env)


def test_make_raw_returns_atari_env():
    env = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, AtariEnv)


def test_make_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid environment ID"):
        make("breakout")


def test_make_unknown_game_raises():
    with pytest.raises(ValueError, match="Unknown game"):
        make("atari/not_a_game-v0")


def test_make_wrappers_list():
    env = make(_BREAKOUT, wrappers=[GrayscaleObservation], jit_compile=False)
    assert isinstance(env, GrayscaleObservation)
    assert isinstance(env, Env)


def test_make_preset_dqn_obs_shape():
    env = make(_BREAKOUT, preset=True, jit_compile=False)
    obs, _ = env.reset(_key)
    assert obs.shape == (84, 84, 4)


def test_make_preset_and_wrappers_raises():
    with pytest.raises(ValueError, match="not both"):
        make(_BREAKOUT, wrappers=[GrayscaleObservation], preset=True)


def test_make_reset_shape():
    env = make(_BREAKOUT, jit_compile=False)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (210, 160, 3))


def test_make_step_shape():
    env = make(_BREAKOUT, jit_compile=False)
    _, state = env.reset(_key)
    obs, _, reward, done, _ = env.step(state, jnp.int32(0))
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_make_step_info_keys():
    env = make(_BREAKOUT, jit_compile=False)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, jnp.int32(0))
    assert "lives" in info
    assert "score" in info
    assert "episode_step" in info


def test_make_case_insensitive():
    env1 = make("atari/breakout-v0", jit_compile=False)
    env2 = make("atari/Breakout-v0", jit_compile=False)
    assert type(env1) is type(env2)


def test_make_preset_episode_statistics_state():
    env = make(_BREAKOUT, preset=True, jit_compile=False)
    _, state = env.reset(_key)
    assert isinstance(state, EpisodeStatisticsState)
