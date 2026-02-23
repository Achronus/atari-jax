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

ROM-backed: requires ale-py to load game ROMs.

Run with:
    pytest tests/make/test_make.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env import (
    AtariEnv,
    Env,
    EnvSpec,
    EpisodeStatisticsState,
    GrayscaleObservation,
    make,
)
from atarax.env.atari_env import AtariEnv

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"


def test_make_returns_env():
    env = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, Env)


def test_make_raw_returns_atari_env():
    env = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, AtariEnv)


def test_make_returns_single_mode():
    env = make(_BREAKOUT, jit_compile=False)
    assert isinstance(env, AtariEnv)
    assert env._compile_mode == "single"


def test_make_env_spec_accepted():
    env = make(EnvSpec("atari", "breakout"), jit_compile=False)
    assert isinstance(env, AtariEnv)


def test_make_invalid_id_raises():
    with pytest.raises(ValueError, match="Invalid environment ID"):
        make("breakout")


def test_make_wrappers_list():
    env = make(_BREAKOUT, wrappers=[GrayscaleObservation], jit_compile=False)
    assert isinstance(env, GrayscaleObservation)
    assert isinstance(env, Env)


def test_make_preset_dqn_obs_shape():
    env = make(_BREAKOUT, preset=True, jit_compile=False)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodeStatisticsState)


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


def test_make_jit_compile_true_reset():
    env = make(_BREAKOUT, preset=True, jit_compile=True)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodeStatisticsState)


def test_make_jit_compile_true_step():
    env = make(_BREAKOUT, preset=True, jit_compile=True)
    _, state = env.reset(_key)
    obs, _, reward, done, _ = env.step(state, env.sample(_key))
    assert obs.shape == (84, 84, 4)
    assert reward.shape == ()
    assert done.shape == ()


def test_make_jit_compile_false_reset():
    env = make(_BREAKOUT, preset=True, jit_compile=False)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodeStatisticsState)
