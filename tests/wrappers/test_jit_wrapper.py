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

"""Tests for JitWrapper.

Run with:
    pytest tests/wrappers/test_jit_wrapper.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax import make
from atarax.env.games.breakout import BreakoutParams
from atarax.game import AtaraxGame, AtaraxParams
from atarax.wrappers import JitWrapper, Wrapper

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_params = BreakoutParams()


def _base_env() -> AtaraxGame:
    """Return an un-compiled AtaraxGame for use as the JitWrapper target."""
    env, _ = make(_BREAKOUT, jit_compile=False)
    return env


def test_jit_wrapper_is_wrapper():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert isinstance(env, Wrapper)


def test_jit_wrapper_has_compiled_methods():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert hasattr(env, "_jit_reset")
    assert hasattr(env, "_jit_step")


def test_jit_wrapper_reset_shape():
    env = JitWrapper(_base_env(), cache_dir=None)
    obs, _ = env.reset(_key, _params)
    chex.assert_shape(obs, (210, 160, 3))
    assert obs.dtype == jnp.uint8


def test_jit_wrapper_step_shape():
    env = JitWrapper(_base_env(), cache_dir=None)
    _, state = env.reset(_key, _params)
    obs, _, reward, done, _ = env.step(_key, state, jnp.int32(0), _params)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_jit_wrapper_repr():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert "JitWrapper" in repr(env)
    assert "breakout" in repr(env).lower()


def test_jit_wrapper_spaces_delegate():
    inner = _base_env()
    env = JitWrapper(inner, cache_dir=None)
    assert env.observation_space == inner.observation_space
    assert env.action_space == inner.action_space


def test_jit_wrapper_unwrapped():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert isinstance(env.unwrapped, AtaraxGame)


def test_make_jit_compile_returns_jit_wrapper():
    env, _ = make(_BREAKOUT, jit_compile=True)
    assert isinstance(env, JitWrapper)
