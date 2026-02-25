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

from atarax.env import Env, make
from atarax.env.atari_env import AtariEnv
from atarax.env.wrappers import JitWrapper, Wrapper

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_N_STEPS = 4


def _base_env() -> Env:
    """Return an un-compiled AtariEnv for use as the JitWrapper target."""
    return make(_BREAKOUT, jit_compile=False)


def test_jit_wrapper_is_env_and_wrapper():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert isinstance(env, Env)
    assert isinstance(env, Wrapper)


def test_jit_wrapper_has_compiled_methods():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert hasattr(env, "_jit_reset")
    assert hasattr(env, "_jit_step")
    assert hasattr(env, "_jit_rollout")


def test_jit_wrapper_reset_shape():
    env = JitWrapper(_base_env(), cache_dir=None)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_type(obs, jnp.uint8)


def test_jit_wrapper_step_shape():
    env = JitWrapper(_base_env(), cache_dir=None)
    _, state = env.reset(_key)
    obs, _, reward, done, _ = env.step(state, jnp.int32(0))
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_jit_wrapper_rollout_shape():
    env = JitWrapper(_base_env(), cache_dir=None)
    _, state = env.reset(_key)
    actions = jnp.zeros(_N_STEPS, dtype=jnp.int32)
    _, (obs, reward, done, _) = env.rollout(state, actions)
    chex.assert_shape(obs, (_N_STEPS, 210, 160, 3))
    chex.assert_shape(reward, (_N_STEPS,))
    chex.assert_shape(done, (_N_STEPS,))


def test_jit_wrapper_repr():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert repr(env) == "JitWrapper<AtariEnv<breakout>>"


def test_jit_wrapper_spaces_delegate():
    inner = _base_env()
    env = JitWrapper(inner, cache_dir=None)
    assert env.observation_space == inner.observation_space
    assert env.action_space == inner.action_space


def test_jit_wrapper_unwrapped():
    env = JitWrapper(_base_env(), cache_dir=None)
    assert isinstance(env.unwrapped, AtariEnv)


def test_make_jit_compile_returns_jit_wrapper():
    env = make(_BREAKOUT, jit_compile=True)
    assert isinstance(env, JitWrapper)
