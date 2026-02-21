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

"""pytest session configuration and shared fixtures.

Enables the JAX persistent XLA compilation cache so that expensive JIT
compilations (e.g. the full emulator reset/step graph) are written to disk
on the first run and reused on all subsequent runs.

FakeEnv is a ROM-free environment that mirrors the AtariEnv/Wrapper
interface; it is available to all test subdirectories.
"""

import os

import jax
import jax.numpy as jnp
import pytest

from atarax.core.state import new_atari_state
from atarax.env._compile import setup_cache
from atarax.env.spaces import Box, Discrete


def pytest_configure(config):
    cache_dir = os.path.join(os.path.dirname(__file__), ".jax-cache")
    setup_cache(cache_dir)


class FakeEnv:
    """Minimal ROM-free env that mirrors the AtariEnv/Wrapper interface."""

    def reset(self, key):
        state = new_atari_state()
        return state.screen, state

    def step(self, state, action):
        new_state = state.__replace__(
            episode_frame=(state.episode_frame + jnp.int32(1)).astype(jnp.int32),
            reward=jnp.float32(1.0),
            terminal=jnp.bool_(False),
            lives=jnp.int32(3),
        )
        done = jnp.bool_(False)
        info = {
            "lives": new_state.lives,
            "episode_frame": new_state.episode_frame,
            "truncated": jnp.bool_(False),
        }
        return new_state.screen, new_state, new_state.reward, done, info

    def sample(self, key):
        return jax.random.randint(key, shape=(), minval=0, maxval=18, dtype=jnp.int32)

    @property
    def observation_space(self):
        return Box(low=0.0, high=255.0, shape=(210, 160, 3), dtype=jnp.uint8)

    @property
    def action_space(self):
        return Discrete(n=18)


@pytest.fixture(scope="session")
def fake_env():
    """Session-scoped FakeEnv instance."""
    return FakeEnv()


@pytest.fixture(scope="session")
def fake_env_class():
    """Session-scoped FakeEnv class — for tests that subclass it."""
    return FakeEnv
