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

FakeEnv is a dependency-free environment that mirrors the Env/Wrapper interface;
it is available to all test subdirectories.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env.spaces import Box, Discrete


@chex.dataclass
class FakeState:
    """Minimal pytree state for use in wrapper tests."""

    lives: chex.Array          # int32
    episode_step: chex.Array   # int32
    reward: chex.Array         # float32
    terminal: chex.Array       # bool


def _make_fake_state() -> FakeState:
    return FakeState(
        lives=jnp.int32(3),
        episode_step=jnp.int32(0),
        reward=jnp.float32(0.0),
        terminal=jnp.bool_(False),
    )


class FakeEnv:
    """Minimal env that mirrors the Env/Wrapper interface for wrapper tests."""

    def reset(self, key):
        state = _make_fake_state()
        obs = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
        return obs, state

    def step(self, state, action):
        new_state = state.__replace__(
            episode_step=(state.episode_step + jnp.int32(1)).astype(jnp.int32),
            reward=jnp.float32(1.0),
            terminal=jnp.bool_(False),
            lives=jnp.int32(3),
        )
        done = jnp.bool_(False)
        obs = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
        info = {
            "lives": new_state.lives,
            "episode_step": new_state.episode_step,
            "truncated": jnp.bool_(False),
        }
        return obs, new_state, new_state.reward, done, info

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
