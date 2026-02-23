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

"""Tests for _WrapperFactory and the Wrapper.__new__ factory mode.

Exercises factory creation, stored metadata, calling the factory to produce a
live wrapper, and the mixed class + factory pattern used by make().

No ROM files required — uses FakeEnv from conftest.

Run with:
    pytest tests/wrappers/test_factory.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from atarax.env.wrappers.base import _WrapperFactory

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


class TestFactoryCreation:
    def test_resize_without_env_is_factory(self):
        assert isinstance(ResizeObservation(h=64, w=64), _WrapperFactory)

    def test_frame_stack_without_env_is_factory(self):
        assert isinstance(FrameStackObservation(n_stack=8), _WrapperFactory)

    def test_atari_preprocessing_without_env_is_factory(self):
        assert isinstance(AtariPreprocessing(h=64, w=64), _WrapperFactory)

    def test_resize_factory_stores_cls(self):
        factory = ResizeObservation(h=64, w=64)
        assert factory._cls is ResizeObservation

    def test_resize_factory_stores_kwargs(self):
        factory = ResizeObservation(h=64, w=64)
        assert factory._kwargs == {"h": 64, "w": 64}

    def test_frame_stack_factory_stores_kwargs(self):
        factory = FrameStackObservation(n_stack=8)
        assert factory._kwargs == {"n_stack": 8}

    def test_atari_preprocessing_factory_stores_kwargs(self):
        factory = AtariPreprocessing(h=64, w=64, n_stack=2)
        assert factory._kwargs == {"h": 64, "w": 64, "n_stack": 2}

    def test_default_params_stored_when_explicit(self):
        factory = ResizeObservation(h=84, w=84)
        assert factory._kwargs == {"h": 84, "w": 84}


class TestNormalModeUnchanged:
    def test_resize_with_env_is_not_factory(self, fake_env):
        env = ResizeObservation(GrayscaleObservation(fake_env), h=64, w=64)
        assert isinstance(env, ResizeObservation)
        assert not isinstance(env, _WrapperFactory)

    def test_resize_default_params_with_env(self, fake_env):
        env = ResizeObservation(GrayscaleObservation(fake_env))
        assert isinstance(env, ResizeObservation)

    def test_frame_stack_with_env_is_not_factory(self, fake_env):
        inner = ResizeObservation(GrayscaleObservation(fake_env), h=20, w=20)
        env = FrameStackObservation(inner, n_stack=4)
        assert isinstance(env, FrameStackObservation)
        assert not isinstance(env, _WrapperFactory)

    def test_no_arg_wrapper_with_env_is_not_factory(self, fake_env):
        env = GrayscaleObservation(fake_env)
        assert isinstance(env, GrayscaleObservation)


class TestFactoryCall:
    def test_resize_factory_creates_resize_instance(self, fake_env):
        factory = ResizeObservation(h=64, w=64)
        env = factory(GrayscaleObservation(fake_env))
        assert isinstance(env, ResizeObservation)

    def test_resize_factory_custom_size_obs_shape(self, fake_env):
        factory = ResizeObservation(h=64, w=64)
        env = factory(GrayscaleObservation(fake_env))
        obs, _ = env.reset(_key)
        chex.assert_shape(obs, (64, 64))

    def test_resize_factory_observation_space(self, fake_env):
        factory = ResizeObservation(h=64, w=64)
        env = factory(GrayscaleObservation(fake_env))
        assert env.observation_space.shape == (64, 64)

    def test_resize_factory_step_obs_shape(self, fake_env):
        factory = ResizeObservation(h=64, w=64)
        env = factory(GrayscaleObservation(fake_env))
        _, state = env.reset(_key)
        obs, _, _, _, _ = env.step(state, _action)
        chex.assert_shape(obs, (64, 64))

    def test_frame_stack_factory_custom_n_stack(self, fake_env):
        frame_factory = FrameStackObservation(n_stack=8)
        inner = ResizeObservation(GrayscaleObservation(fake_env), h=20, w=20)
        env = frame_factory(inner)
        obs, _ = env.reset(_key)
        chex.assert_shape(obs, (20, 20, 8))
        assert env.observation_space.shape == (20, 20, 8)

    def test_same_factory_callable_multiple_times(self, fake_env):
        """One factory instance can wrap different envs."""
        factory = ResizeObservation(h=32, w=32)
        env_a = factory(GrayscaleObservation(fake_env))
        env_b = factory(GrayscaleObservation(fake_env))
        obs_a, _ = env_a.reset(_key)
        obs_b, _ = env_b.reset(_key)
        chex.assert_shape(obs_a, (32, 32))
        chex.assert_shape(obs_b, (32, 32))


class TestMixedWrappersList:
    def test_class_and_factory_applied_in_order(self, fake_env):
        wrappers = [GrayscaleObservation, ResizeObservation(h=64, w=64)]
        env = fake_env
        for w in wrappers:
            env = w(env)
        obs, _ = env.reset(_key)
        chex.assert_shape(obs, (64, 64))

    def test_multiple_factories_in_list(self, fake_env):
        wrappers = [
            GrayscaleObservation,
            ResizeObservation(h=64, w=64),
            FrameStackObservation(n_stack=2),
        ]
        env = fake_env
        for w in wrappers:
            env = w(env)
        obs, _ = env.reset(_key)
        chex.assert_shape(obs, (64, 64, 2))

    def test_two_factories_different_params_are_independent(self, fake_env):
        """Different factory instances should produce different configurations."""
        factory_small = ResizeObservation(h=32, w=32)
        factory_large = ResizeObservation(h=64, w=64)

        env_small = factory_small(GrayscaleObservation(fake_env))
        env_large = factory_large(GrayscaleObservation(fake_env))

        assert env_small.observation_space.shape == (32, 32)
        assert env_large.observation_space.shape == (64, 64)
