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

"""Tests for AtariEnv compile_mode parameter validation.

These tests exercise `AtariEnv(compile_mode=...)` and `AtariEnv(group=...)`
directly — not the factory functions.

Run with:
    pytest tests/make/test_compile_mode.py -v
"""

import pytest

from atarax.env.atari_env import AtariEnv


def test_invalid_compile_mode_raises():
    with pytest.raises(ValueError, match="compile_mode"):
        AtariEnv("breakout", compile_mode="turbo")


def test_group_mode_requires_group_arg():
    with pytest.raises(ValueError, match="group"):
        AtariEnv("breakout", compile_mode="group")


def test_group_arg_forbidden_in_all_mode():
    with pytest.raises(ValueError, match="group"):
        AtariEnv("breakout", compile_mode="all", group="atari5")


def test_group_arg_forbidden_in_single_mode():
    with pytest.raises(ValueError, match="group"):
        AtariEnv("breakout", compile_mode="single", group="atari5")


def test_group_unknown_predefined_name_raises():
    with pytest.raises(ValueError, match="Unknown group"):
        AtariEnv("breakout", compile_mode="group", group="not_a_real_group")


def test_group_unknown_game_name_raises():
    with pytest.raises(ValueError, match="Unknown game"):
        AtariEnv("breakout", compile_mode="group", group=["breakout", "not_a_game"])


def test_group_game_id_not_in_group_raises():
    with pytest.raises(ValueError, match="not a member"):
        AtariEnv("alien", compile_mode="group", group="atari5")
