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

"""Tests for EnvSpec.

Run with:
    pytest tests/env/test_spec.py -v
"""

import pytest

from atarax.env.spec import EnvSpec


def test_parse_string():
    spec = EnvSpec.parse("atari/breakout-v0")
    assert spec.engine == "atari"
    assert spec.env_name == "breakout"
    assert spec.version == 0


def test_parse_case_insensitive():
    spec = EnvSpec.parse("atari/Breakout-v0")
    assert spec.env_name == "breakout"


def test_parse_passthrough():
    original = EnvSpec("atari", "breakout")
    result = EnvSpec.parse(original)
    assert result is original


def test_parse_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid environment ID"):
        EnvSpec.parse("breakout")


def test_id_property():
    spec = EnvSpec("atari", "breakout")
    assert spec.id == "atari/breakout-v0"


def test_str():
    spec = EnvSpec("atari", "breakout")
    assert str(spec) == "atari/breakout-v0"


def test_equality():
    assert EnvSpec("atari", "breakout") == EnvSpec("atari", "breakout")
    assert EnvSpec("atari", "breakout") != EnvSpec("atari", "pong")
