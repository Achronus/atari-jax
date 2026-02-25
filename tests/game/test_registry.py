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

"""Tests for the game registry: GAMES, GAME_SPECS, GAME_GROUPS, get_game().

Run with:
    pytest tests/game/test_registry.py -v
"""

import pytest

from atarax.env.atari_env import AtariEnv
from atarax.env.spec import EnvSpec
from atarax.games.registry import GAME_GROUPS, GAME_SPECS, GAMES, get_game


def test_games_is_dict():
    assert isinstance(GAMES, dict)


def test_games_contains_breakout():
    assert "breakout" in GAMES


def test_games_values_are_atari_env_types():
    for cls in GAMES.values():
        assert issubclass(cls, AtariEnv)


def test_game_specs_is_list():
    assert isinstance(GAME_SPECS, list)


def test_game_specs_all_env_spec():
    assert all(isinstance(s, EnvSpec) for s in GAME_SPECS)


def test_game_specs_engine_is_atari():
    assert all(s.engine == "atari" for s in GAME_SPECS)


def test_game_specs_matches_games_keys():
    names = {s.env_name for s in GAME_SPECS}
    assert names == set(GAMES.keys())


def test_game_specs_contains_breakout():
    games = [s.env_name for s in GAME_SPECS]
    assert "breakout" in games


def test_game_groups_is_dict():
    assert isinstance(GAME_GROUPS, dict)


def test_game_groups_expected_keys():
    assert {"atari5", "atari10", "atari26", "atari57"} == set(GAME_GROUPS.keys())


def test_game_groups_values_are_lists_of_env_spec():
    for group_name, specs in GAME_GROUPS.items():
        assert isinstance(specs, list), f"{group_name} value is not a list"
        assert all(isinstance(s, EnvSpec) for s in specs), (
            f"{group_name} contains non-EnvSpec entries"
        )


def test_game_groups_sizes():
    assert len(GAME_GROUPS["atari5"]) == 5
    assert len(GAME_GROUPS["atari10"]) == 10
    assert len(GAME_GROUPS["atari26"]) == 26
    assert len(GAME_GROUPS["atari57"]) == 57


def test_game_groups_breakout_in_all():
    for name, specs in GAME_GROUPS.items():
        games = [s.env_name for s in specs]
        assert "breakout" in games, f"breakout missing from {name}"


def test_get_game_returns_atari_env_type():
    cls = get_game("breakout")
    assert issubclass(cls, AtariEnv)


def test_get_game_case_insensitive():
    assert get_game("breakout") is get_game("Breakout")


def test_get_game_unknown_raises():
    with pytest.raises(ValueError, match="Unknown game"):
        get_game("not_a_game")
