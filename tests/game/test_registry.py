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

"""Tests for the game registry and AtariEnvs groups.

Run with:
    pytest tests/game/test_registry.py -v
"""

import pytest

from atarax.envs import (
    ATARI_57,
    ATARI_BASE,
    ATARI_EASY,
    ATARI_HARD,
    ATARI_MEDIUM,
    AtariEnvs,
)
from atarax.game import AtaraxGame
from atarax.games.registry import GAME_SPECS, GAMES, get_game
from atarax.spec import EnvSpec

# ---------------------------------------------------------------------------
# GAMES dict
# ---------------------------------------------------------------------------


def test_games_is_dict():
    assert isinstance(GAMES, dict)


def test_games_values_are_atarax_game_types():
    for cls in GAMES.values():
        assert issubclass(cls, AtaraxGame)


# ---------------------------------------------------------------------------
# GAME_SPECS list
# ---------------------------------------------------------------------------


def test_game_specs_is_list():
    assert isinstance(GAME_SPECS, list)


def test_game_specs_length_is_57():
    assert len(GAME_SPECS) == 57


def test_game_specs_all_env_spec():
    assert all(isinstance(s, EnvSpec) for s in GAME_SPECS)


def test_game_specs_engine_is_atari():
    assert all(s.engine == "atari" for s in GAME_SPECS)


def test_game_specs_version_is_zero():
    assert all(s.version == 0 for s in GAME_SPECS)


# ---------------------------------------------------------------------------
# get_game
# ---------------------------------------------------------------------------


def test_get_game_unknown_raises():
    with pytest.raises(ValueError, match="not yet implemented"):
        get_game("atari/not_a_game-v0")


# ---------------------------------------------------------------------------
# AtariEnvs group sizes
# ---------------------------------------------------------------------------


def test_atari_base_size():
    assert ATARI_BASE.n_envs == 14


def test_atari_easy_size():
    assert ATARI_EASY.n_envs == 16


def test_atari_medium_size():
    assert ATARI_MEDIUM.n_envs == 13


def test_atari_hard_size():
    assert ATARI_HARD.n_envs == 14


def test_atari_57_size():
    assert ATARI_57.n_envs == 57


def test_tier_sizes_sum_to_57():
    total = (
        ATARI_BASE.n_envs
        + ATARI_EASY.n_envs
        + ATARI_MEDIUM.n_envs
        + ATARI_HARD.n_envs
    )
    assert total == 57


# ---------------------------------------------------------------------------
# AtariEnvs naming convention
# ---------------------------------------------------------------------------


def test_get_name_snake_case():
    group = AtariEnvs()
    assert group.get_name("SpaceInvaders") == "atari/space_invaders-v0"
    assert group.get_name("MsPacman") == "atari/ms_pacman-v0"
    assert group.get_name("UpNDown") == "atari/up_n_down-v0"
    assert group.get_name("Breakout") == "atari/breakout-v0"
    assert group.get_name("Jamesbond") == "atari/jamesbond-v0"


def test_all_names_engine_prefix():
    for name in ATARI_57.all_names():
        assert name.startswith("atari/"), f"Expected atari/ prefix: {name}"
        assert name.endswith("-v0"), f"Expected -v0 suffix: {name}"


def test_surround_in_atari_57():
    names = ATARI_57.all_names()
    assert "atari/surround-v0" in names


def test_pooyan_not_in_atari_57():
    names = ATARI_57.all_names()
    assert "atari/pooyan-v0" not in names


# ---------------------------------------------------------------------------
# EnvGroup iteration and membership
# ---------------------------------------------------------------------------


def test_iter_yields_strings():
    for name in ATARI_BASE:
        assert isinstance(name, str)


def test_contains():
    assert "Breakout" in ATARI_BASE
    assert "NotAGame" not in ATARI_BASE


def test_getitem_single():
    sub = ATARI_BASE[0]
    assert sub.n_envs == 1


def test_getitem_slice():
    sub = ATARI_BASE[:3]
    assert sub.n_envs == 3
