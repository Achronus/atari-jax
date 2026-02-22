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

"""Unit tests for precompile_all() and the manifest helper functions.

No ROM files or JAX compilation are required.  The make() / make_vec() calls
inside precompile_all() are replaced with MagicMock objects, and GAME_IDS is
patched to a single entry so the outer loop runs once.

Run with:
    pytest tests/env/test_precompile.py -v
"""

import json
import pathlib
from unittest.mock import MagicMock, patch

from atarax.env.make import (
    _MANIFEST_NAME,
    _config_entry,
    _manifest_append,
    _manifest_has_config,
    _wrapper_str,
    precompile_all,
)
from atarax.env.wrappers import GrayscaleObservation, ResizeObservation
from atarax.env.wrappers.base import _WrapperFactory

_DEFAULT_ENTRY = {"n_envs": 1, "n_steps": None, "preset": False, "wrappers": None}


def _write_manifest(directory: pathlib.Path, configs: list) -> None:
    (directory / _MANIFEST_NAME).write_text(json.dumps(configs))


def _read_manifest(directory: pathlib.Path) -> list:
    return json.loads((directory / _MANIFEST_NAME).read_text())


def _mock_env() -> MagicMock:
    """Return a mock env whose reset/step/sample signatures satisfy precompile_all."""
    env = MagicMock()
    state = MagicMock()
    env.reset.return_value = (MagicMock(), state)
    env.step.return_value = (MagicMock(), state, 0.0, False, {})
    env.sample.return_value = MagicMock()
    return env


class TestConfigEntry:
    def test_defaults(self):
        entry = _config_entry(1, None, False, None)
        assert entry == _DEFAULT_ENTRY

    def test_with_steps_and_preset(self):
        entry = _config_entry(4, 128, True, None)
        assert entry == {"n_envs": 4, "n_steps": 128, "preset": True, "wrappers": None}

    def test_wrapper_names_serialised(self):
        entry = _config_entry(1, None, False, [GrayscaleObservation])
        assert entry["wrappers"] == ["GrayscaleObservation"]

    def test_multiple_wrappers(self):
        from atarax.env.wrappers import ClipReward

        entry = _config_entry(1, None, False, [GrayscaleObservation, ClipReward])
        assert entry["wrappers"] == ["GrayscaleObservation", "ClipReward"]

    def test_factory_serialised_with_kwargs(self):
        entry = _config_entry(1, None, False, [ResizeObservation(h=64, w=64)])
        assert entry["wrappers"] == ["ResizeObservation(h=64, w=64)"]

    def test_factory_distinct_from_different_params(self):
        e1 = _config_entry(1, None, False, [ResizeObservation(h=64, w=64)])
        e2 = _config_entry(1, None, False, [ResizeObservation(h=84, w=84)])
        assert e1 != e2

    def test_mixed_class_and_factory(self):
        entry = _config_entry(
            1, None, False, [GrayscaleObservation, ResizeObservation(h=64, w=64)]
        )
        assert entry["wrappers"] == ["GrayscaleObservation", "ResizeObservation(h=64, w=64)"]


class TestWrapperStr:
    def test_class_returns_name(self):
        assert _wrapper_str(GrayscaleObservation) == "GrayscaleObservation"

    def test_factory_with_kwargs(self):
        factory = ResizeObservation(h=64, w=64)
        assert _wrapper_str(factory) == "ResizeObservation(h=64, w=64)"

    def test_factory_kwargs_sorted_alphabetically(self):
        from atarax.env.wrappers import AtariPreprocessing

        factory = AtariPreprocessing(n_stack=2, h=64, w=64)
        assert _wrapper_str(factory) == "AtariPreprocessing(h=64, n_stack=2, w=64)"

    def test_factory_no_kwargs_returns_class_name(self):
        factory = _WrapperFactory(GrayscaleObservation)
        assert _wrapper_str(factory) == "GrayscaleObservation"

    def test_repr_value_types(self):
        from atarax.env.wrappers import FrameStackObservation

        factory = FrameStackObservation(n_stack=8)
        assert _wrapper_str(factory) == "FrameStackObservation(n_stack=8)"


class TestManifestHasConfig:
    def test_missing_file_returns_false(self, tmp_path):
        assert not _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_entry_present_returns_true(self, tmp_path):
        _write_manifest(tmp_path, [_DEFAULT_ENTRY])
        assert _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_entry_absent_returns_false(self, tmp_path):
        _write_manifest(
            tmp_path, [{"n_envs": 4, "n_steps": None, "preset": True, "wrappers": None}]
        )
        assert not _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_corrupt_json_returns_false(self, tmp_path):
        (tmp_path / _MANIFEST_NAME).write_text("not valid json{{")
        assert not _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_entry_among_multiple_configs(self, tmp_path):
        other = {"n_envs": 8, "n_steps": 256, "preset": True, "wrappers": None}
        _write_manifest(tmp_path, [other, _DEFAULT_ENTRY])
        assert _manifest_has_config(tmp_path, _DEFAULT_ENTRY)


class TestManifestAppend:
    def test_creates_file_with_entry(self, tmp_path):
        _manifest_append(tmp_path, _DEFAULT_ENTRY)
        configs = _read_manifest(tmp_path)
        assert _DEFAULT_ENTRY in configs

    def test_does_not_duplicate_entry(self, tmp_path):
        _manifest_append(tmp_path, _DEFAULT_ENTRY)
        _manifest_append(tmp_path, _DEFAULT_ENTRY)
        configs = _read_manifest(tmp_path)
        assert configs.count(_DEFAULT_ENTRY) == 1

    def test_appends_to_existing_entries(self, tmp_path):
        first = {"n_envs": 4, "n_steps": 128, "preset": True, "wrappers": None}
        _write_manifest(tmp_path, [first])
        _manifest_append(tmp_path, _DEFAULT_ENTRY)
        configs = _read_manifest(tmp_path)
        assert first in configs and _DEFAULT_ENTRY in configs

    def test_multiple_distinct_configs_coexist(self, tmp_path):
        entries = [
            {"n_envs": 1, "n_steps": None, "preset": False, "wrappers": None},
            {"n_envs": 4, "n_steps": 128, "preset": True, "wrappers": None},
            {
                "n_envs": 8,
                "n_steps": 256,
                "preset": False,
                "wrappers": ["GrayscaleObservation"],
            },
        ]
        for e in entries:
            _manifest_append(tmp_path, e)
        configs = _read_manifest(tmp_path)
        for e in entries:
            assert e in configs

    def test_handles_corrupt_existing_file(self, tmp_path):
        (tmp_path / _MANIFEST_NAME).write_text("{{broken")
        _manifest_append(tmp_path, _DEFAULT_ENTRY)  # must not raise
        configs = _read_manifest(tmp_path)
        assert _DEFAULT_ENTRY in configs


_ONE_GAME = {"breakout": 12}


class TestPrecompileAll:
    def test_skips_when_config_already_cached(self, tmp_path):
        _manifest_append(tmp_path, _DEFAULT_ENTRY)

        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make") as mock_make,
        ):
            precompile_all(cache_dir=tmp_path)

        mock_make.assert_not_called()

    def test_writes_manifest_after_compilation(self, tmp_path):
        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make", return_value=_mock_env()),
        ):
            precompile_all(cache_dir=tmp_path)

        assert _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_clear_cache_removes_old_entries(self, tmp_path):
        stale = {"n_envs": 99, "n_steps": None, "preset": True, "wrappers": None}
        _manifest_append(tmp_path, stale)

        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make", return_value=_mock_env()),
        ):
            precompile_all(cache_dir=tmp_path, clear_cache=True)

        assert not _manifest_has_config(tmp_path, stale)

    def test_clear_cache_writes_new_manifest(self, tmp_path):
        stale = {"n_envs": 99, "n_steps": None, "preset": True, "wrappers": None}
        _manifest_append(tmp_path, stale)

        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make", return_value=_mock_env()),
        ):
            precompile_all(cache_dir=tmp_path, clear_cache=True)

        assert _manifest_has_config(tmp_path, _DEFAULT_ENTRY)

    def test_no_manifest_when_cache_dir_none(self):
        """cache_dir=None disables all manifest I/O."""
        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make", return_value=_mock_env()),
        ):
            precompile_all(cache_dir=None)  # must not raise

    def test_n_envs_gt_1_calls_make_vec(self, tmp_path):
        mock_vec = MagicMock()
        states = MagicMock()
        mock_vec.reset.return_value = (MagicMock(), states)
        mock_vec.step.return_value = (MagicMock(), states, MagicMock(), MagicMock(), {})

        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make_vec", return_value=mock_vec) as mock_make_vec,
        ):
            precompile_all(n_envs=4, cache_dir=tmp_path)

        mock_make_vec.assert_called_once()
        entry = _config_entry(4, None, False, None)
        assert _manifest_has_config(tmp_path, entry)

    def test_second_call_with_different_config_does_not_skip(self, tmp_path):
        """Two distinct configs must both be compiled and stored."""
        with (
            patch("atarax.env.make.GAME_IDS", _ONE_GAME),
            patch("atarax.env.make.make", return_value=_mock_env()) as mock_make,
        ):
            precompile_all(cache_dir=tmp_path)
            precompile_all(n_envs=1, preset=True, cache_dir=tmp_path)

        assert mock_make.call_count == 2  # once per unique config
        assert _manifest_has_config(tmp_path, _DEFAULT_ENTRY)
        assert _manifest_has_config(tmp_path, _config_entry(1, None, True, None))
