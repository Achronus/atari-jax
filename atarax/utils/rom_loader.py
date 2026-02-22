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

import chex
import jax.numpy as jnp


def load_rom(game_id: str) -> chex.Array:
    """
    Load ROM bytes for a named game using the ale-py ROM index.

    ale-py is an optional dependency — this function raises a clear
    `ImportError` if it is not installed rather than failing at module
    import time.

    Parameters
    ----------
    game_id : str
        ALE game identifier (e.g. `"breakout"`, `"pong"`).

    Returns
    -------
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes ready for use with `emulate_frame`.
    """
    try:
        from ale_py.roms import get_rom_path
    except ImportError as exc:
        raise ImportError(
            "ale-py is required to load ROMs by name. "
            "Install it with: pip install ale-py"
        ) from exc

    path = get_rom_path(game_id)
    return jnp.array(bytearray(path.read_bytes()), dtype=jnp.uint8)
