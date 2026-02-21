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
import jax
import jax.numpy as jnp

_LUMA = jnp.array([0.299, 0.587, 0.114], dtype=jnp.float32)


def preprocess(frame: chex.Array) -> chex.Array:
    """
    Convert a raw TIA frame to the standard DQN 84×84 grayscale observation.

    Applies the NTSC luminance formula to convert RGB to grayscale, then
    resizes from `(210, 160)` to `(84, 84)` using bilinear interpolation.

    Parameters
    ----------
    frame : chex.Array
        uint8[210, 160, 3] — Raw RGB frame from `state.screen`.

    Returns
    -------
    obs : chex.Array
        uint8[84, 84] — Preprocessed grayscale observation.
    """
    gray = jnp.dot(frame.astype(jnp.float32), _LUMA)
    resized = jax.image.resize(gray, (84, 84), method="bilinear")
    return resized.astype(jnp.uint8)
