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

"""Shared pygame renderer for human-mode display of an AtariState screen."""

import numpy as np

from atari_jax.core.state import AtariState


def render(
    state: AtariState,
    *,
    scale: int = 2,
    caption: str = "atari-jax",
) -> None:
    """
    Display the current frame in a pygame window.

    Lazy-imports pygame so the library remains optional; install it with
    ``pip install pygame`` when human rendering is needed.

    Parameters
    ----------
    state : AtariState
        Current machine state.  Uses `state.screen` (uint8[210, 160, 3]).
    scale : int
        Integer upscale factor applied to the native 160 × 210 pixel screen.
    caption : str
        Window title shown in the title bar.
    """
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            "pygame is required for human rendering. "
            "Install it with: pip install pygame"
        ) from exc

    screen_np = np.asarray(state.screen)          # (210, 160, 3) uint8
    h, w = screen_np.shape[:2]

    # pygame.surfarray.make_surface expects (width, height, 3)
    surf = pygame.surfarray.make_surface(screen_np.transpose(1, 0, 2))

    if not pygame.get_init():
        pygame.init()

    display = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption(caption)
    scaled = pygame.transform.scale(surf, (w * scale, h * scale))
    display.blit(scaled, (0, 0))
    pygame.display.flip()
