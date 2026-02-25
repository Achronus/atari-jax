# Frostbite

> Game ID: `"atari/frostbite-v0"`

Hop across ice floes to build an igloo while avoiding hazards: fish, crabs, and falling into freezing water. Temperature drops constantly; build the igloo to shelter against the cold.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (jump / toggle direction) |
| `2` | UP (jump to floe above) |
| `3` | RIGHT |
| `4` | DOWN (jump to floe below) |
| `5` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Ice block added to igloo | +10 |
| Fish caught | +200 |
| Igloo complete | +1000 |

Each newly visited floe per cycle adds one ice block (+10). Completing all 8 blocks awards +1000 and resets the temperature and visited-floe tracking.

## Episode End

The episode ends when all lives are lost. A life is lost by colliding with an enemy (crab or fish) or when the temperature counter reaches zero.

## Lives

The player starts with 4 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Floe row y-centres | y = 50, 90, 130, 170 |
| Ground (igloo build area) | y >= 185 |
| Igloo centre x | x = 80 |
| Blocks needed | 8 |
| Floes per row | 5 (width 24 px, spacing 32 px) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/frostbite-v0")
play("atari/frostbite-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (jump / toggle direction) |
| `↑` / `W` | UP (jump to floe above) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN (jump to floe below) |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
