# Star Gunner

> Game ID: `"atari/star_gunner-v0"`

Destroy waves of alien fighters that swoop in formation. Move in all four directions; enemies fly in diagonal patterns and fire back. Clear all aliens in a wave to advance.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(7)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | FIRE+UP |

## Reward

| Event | Reward |
| --- | --- |
| Alien fighter destroyed (wave 1) | `+100` |
| Alien fighter destroyed (wave 2) | `+200` |
| Alien fighter destroyed (wave 3) | `+300` |
| Alien fighter destroyed (wave 4+) | `+400` |
| Wave clear bonus | `+1000` |

## Episode End

The episode ends when all lives are lost. A life is lost when an alien reaches the player's altitude or when the player is hit by an alien bullet.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player ship | x ∈ [5, 147], y ∈ [100, 185] |
| Alien formation (row 1) | y = 20 |
| Alien formation (row 2) | y = 40 |
| Aliens per row | 6 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/star_gunner-v0")
play("atari/star_gunner-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
