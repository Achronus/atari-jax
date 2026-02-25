# Phoenix

> Game ID: `"atari/phoenix-v0"`

Space shooter: waves of birds dive at the player from the top.  Shoot them
before they reach the bottom.  A shield can block alien fire briefly.
Clear all birds to advance to the next wave.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | RIGHT |
| `3` | LEFT |
| `4` | SHIELD |
| `5` | FIRE + RIGHT |

## Reward

Points are awarded for each bird shot, scaled by row.

| Event | Points |
| --- | --- |
| Bird (rows 0–1, top) | +10 |
| Bird (rows 2–3, bottom) | +20 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a bird
reaches the player's level or an alien bullet hits the player (shield
blocks the bullet).

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player | y = 175, x ∈ [5, 147] |
| Bird formation | 4 rows × 8 cols; starts x = 8, y = 20; col spacing = 16px, row spacing = 12px |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/phoenix-v0")
play("atari/phoenix-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `→` / `D` | Move right |
| `←` / `A` | Move left |
| `Shift` | Shield |
| `Esc` / close window | Quit |
