# Assault

> Game ID: `"atari/assault-v0"`

Pilot a rotating turret at the bottom of the screen, shooting waves of
descending enemy ships before they reach your position.  Enemies fire back;
dodge or be destroyed.

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
| `2` | UP — thrust / move forward |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | UP + FIRE |

## Reward

Points are awarded for each enemy ship destroyed.

| Event | Points |
| --- | --- |
| Enemy destroyed | +10 |

## Episode End

The episode ends when all lives are lost.  A life is lost when the player
is hit by an enemy bullet.

## Lives

The player starts with 4 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player cannon | y = 185, x movable ∈ [10, 150] |
| Enemy formation | 3 rows × 3–5 columns; y starting at 30 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/assault-v0")
play("atari/assault-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Move / thrust |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
