# Kangaroo

> Game ID: `"atari/kangaroo-v0"`

A mother kangaroo must climb four platform levels to rescue her joey,
punching monkeys that throw apples along the way.  Reach the top floor to
rescue the joey and reset to the ground.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — punch |
| `2` | UP — jump / climb ladder |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

Points are awarded for punching monkeys, dodging apples, and rescuing the joey.

| Event | Points |
| --- | --- |
| Monkey punched | +200 |
| Apple dodged (rolls off screen) | +100 |
| Joey rescued (reach top floor) | +1000 |

## Episode End

The episode ends when all lives are lost.  A life is lost when an apple hits
the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Platforms (4 floors) | y = 185, 140, 95, 50 (bottom edge) |
| Player x range | x ∈ [10, 150] |
| Ladders | x = 80 (floor 0→1), x = 50 (floor 1→2), x = 110 (floor 2→3) |
| Monkeys (3) | one per floor 1–3; patrol x ∈ [10, 150] |
| Joey | top-right of floor 3 (x ∈ [130, 148]) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/kangaroo-v0")
play("atari/kangaroo-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Punch |
| `↑` / `W` | Jump / climb |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
