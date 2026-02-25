# H.E.R.O

> Game ID: `"atari/hero-v0"`

Roderick Hero uses a helicopter backpack and laser to descend through mine shafts, rescuing trapped miners while blasting enemies and walls and managing energy.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(8)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (shoot laser) |
| `2` | UP (fly up) |
| `3` | RIGHT |
| `4` | DOWN (descend) |
| `5` | LEFT |
| `6` | UP + FIRE |
| `7` | RIGHT + FIRE |

## Reward

| Event | Reward |
| --- | --- |
| Enemy destroyed | +75 |
| Wall section blasted | +10 |
| Miner rescued | +1000 |

## Episode End

The episode ends when all lives are lost. A life is lost by touching an enemy creature or by running out of energy (the energy bar depletes by 1 each emulated frame).

## Lives

The player starts with 4 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player x range | x ∈ [5, 155] |
| Player y range | y ∈ [20, 195] |
| Miner (rescue target) | x = 80, y = 195 |
| Wall row 1 | y = 80 |
| Wall row 2 | y = 140 |
| Energy bar | top 8 rows (width proportional to energy) |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/hero-v0")
play("atari/hero-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (shoot laser) |
| `↑` / `W` | UP (fly up) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN (descend) |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
