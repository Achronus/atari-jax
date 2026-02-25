# Crazy Climber

> Game ID: `"atari/crazy_climber-v0"`

Climb a skyscraper by grabbing window ledges while avoiding obstacles thrown from windows and a giant condor that drops droppings. Reach the top as many times as possible without losing all lives.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(8)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | UP |
| `2` | UP + RIGHT |
| `3` | RIGHT |
| `4` | DOWN + RIGHT |
| `5` | DOWN |
| `6` | DOWN + LEFT |
| `7` | LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Reach a bonus flag | +1000 |
| Reach the top | +3000 |

Bonus flags are located at y = 150, 110, 70, and 30. After reaching the top, the climber resets to the bottom and climbing continues.

## Episode End

The episode ends when all lives are lost. Lives are lost by grabbing an open window, being hit by a falling flowerpot, or being struck by a condor dropping.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Building x range | x ∈ [10, 150] |
| Ground (start) | y = 185 |
| Top (goal) | y = 20 |
| Condor altitude | y = 60 |
| Bonus flag y-positions | y = 150, 110, 70, 30 |
| Window centres (x) | 40, 80, 120 per floor |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/crazy_climber-v0")
play("atari/crazy_climber-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
