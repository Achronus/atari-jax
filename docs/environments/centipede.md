# Centipede

> Game ID: `"atari/centipede-v0"`

Shoot a multi-segment centipede as it zigzags down through a mushroom field.
Each hit segment leaves a mushroom; the head bounces off mushrooms and walls.
Spiders and fleas make occasional appearances.

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
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |

## Reward

Points are awarded for each target destroyed.

| Target | Points |
| --- | --- |
| Centipede body segment | +10 |
| Centipede head | +100 |
| Spider | +300 |
| Flea | +200 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a centipede
segment or the spider touches the player.  Destroying all 10 segments
advances the wave.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Grid | 10 × 10 px tiles; x ∈ [5, 155], y ∈ [0, 200] |
| Player zone | bottom 5 rows y ∈ [140, 195] |
| Player start | x = 72, y ≈ 150 |
| Centipede start | 10 segments at y = 10 |
| Mushrooms | ≈30 scattered in upper field |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/centipede-v0")
play("atari/centipede-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
