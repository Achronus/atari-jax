# Asteroids

> Game ID: `"atari/asteroids-v0"`

Pilot a rotating spaceship through an asteroid field, destroying rocks with
torpedoes.  Large asteroids split into smaller ones when hit.  Clearing all
rocks advances to the next wave.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(14)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP — thrust |
| `3` | RIGHT — rotate clockwise |
| `4` | LEFT — rotate counter-clockwise |
| `5` | DOWN — hyperspace warp |
| `6` | UP + FIRE |
| `7` | RIGHT + FIRE |
| `8` | LEFT + FIRE |
| `9` | DOWN + FIRE |
| `10` | UP + RIGHT |
| `11` | UP + LEFT |
| `12` | UP + RIGHT + FIRE |
| `13` | UP + LEFT + FIRE |

## Reward

Points are awarded for each asteroid or saucer destroyed.

| Target | Points |
| --- | --- |
| Large asteroid | +20 |
| Medium asteroid | +50 |
| Small asteroid | +100 |

## Episode End

The episode ends when all lives are lost.  A life is lost each time the
ship collides with an asteroid.  Clearing all rocks advances the wave.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Playfield | 160 × 210 px, wraps on all edges |
| Ship start | x = 80, y = 105 |
| Rock radii | large ≈ 12 px, medium ≈ 7 px, small ≈ 4 px |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/asteroids-v0")
play("atari/asteroids-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Thrust |
| `→` / `D` | Rotate clockwise |
| `←` / `A` | Rotate counter-clockwise |
| `↓` / `S` | Hyperspace warp |
| `Esc` / close window | Quit |
