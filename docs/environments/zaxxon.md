# Zaxxon

> Game ID: `"atari/zaxxon-v0"`

Isometric scrolling space fortress assault. Fly your spaceship through a fortress, navigating walls, shooting gun emplacements and jet fighters, then fight the robot boss Zaxxon.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(9)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP (gain altitude) |
| `3` | RIGHT |
| `4` | DOWN (lose altitude) |
| `5` | LEFT |
| `6` | FIRE+UP |
| `7` | FIRE+RIGHT |
| `8` | FIRE+LEFT |

## Reward

| Event | Reward |
| --- | --- |
| Gun emplacement destroyed | `+150` |
| Fuel tank destroyed | `+200` |
| Jet fighter destroyed | `+1000` |
| Zaxxon boss destroyed | `+1000` |

## Episode End

The episode ends when all lives are lost. A life is lost by colliding with an enemy at the wrong altitude or by running out of fuel.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player (fixed screen position) | x = 60, y = 105 |
| Altitude range | y ∈ [30, 180] (higher y = lower altitude) |
| Ground (isometric horizon) | y ≥ 170 |
| Enemies (world coords) | spread from x = 200 to x = 1200 |
| Fuel | starts at 2000, decrements each frame |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/zaxxon-v0")
play("atari/zaxxon-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP (gain altitude) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN (lose altitude) |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
