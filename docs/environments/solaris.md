# Solaris

> Game ID: `"atari/solaris-v0"`

Defend the solar system from Zylon squadrons. Navigate a star map to reach sectors under attack, then engage in close-range combat. Manage fuel and protect friendly planets.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(8)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE |
| `2` | UP (thrust forward) |
| `3` | RIGHT |
| `4` | DOWN (thrust back) |
| `5` | LEFT |
| `6` | WARP (hyperspace jump) |
| `7` | FIRE+UP |

## Reward

| Event | Reward |
| --- | --- |
| Zylon fighter destroyed | `+250` |
| Zylon base destroyed | `+500` |

## Episode End

The episode ends when all lives are lost. A life is lost by colliding with a Zylon, being hit by an enemy projectile, or running out of fuel.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player ship | fixed at y = 160, x ∈ [5, 147] |
| Enemies | scroll down from y = 10 toward player |
| Fuel | starts at 2000, decrements each frame with extra drain when thrusting |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/solaris-v0")
play("atari/solaris-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE |
| `↑` / `W` | UP (thrust forward) |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN (thrust back) |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
