# Beam Rider

> Game ID: `"atari/beam_rider-v0"`

Ride a beam toward a distant target, shooting enemy drones in 15 vertical
lanes while dodging projectiles.  Clear all drones to advance to the next
sector.

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
| `2` | LEFT — move one lane left |
| `3` | RIGHT — move one lane right |
| `4` | LEFT + FIRE |
| `5` | RIGHT + FIRE |

## Reward

Points are awarded for destroying drones and clearing sectors.

| Event | Points |
| --- | --- |
| Drone destroyed | +100 |
| Sector complete | +1000 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a drone
reaches the player's level or a bullet hits the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Lanes | 15 vertical lanes x ∈ [8, 148] |
| Player | y = 170, starts in lane 7 (centre) |
| Drones | spawn at y = 30, descend toward player |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/beam_rider-v0")
play("atari/beam_rider-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `←` / `A` | Move lane left |
| `→` / `D` | Move lane right |
| `Esc` / close window | Quit |
