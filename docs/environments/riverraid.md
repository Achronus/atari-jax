# River Raid

> Game ID: `"atari/riverraid-v0"`

Fly a jet down an endless river; shoot tanks, helicopters, and ships while
managing fuel.  Fly over fuel depots to refuel.  Collide with the banks or
run out of fuel and lose a life.

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
| `4` | UP — increase speed |
| `5` | DOWN — decrease speed |

## Reward

Points are awarded for each enemy destroyed.

| Event | Points |
| --- | --- |
| Tanker destroyed | +30 |
| Helicopter destroyed | +60 |
| Jet destroyed | +100 |
| Ship destroyed | +80 |

## Episode End

The episode ends when all lives are lost.  A life is lost by hitting the
river banks, colliding with an enemy, or running out of fuel.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| River channel | x ∈ [20, 140] |
| Banks (grass) | x ∈ [0, 20) and x ∈ [140, 160) |
| Player jet | y = 170, x ∈ [20, 132] |
| Enemies (8) | scroll down with the river |
| Fuel depots (4) | scroll down; fly over to refuel |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/riverraid-v0")
play("atari/riverraid-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `→` / `D` | Move right |
| `←` / `A` | Move left |
| `↑` / `W` | Increase speed |
| `↓` / `S` | Decrease speed |
| `Esc` / close window | Quit |
