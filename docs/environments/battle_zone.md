# Battle Zone

> Game ID: `"atari/battle_zone-v0"`

Destroy enemy tanks and saucers in a top-down arena.  Enemies approach from
all sides; shoot them before they fire back.  Clearing all enemies advances
to the next wave.

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
| `2` | FORWARD |
| `3` | BACK |
| `4` | LEFT — rotate counter-clockwise |
| `5` | RIGHT — rotate clockwise |
| `6` | FORWARD + FIRE |
| `7` | BACK + FIRE |

## Reward

Points are awarded for each enemy destroyed.

| Target | Points |
| --- | --- |
| Tank destroyed | +1000 |
| Saucer destroyed | +5000 |
| Super-tank destroyed | +3000 |

## Episode End

The episode ends when all lives are lost.  A life is lost when an enemy
bullet hits the player's tank.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player tank | always at screen centre (x = 80, y = 130) |
| Arena | 160 × 210 px viewport |
| Enemies | 4 enemies; respawn every 200 frames |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/battle_zone-v0")
play("atari/battle_zone-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Forward |
| `↓` / `S` | Back |
| `←` / `A` | Rotate left |
| `→` / `D` | Rotate right |
| `Esc` / close window | Quit |
