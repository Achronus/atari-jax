# Name This Game

> Game ID: `"atari/name_this_game-v0"`

Underwater shooter: a scuba diver collects treasures while an octopus and
a shark descend from above.  Shoot the octopus' tentacles and the shark
before they reach the player.

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
| `4` | RIGHT + FIRE |
| `5` | LEFT + FIRE |

## Reward

Points are awarded for shooting tentacles, hitting the shark, and collecting
treasure.

| Event | Points |
| --- | --- |
| Tentacle shot | +100 |
| Shark shot | +200 |
| Treasure collected | +50 |

## Episode End

The episode ends when all lives are lost.  A life is lost when a tentacle
reaches the player or the shark touches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Player | y = 160, x ∈ [5, 147] |
| Octopus body | y = 40, x ∈ [68, 92] |
| Tentacles (4) | descend from y = 50; x = 30, 60, 100, 130 |
| Shark | y = 60, patrols x ∈ [5, 145] |
| Treasures (3) | x = 40, 80, 120 at player level |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/name_this_game-v0")
play("atari/name_this_game-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `→` / `D` | Move right |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
