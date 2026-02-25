# Pitfall!

> Game ID: `"atari/pitfall-v0"`

Harry must collect 32 treasures scattered across jungle screens within
20 minutes while avoiding hazards: rolling logs, crocodiles, and underground
tar pits.  Navigate above ground or descend to underground tunnels.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(6)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — unused |
| `2` | UP — jump / exit underground |
| `3` | RIGHT — run right / advance screen |
| `4` | DOWN — enter underground |
| `5` | LEFT — run left / reverse screen |

## Reward

Points are awarded for collecting treasures; rolling logs deduct points.

| Event | Points |
| --- | --- |
| Treasure collected | +2000 to +5000 |
| Log collision | −100 |

## Episode End

The episode ends when all lives are lost, all 32 treasures are collected,
or the 20-minute timer expires.  A life is lost when a crocodile with an
open mouth catches the player.

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Ground level | y = 160 |
| Underground level | y = 195 |
| Screen width | 160px (transition at x = 10 / x = 150) |
| Treasure | centred at x = 80 per screen, y = ground level |
| Rolling logs (3) | patrol x ∈ [0, 160] at ground level |
| Crocodiles (2) | stationary at x = 60, 110 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/pitfall-v0")
play("atari/pitfall-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `↑` / `W` | Jump / exit underground |
| `→` / `D` | Move right |
| `↓` / `S` | Enter underground |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
