# James Bond 007

> Game ID: `"atari/jamesbond-v0"`

James Bond must navigate between three zones — air, sea, and land —
shooting enemies in each.  Enemies approach from the right; Bond moves
left and right within his zone and switches between zones to intercept.

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
| `2` | UP — switch to air zone |
| `3` | RIGHT |
| `4` | DOWN — switch to sea zone |
| `5` | LEFT |
| `6` | UP + FIRE |
| `7` | DOWN + FIRE |

## Reward

Points are awarded for each enemy destroyed, scaled by zone.

| Event | Points |
| --- | --- |
| Air enemy destroyed | +100 |
| Sea enemy destroyed | +200 |
| Land enemy destroyed | +300 |

## Episode End

The episode ends when all lives are lost.  A life is lost when an enemy
reaches Bond's position in the same zone.

## Lives

The player starts with 5 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Air zone | y = 50 (centre) |
| Sea zone | y = 110 (centre) |
| Land zone | y = 160 (centre) |
| Player (Bond) | x ∈ [5, 60], starts at x = 20 |
| Enemies | approach from x = 165, move left |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/jamesbond-v0")
play("atari/jamesbond-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Fire |
| `↑` / `W` | Switch to air zone |
| `→` / `D` | Move right |
| `↓` / `S` | Switch to sea zone |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
