# Atlantis

> Game ID: `"atari/atlantis-v0"`

Defend the city of Atlantis from waves of alien ships using three gun
emplacements.  Ships descend in passes; each one that reaches the city
destroys a section.  The episode ends when all six city sections are
destroyed.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(4)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE LEFT — diagonal left shot |
| `2` | FIRE CENTRE — vertical shot |
| `3` | FIRE RIGHT — diagonal right shot |

## Reward

Each alien ship destroyed earns 250 points.

| Event | Points |
| --- | --- |
| Alien destroyed | +250 |

## Episode End

The episode ends when all six city sections have been destroyed.

## Lives

Atlantis uses city sections rather than traditional lives.  The game starts
with all 6 sections intact; `lives` tracks the number of surviving sections.
The episode ends when `lives` reaches `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| City sections | 6 sections at y = 175 |
| Left cannon | x = 14, y = 170 |
| Centre cannon | x = 80, y = 170 |
| Right cannon | x = 146, y = 170 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/atlantis-v0")
play("atari/atlantis-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `←` / `A` | Fire left cannon |
| `Space` | Fire centre cannon |
| `→` / `D` | Fire right cannon |
| `Esc` / close window | Quit |
