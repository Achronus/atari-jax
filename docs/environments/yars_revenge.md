# Yars' Revenge

> Game ID: `"atari/yars_revenge-v0"`

As a giant fly, nibble through the Qotile shield to create a gap, then fire the Zorlon Cannon through the gap to destroy the Qotile. The Qotile shoots a swirl that must be dodged; a neutral zone in the centre of the screen provides temporary protection.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(7)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE (nibble shield / fire cannon) |
| `2` | UP |
| `3` | RIGHT |
| `4` | DOWN |
| `5` | LEFT |
| `6` | FIRE+UP |

## Reward

| Event | Reward |
| --- | --- |
| Shield block nibbled | `+69` |
| Qotile destroyed (cannon hit) | `+6000` |

The Zorlon Cannon becomes available after nibbling 4 or more shield blocks. On a Qotile kill, the shield resets and the nibble counter resets to zero.

## Episode End

The episode ends when all lives are lost. A life is lost when the player is hit by the Qotile's swirl projectile (the neutral zone at x ∈ [60, 80] provides immunity).

## Lives

The player starts with 3 lives.

## Screen Geometry

| Element | Position |
| --- | --- |
| Neutral zone | x ∈ [60, 80] (no firing; full protection) |
| Shield column | x = 120 |
| Shield blocks | 16 blocks, y ∈ [30, 185] |
| Qotile | x = 148, tracks player y |
| Player x range | x ∈ [5, 116] |
| Player y range | y ∈ [25, 185] |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/yars_revenge-v0")
play("atari/yars_revenge-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | FIRE (nibble / cannon) |
| `↑` / `W` | UP |
| `→` / `D` | RIGHT |
| `↓` / `S` | DOWN |
| `←` / `A` | LEFT |
| `Esc` / close window | Quit |
