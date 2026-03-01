# Boxing

> Game ID: `"atari/boxing-v0"`

Trade punches with a CPU opponent in a boxing ring.  Score points for each
punch that lands.  First to 100 landed punches wins by knockout; otherwise
the higher score after the time limit wins.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

### Action table

| Index | Meaning |
| --- | --- |
| `0` | NOOP |
| `1` | FIRE — punch |
| `2` | UP |
| `3` | RIGHT |
| `4` | LEFT |
| `5` | DOWN |
| `6` | UPRIGHT |
| `7` | UPLEFT |
| `8` | DOWNRIGHT |
| `9` | DOWNLEFT |
| `10` | UPFIRE |
| `11` | RIGHTFIRE |
| `12` | LEFTFIRE |
| `13` | DOWNFIRE |
| `14` | UPRIGHTFIRE |
| `15` | UPLEFTFIRE |
| `16` | DOWNRIGHTFIRE |
| `17` | DOWNLEFTFIRE |

## Reward

The reward per step is the change in score differential: `+1` for each
player punch that lands, `−1` for each CPU punch that lands.

## Episode End

The episode ends on a knockout (either boxer reaches 100 landed punches) or
when the time limit is reached.

## Lives

No lives system.  `lives` is always `0`.

## Screen Geometry

| Element | Position |
| --- | --- |
| Boxing ring | x ∈ [8, 152), y ∈ [50, 185) |
| Player start | x = 64, y = 150 |
| CPU start | x = 88, y = 70 |

## Interactive Play

```python
from atarax.utils.render import play

play("atari/boxing-v0")
play("atari/boxing-v0", scale=2, fps=30)
```

### Keyboard controls

| Key | Action |
| --- | --- |
| `Space` | Punch |
| `↑` / `W` | Move up |
| `→` / `D` | Move right |
| `↓` / `S` | Move down |
| `←` / `A` | Move left |
| `Esc` / close window | Quit |
