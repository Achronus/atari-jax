# Alien

> ALE name: `alien` · Game ID: `0`

Navigate a space station, shooting alien creatures and collecting pulsating eggs while avoiding contact with the deadly alien and its offspring.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by shooting alien creatures, destroying alien eggs, and clearing rooms. Score digits use a custom encoding where each digit is shifted right by 3 bits, and the final score is multiplied by 10.

## Episode End

The episode ends when the player loses their last life. A life is lost by coming into contact with an alien or being caught in an explosion.

## Lives

The player starts with a number of lives tracked in RAM. The episode terminates when lives drop from a positive count to zero.
