# Q*bert

> ALE name: `qbert` · Game ID: `39`

Hop across a pyramid of cubes to change each one to the target colour while avoiding Coily the snake and other enemies.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for hopping on each cube and for completing a level. On the terminal step the reward is forced to zero because the score resets before it can be read.

## Episode End

The episode ends when the lives byte reaches the game-over sentinel value (`0xFE`), representing a signed value of -2 after starting at +2.

## Lives

The player starts with 4 lives encoded as a signed byte starting at 2 (displayed lives = RAM value + 2, clamped to zero). A life is lost by falling off the pyramid or being caught by an enemy.
