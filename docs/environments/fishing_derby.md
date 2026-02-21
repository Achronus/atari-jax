# Fishing Derby

> ALE name: `fishing_derby` · Game ID: `20`

Compete against a CPU opponent to catch the most valuable fish, casting your line into a pond while a shark patrols the depths.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score differential (player score minus opponent score) per step. Points are earned for each fish caught; the reward is positive when the player catches a fish and negative when the opponent catches one.

## Episode End

The episode ends when either player's score reaches 99, detected when the raw score byte equals `0x99`.

## Lives

No lives system. Fishing Derby ends when a player reaches a score of 99 rather than through life loss.
