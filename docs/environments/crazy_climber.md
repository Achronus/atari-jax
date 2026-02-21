# Crazy Climber

> ALE name: `crazy_climber` · Game ID: `15`

Scale the faces of skyscrapers using both hands to grip windows, dodging falling objects, hostile birds, and opening windows.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for climbing floors and reaching the helicopter at the top of each building. Score values are multiples of 100. Negative score deltas are clamped to zero to guard against a score-reset glitch.

## Episode End

The episode ends when all lives are exhausted, detected when the lives count transitions from a positive value to zero.

## Lives

The player starts with several lives stored directly in a RAM byte. A life is lost each time the climber falls due to a window closing, a falling object, or a bird attack.
