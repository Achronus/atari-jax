# Boxing

> ALE name: `boxing` · Game ID: `11`

Trade punches with a CPU opponent in a two-minute boxing match, scoring points for each landed punch and aiming for a knockout.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score differential (player punches landed minus opponent punches landed) per step. Each landed punch scores one point; the reward ranges from -1 to +1 per step. A knockout is encoded as a score of 100.

## Episode End

The episode ends either on a knockout (when either boxer accumulates 100 landed punches) or when the two-minute fight clock reaches 0:00.

## Lives

No lives system. Boxing is a single match with no lives concept.
