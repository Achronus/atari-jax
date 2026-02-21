# Venture

> ALE name: `venture` · Game ID: `52`

Explore dungeon rooms in a top-down view, collecting treasures and battling monsters with a bow and arrow while avoiding the hallway guardians.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for collecting treasure items within dungeon rooms. Score values are multiples of 100. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when all three death-sequence conditions simultaneously hold: the lives byte reaches zero, the death audio byte equals `0xFF`, and a death-sequence bit flag is set.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost when the player is caught by a dungeon monster or a hallway guardian.
