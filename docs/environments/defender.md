# Defender

> ALE name: `defender` · Game ID: `16`

Pilot a spacecraft in a side-scrolling world to defend humanoids on the ground from waves of alien ships that descend to abduct them.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting enemy ships, and bonus points are awarded for rescuing abducted humanoids. Score digits are stored one per RAM byte in the lower nibble, with a sentinel value of `0xA` treated as zero.

## Episode End

The episode ends when all lives are exhausted, detected when the lives count transitions from a positive value to zero.

## Lives

The player starts with several lives stored directly in a RAM byte. A life is lost when the player's ship is destroyed by enemy fire or collision.
