# Ms. Pac-Man

> ALE name: `ms_pacman` · Game ID: `32`

Guide Ms. Pac-Man through a series of mazes eating dots and power pellets while evading four coloured ghosts.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for eating dots (10 each), power pellets (50 each), ghosts while energised (200–1600 each), and fruit bonuses. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives nibble reaches zero and the death timer reaches its sentinel value (`0x53`), confirming the final life-loss animation has played.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost each time Ms. Pac-Man is caught by a ghost while not energised.
