# Environments

All 57 Atari 2600 environments from the standard RL benchmark (Mnih et al. 2015),
implemented as pure JAX functions — no ROM loading, no hardware emulation.

Every environment shares the same observation space: `Box(uint8, shape=(210, 160, 3))`.
Action spaces vary per game; each page lists the exact `Discrete(N)` size and
action table.

## Usage

```python
from atarax.env.make import make

env = make("atari/breakout-v0")           # single environment
env = make("atari/breakout-v0", preset=True)  # DQN preprocessing stack
```

## Game Groups

Standard benchmark subsets used in the RL literature:

| Group | Size | Description |
| --- | --- | --- |
| `"atari5"` | 5 | Breakout, Ms. Pac-Man, Pong, Q*bert, Space Invaders |
| `"atari10"` | 10 | atari5 + Alien, Beam Rider, Enduro, Montezuma's Revenge, Pitfall! |
| `"atari26"` | 26 | The original 26-game evaluation set |
| `"atari57"` | 57 | Full 57-game suite |

```python
from atarax.games.registry import GAME_GROUPS

specs = GAME_GROUPS["atari57"]   # list of EnvSpec
```

## All Environments

| Game | Game ID | Description |
| --- | --- | --- |
| [Alien](alien.md) | `"atari/alien-v0"` | Navigate a space station shooting alien creatures and collecting eggs while avoiding the deadly alien. |
| [Amidar](amidar.md) | `"atari/amidar-v0"` | Paint every cell of a grid by moving along paths while evading enemies. |
| [Assault](assault.md) | `"atari/assault-v0"` | Pilot a turret to destroy waves of attacking alien ships before they reach you. |
| [Asterix](asterix.md) | `"atari/asterix-v0"` | Guide Asterix collecting objects and avoiding enemies in this side-scrolling arcade game. |
| [Asteroids](asteroids.md) | `"atari/asteroids-v0"` | Pilot a spaceship and destroy asteroids and flying saucers while avoiding collisions. |
| [Atlantis](atlantis.md) | `"atari/atlantis-v0"` | Defend the city of Atlantis from alien ships using three gun emplacements. |
| [Bank Heist](bank_heist.md) | `"atari/bank_heist-v0"` | Drive a getaway car through a city robbing banks while evading police. |
| [Battle Zone](battle_zone.md) | `"atari/battle_zone-v0"` | Destroy enemy tanks and saucers in a first-person 3D tank combat arena. |
| [Beam Rider](beam_rider.md) | `"atari/beam_rider-v0"` | Ride a beam across space shooting enemy ships while dodging obstacles. |
| [Berzerk](berzerk.md) | `"atari/berzerk-v0"` | Navigate a maze shooting robots while evading the invincible Evil Otto. |
| [Bowling](bowling.md) | `"atari/bowling-v0"` | Bowl a ten-frame game, aiming for strikes and spares to maximise your score. |
| [Boxing](boxing.md) | `"atari/boxing-v0"` | Trade punches with a CPU opponent in a two-minute boxing match. |
| [Breakout](breakout.md) | `"atari/breakout-v0"` | Use a paddle to bounce a ball and destroy all the bricks in the wall above. |
| [Centipede](centipede.md) | `"atari/centipede-v0"` | Shoot a descending centipede and other insects before they reach the bottom. |
| [Chopper Command](chopper_command.md) | `"atari/chopper_command-v0"` | Pilot a helicopter gunship to protect a truck convoy from enemy aircraft. |
| [Crazy Climber](crazy_climber.md) | `"atari/crazy_climber-v0"` | Scale skyscrapers while dodging falling objects and hostile birds. |
| [Defender](defender.md) | `"atari/defender-v0"` | Defend humanoids from alien abduction in a side-scrolling space shooter. |
| [Demon Attack](demon_attack.md) | `"atari/demon_attack-v0"` | Destroy waves of attacking demons from your cannon on a hostile planet. |
| [Double Dunk](double_dunk.md) | `"atari/double_dunk-v0"` | Play two-on-two basketball and outscore the opposing team. |
| [Enduro](enduro.md) | `"atari/enduro-v0"` | Race through day and night conditions, passing a quota of cars each day to continue. |
| [Fishing Derby](fishing_derby.md) | `"atari/fishing_derby-v0"` | Fish against a CPU opponent; first angler to 99 points wins. |
| [Freeway](freeway.md) | `"atari/freeway-v0"` | Guide a chicken across a busy multi-lane highway within a time limit. |
| [Frostbite](frostbite.md) | `"atari/frostbite-v0"` | Jump across floating ice floes to build an igloo while avoiding Arctic hazards. |
| [Gopher](gopher.md) | `"atari/gopher-v0"` | Protect three carrots from a tunnelling gopher by shooting it and filling its holes. |
| [Gravitar](gravitar.md) | `"atari/gravitar-v0"` | Pilot a spaceship through gravitational hazards, destroying targets and rescuing humanoids. |
| [H.E.R.O.](hero.md) | `"atari/hero-v0"` | Use a jetpack and laser to rescue miners trapped in underground caverns. |
| [Ice Hockey](ice_hockey.md) | `"atari/ice_hockey-v0"` | Play one-on-one ice hockey, scoring goals against the CPU opponent. |
| [James Bond](jamesbond.md) | `"atari/jamesbond-v0"` | Play as Agent 007 piloting a multi-mode vehicle through enemy-filled missions. |
| [Kangaroo](kangaroo.md) | `"atari/kangaroo-v0"` | Guide a mother kangaroo up platforms to rescue her joey from monkeys. |
| [Krull](krull.md) | `"atari/krull-v0"` | Fight through scenes from the film Krull, collecting weapons and defeating enemies. |
| [Kung-Fu Master](kung_fu_master.md) | `"atari/kung_fu_master-v0"` | Battle through five floors of enemies using kicks, punches, and jumps. |
| [Montezuma's Revenge](montezuma_revenge.md) | `"atari/montezuma_revenge-v0"` | Explore an underground pyramid collecting treasures and keys. |
| [Ms. Pac-Man](ms_pacman.md) | `"atari/ms_pacman-v0"` | Guide Ms. Pac-Man through a maze eating dots while evading ghosts. |
| [Name This Game](name_this_game.md) | `"atari/name_this_game-v0"` | Defend a treasure chest from sea creatures as a scuba diver. |
| [Phoenix](phoenix.md) | `"atari/phoenix-v0"` | Shoot waves of alien birds and then destroy the giant mothership. |
| [Pitfall!](pitfall.md) | `"atari/pitfall-v0"` | Swing on vines, jump over obstacles, and collect treasures in a jungle. |
| [Pong](pong.md) | `"atari/pong-v0"` | Volley a ball with paddles; first to 21 points wins. |
| [Pooyan](pooyan.md) | `"atari/pooyan-v0"` | Ride a cable car shooting wolves on balloons to protect your piglet family. |
| [Private Eye](private_eye.md) | `"atari/private_eye-v0"` | Drive through a city as a detective, tracking criminals and recovering stolen items. |
| [Q\*bert](qbert.md) | `"atari/qbert-v0"` | Hop across a pyramid of cubes changing their colour while avoiding enemies. |
| [River Raid](riverraid.md) | `"atari/riverraid-v0"` | Pilot a jet plane down a river, shooting ships and helicopters while managing fuel. |
| [Road Runner](road_runner.md) | `"atari/road_runner-v0"` | Race ahead of Wile E. Coyote collecting birdseed and dodging obstacles. |
| [Robotank](robotank.md) | `"atari/robotank-v0"` | Command a tank to destroy enemy robot squadrons in a first-person simulation. |
| [Seaquest](seaquest.md) | `"atari/seaquest-v0"` | Pilot a submarine rescuing divers and shooting enemies while managing oxygen. |
| [Skiing](skiing.md) | `"atari/skiing-v0"` | Ski downhill through a slalom course as quickly as possible. |
| [Solaris](solaris.md) | `"atari/solaris-v0"` | Pilot a starship across a galactic map, fighting enemies to liberate star systems. |
| [Space Invaders](space_invaders.md) | `"atari/space_invaders-v0"` | Shoot descending alien invaders before they reach the ground. |
| [Star Gunner](star_gunner.md) | `"atari/star_gunner-v0"` | Pilot a star fighter through waves of alien ships. |
| [Tennis](tennis.md) | `"atari/tennis-v0"` | Play a singles tennis match against a CPU opponent. |
| [Time Pilot](time_pilot.md) | `"atari/time_pilot-v0"` | Fly through historical eras destroying enemy planes and rescuing soldiers. |
| [Tutankham](tutankham.md) | `"atari/tutankham-v0"` | Navigate through a tomb shooting enemies and collecting treasures. |
| [Up 'n Down](up_n_down.md) | `"atari/up_n_down-v0"` | Drive along a hilly road, jumping over or destroying other vehicles for points. |
| [Venture](venture.md) | `"atari/venture-v0"` | Explore dungeon rooms collecting treasures while battling monsters. |
| [Video Pinball](video_pinball.md) | `"atari/video_pinball-v0"` | Play a digital pinball table, using flippers to score points. |
| [Wizard of Wor](wizard_of_wor.md) | `"atari/wizard_of_wor-v0"` | Fight through dungeon mazes shooting monsters, including the Wizard of Wor. |
| [Yar's Revenge](yars_revenge.md) | `"atari/yars_revenge-v0"` | Eat through a shield to fire a cannon at the Qotile. |
| [Zaxxon](zaxxon.md) | `"atari/zaxxon-v0"` | Pilot a spacecraft through a fortress in a diagonal-scrolling isometric shooter. |
