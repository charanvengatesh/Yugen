# Death

What happens when the bar reaches zero — what the build does today, and the
staged plan for what it should do instead.

The suffocation work that prompted this is already in: `Player::update_suffocation`
in `crates/yugen-core/src/entities/player.rs`. It matters here because it is the
first death in the game that arrives *slowly and locally* — you can see it
coming, you know exactly where you are standing when it lands, and a corpse
dropped at that spot would be findable. Lava and a spear through the chest are
not like that. A death model that only ever had to answer "you fell in the
lava, start again" now has a case it can do better by.

## Where it stands

Four things, and between them they are the whole of it:

| What | Where |
| --- | --- |
| Out of health | `Player::dead()` — `entities/player.rs` |
| Death ends the run | `death_ends_the_run` — `yugen-render/src/glue.rs` |
| Confirm restarts it | `confirm_advances_the_scene` — same file |
| The restart itself | `start_a_run`, on `OnEnter(Scene::Playing)` |

So: health hits zero, the scene machine goes to `GameOver`, the card is drawn,
Enter or Space returns to `Playing`, and `OnEnter` fires `start_a_run` — which
**regenerates the world from the seed** and refills the pack. `Player::reset`
puts the body back at the worldgen spawn and restores full health;
`Inventory::clear` empties every slot *and* the worn armour, on the stated
grounds that a body keeping its armour across a death would carry the one thing
the death was supposed to cost.

That is a roguelike restart, and it is a coherent thing to be. It is also
inherited: it is what the TypeScript's `updateGameOver` did, and the port kept
it faithfully rather than deciding it.

## The problem with it

**The excavation dies with you.** This is a falling-sand game whose entire
verb set is *dig, collapse, rebuild*. An hour of tunnelling is the run's real
artefact, and death deletes it and hands back the same seed to do again. The
world being regenerated rather than repaired is what makes the loss total —
there is no shaft to walk back down.

**Death has one cost, paid instantly.** You lose the pack, but you also lose the
map, the distance, and the state — so there is nothing to *go back for*. A death
you can respond to is a better death than a death you can only restart from.

**Nothing tells you what killed you.** The card has one action and no cause.
With suffocation in, that gap gets worse: choking under a slump is exactly the
death a player will want explained, because the fix is a behaviour change
(don't stand under a loaded dune) rather than bad luck.

## The model to move to

**The world persists. The body respawns. The pack stays where it fell.**

Concretely: dying drops everything you were carrying as a stack at the death
site, returns the body to a respawn point at full health, and leaves the world
— every tunnel, every rearranged dune — exactly as you left it. Getting your
kit back is a run *into* the place that killed you, against a clock.

Three reasons this one and not another:

- **The retrieval run is the game already.** `WorldItems` exists, drops
  physics-settle onto terrain, and stacks already expire on a `LIFETIME` of 180
  seconds. A corpse is a bag of drops at a coordinate. The mechanic is
  implemented; nothing here invents a system.
- **It gives the persistent world a reason to exist.** Keeping the excavation is
  only worth the save-format work if something makes you return to a specific
  spot in it. The corpse is that.
- **It keeps the cost real.** `Inventory::clear`'s argument survives intact —
  you still lose the pack at the moment of death. You are merely given a way to
  earn it back, and the 180-second lifetime means you often will not.

### Staged

**1. Split "restart" from "respawn".** The one blocking change, and everything
else waits on it. `start_a_run` currently does both jobs in one system on
`OnEnter(Scene::Playing)`, and it rebuilds the world unconditionally because
there was never a path that wanted the world kept. Split it: a *restart* (from
the menu, a new seed) rebuilds; a *respawn* (from `GameOver`) resets the body
and touches nothing else.

Read `crate::scenes` first. It documents the trap this sits on — Bevy re-fires
`OnEnter` when a state is `set` to the value it already holds — and
`start_a_run`'s own comment explains why it deliberately runs on *every* entry
to `Playing` rather than special-casing the first. Splitting it means taking on
the case that comment was avoiding, so the new seam has to say which entry it is
reacting to rather than inferring it.

**2. Drop the pack at the death site.** In `death_ends_the_run`, before the
transition: capture the body's `(x, y)`, walk the inventory into a `DropBag`
(`items/drops.rs` — a count per `ItemCode`, which is exactly the shape needed),
hand it to `WorldItems::spawn_bag(bag, x, y)`, then `Inventory::clear()`. Worn
armour goes into the bag too, on the same argument that currently sends it to
nothing.

One decision to make here: whether the corpse gets the standard 180-second
`LIFETIME` or its own longer one. 180 seconds is tuned for a stack of gravel
knocked loose while mining, not for a run back across the map you just died
crossing. It probably wants its own number, and that number is the difficulty
dial for the whole model.

**3. A respawn point that is not the worldgen spawn.** Today `Player::reset`
goes to the `SpawnPoint` worldgen chose, held on the player as two floats. Make
it settable, default it to that spawn, and let something in the world — a bed, a
marker, a fire — move it. Once the world persists, a fixed origin spawn means
every death is also a long walk, which taxes exploring the far edge of the map
more than it taxes dying.

Persisting it means a save-format change (`sim/save.rs` already stores the
body's position and health).

**4. Say what killed you.** Track the last damage source on the player — hazard
tag, mob id, or suffocation — and put it on the death card
(`yugen-render/src/ui/mod.rs`) with the death coordinates. The coordinates are
not flavour once step 2 lands: they are how you find the bag.

**5. Only then, the toggle.** Once death is recoverable, "permadeath" becomes a
mode worth offering rather than the only behaviour available. It is the current
behaviour, kept, and it belongs in the settings the menus already persist.

## Open

- **Does a death write the save?** If the world persists across death but a
  quit-without-saving rewinds it, players will quit to undo deaths. That is a
  save-cadence decision, not a death decision, but death is where it starts to
  bite.
- **Does the world keep running during `GameOver`?** Today it cannot matter,
  since the world is discarded. With it kept, the automata being paused or not
  behind the card decides whether a dune goes on slumping over your corpse.
  Worth deciding deliberately, because it is a genuinely good scene if it does.
- **Corpse in solid rock.** Dying by suffocation drops the bag *inside* the
  matter that killed you. `WorldItems::spawn` needs checking against that case:
  either the stack settles out to the nearest free cell, or the drop is placed
  at the last free position the body occupied.
