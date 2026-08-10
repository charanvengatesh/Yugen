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

**1. Split "restart" from "respawn". — DONE.** The one blocking change, and
everything else waited on it. `start_a_run` did both jobs in one system on
`OnEnter(Scene::Playing)` and rebuilt the world unconditionally, because there
was never a path that wanted the world kept.

The seam is `scenes::Entry`, a resource the transition sets: a *restart* (menu,
world picker, `--play`, every capture harness) rebuilds; a *respawn* (from
`GameOver`) resets the body and drops arrows in flight, and touches nothing
else. It is DECLARED rather than inferred from the scene being left, for the
reason `crate::scenes` documents — Bevy re-fires `OnEnter` when a state is `set`
to the value it already holds, so there is a legal entry whose previous scene is
`Playing`, and a new entry point added later would otherwise inherit whichever
branch its predecessor happened to land in.

`Restart` is the default, which is the destructive reading: everything entering
`Playing` without mentioning `Entry` gets exactly the behaviour it had before
the split. `start_a_run` consumes the intent, so it belongs to one transition
and cannot leak into the next.

**2. Drop the pack at the death site. — DONE.** `death_ends_the_run` walks the
pack into a `DropBag`, hands it to `GroundItems::spawn_bag` at the body's
coordinates, then clears the pack. Worn armour goes in too. The bag is spawned
BEFORE the clear so a panic between the two cannot lose the pack into neither
place, and the save is requested after both, so what reaches disk is the world
as the death left it.

Two things it deliberately does not do, both still open:

- **The corpse takes the standard 180-second `LIFETIME`.** That number is tuned
  for a stack of gravel knocked loose while mining, not for a run back across
  the map you just died crossing. It probably wants its own, and that number is
  the difficulty dial for the whole model — so it is left to be turned
  deliberately rather than picked in passing.
- **`RunState` still does not persist world items.** A corpse survives a
  respawn, because the world now stays in memory, but not a quit and reload.
  That is this document's remaining save work.

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
