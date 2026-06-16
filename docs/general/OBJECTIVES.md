# Placement Objectives

These are the structural objectives we use to guide macro placement.

## The Five Objectives

- Wirelength: keep connected macros close so wires are short. Shorter wires mean less resistance, less signal delay, and less crosstalk risk.
- Density uniformity: do not cluster macros in one corner. Uneven density creates hot spots where voltage sags and heat builds up, which hurts performance and aging.
- I/O keepout zones: keep macros away from the chip periphery. That space needs to stay clear for decoupling capacitors, ESD protection, and signal buffers.
- Grid alignment: nudge macros to snap to a virtual grid. This makes clock tree distribution more even and reduces extra buffering, skew, and latency.
- Notch avoidance: do not let macros get too close without enough routing gap. Tight gaps force wires into narrow corridors, which raises current density, increases electromigration risk, and traps crosstalk aggressors.

## Why This Works

None of these five objectives directly measures IR drop, electromigration, or clock skew. Each one removes a structural condition that tends to cause those failures downstream.

The intent is that, by the time routing happens, the placement already has the structure the router needs. That does not guarantee perfect post-route numbers, but it makes the routing problem cleaner so the downstream measurements tend to improve even without post-route data in the training loop.
