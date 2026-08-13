# Development (carried by value)

A pass names one thesis, opens what `AGENTS.md` maps, changes the tree in this project’s idiom, updates `architecture/runtime-adapt.md` if the tool changed, and runs the recorded green command or a stated subset.

Finished means: the compiler still configures; any interface change (`ldc2` flags, defaultlib names, IR ABI) is stated; the cell that was run is written down. Large work is sequenced so each step has its own green.

A **runtime-adapt** pass is finished when `dub test --root=tools/runtime-adapt --compiler=ldc2` is green and, if emit changed, generate from this checkout has been read via `.work/generated/FILE-CMP.md`. Do not emit a tag range; `--clean` drops gitignored caches.
