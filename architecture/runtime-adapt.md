# runtime-adapt

Host dub package (`tools/runtime-adapt`), not installed with LDC.

- No args: help.
- `--generate` emits this checkout into `--output` (default
  `tools/runtime-adapt/.work/generated`, paths relative to the LDC root).
- Stock = this `runtime/` minus `ldc/`. `ldc/*` = parse `dmd/` `gen/`
  `driver/`. Goal bodies are never copied.
- At most one `--reference TAG` (cached under gitignored `workspace/` or
  `.work/`). `--all-versions` / `--range` only prefetch. `--clean` drops
  caches.
- Constraints are inferred from `consecutiveTags` in `source/versions.d`
  (one closed interval per minor).

Green: `dub test --root=tools/runtime-adapt --compiler=ldc2`.
Extend: `tools/runtime-adapt/EXTENDING.md`.
