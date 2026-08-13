# runtime-adapt

Maintainer tool **in this LDC checkout**, on the `tools/runtime-adapt` branch.
Not installed with LDC (`tools/CMakeLists.txt` is unchanged).

Stock druntime/phobos from **this repo** is the input. LDC’s `ldc/*` is
generated from **this repo’s** `driver/` and `gen/`. The goal tree is
never copied. When LDC lands a commit, follow **`EXTENDING.md`**.

## What git tracks

Only the tool. Caches are gitignored and filled on demand.

| Tracked | Untracked (pulled / generated) |
|---|---|
| `source/`, `tests/` | `workspace/refs/`, `workspace/stock/` |
| `README.md`, `EXTENDING.md` | `.work/` (generate + git archives) |
| `dub.sdl`, `.gitignore` | `bin/`, `clones/`, `overlays/` |
| `workspace/README.md`, `workspace/*.ps1` | `dub.selections.json` |

## Compile and test

Always from the **LDC repo root**:

```text
dub build --root=tools/runtime-adapt --compiler=ldc2
dub test  --root=tools/runtime-adapt --compiler=ldc2
dub run   --root=tools/runtime-adapt --compiler=ldc2 -- [args]
```
Constraints are inferred from `consecutiveTags` in `source/versions.d`
(no `constraints.json`).

`dub test` generates from **this checkout** (stock walk + `ldc/*` from
`driver/`/`gen/`). It does not use overlay fixtures or a tag range.

## Generate (explicit; at most one tag)

No arguments prints help. Emit this checkout with `--generate`. `--output`
is under the LDC repo root (relative paths are from that root).

```text
dub run --root=tools/runtime-adapt --compiler=ldc2 --
dub run --root=tools/runtime-adapt --compiler=ldc2 -- --generate
dub run --root=tools/runtime-adapt --compiler=ldc2 -- --generate --output tools/runtime-adapt/.work/generated
dub run --root=tools/runtime-adapt --compiler=ldc2 -- --generate --reference v1.36.0
```

Default `--output` is `tools/runtime-adapt/.work/generated` (gitignored).
Reports land next to the tree: `FILE-CMP.md`, `AST-DIFF.md`.

## Cache a window (no emit)

Ranges only **prefetch** into untracked `workspace/refs/` / `.work/ref-*`:

```text
--prefetch-refs
--all-versions                  # last 12 minors on the ladder
--range v1.40.0..v1.42.0
--clean                         # delete .work/, workspace/{refs,stock}, bin/, clones/
```

## Development flow

1. Edit `source/ldcmods/<name>.d` (or add a registry row). See `EXTENDING.md`.
2. `dub test --root=tools/runtime-adapt --compiler=ldc2`
3. Generate this checkout; read `FILE-CMP.md`.
4. `missed-ldc` → `adapt.d` / `principles.d`. `ldc-emit` → the emitter.

LDC does **not** predefine `DigitalMars`. Stock `version (DigitalMars)`
stays false so `std.compiler.Vendor` is `llvm`.
