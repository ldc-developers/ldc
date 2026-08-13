# Extending runtime-adapt when LDC changes

LDC’s own history is the workflow. Commits are small, paired, and named
after the *compiler* change first:

| Example commit | What landed |
|---|---|
| `2e3fcce194` Add support for LLVM 24 to `intrinsics.di` | one-line version ladder |
| `613fe94dc9` Add llvm convert vector to `ldc.intrinsics` | `gen/pragma.cpp` + `gen/tocall.cpp` + `ldc/intrinsics.di` + `tests/codegen/` |
| `ac64c3ccad` Add `llvm_is_fpclass` | `ldc/intrinsics.di` only |
| `0c0364b644` Add predefined dcompute versions | `driver/main.cpp` only |
| `28933e1577` Add LDC-specific DRuntime changes | new `ldc/eh_wasm.d` + CMake |
| `21ba72d91f` Add support for Wasm EH | `gen/trycatchfinally.cpp` then runtime (later deleted `eh_wasm.d`) |
| `9ae29dc402` druntime: Treat WASI as Posix | stock file, `version (WASI)` / `Posix` |

The tool is laid out the same way: **one emitter file per
`runtime/druntime/src/ldc/` file**, registered in one table.

## Where to edit

```
LDC commit touches…                    You edit…
─────────────────────────────────────  ──────────────────────────────────────
gen/pragma.cpp  (new pragma / llvm.*)  compilerparse already scans it;
                                       signatures → source/ldcmods/intrinsics.d
gen/uda.cpp     (new @uda)             source/ldcmods/attributes.d
gen/runtime.cpp (new _d_* hook)        compilerparse.hookHome + adapt.d
                                       (only if stock lacks it and goal has it)
driver/main.cpp (new version ident)    compilerparse already scans
                                       addPredefinedGlobalIdent — do not invent
                                       DigitalMars (LDC does not predefine it)
new runtime/druntime/src/ldc/foo.d     source/ldcmods/foo.d
                                       + one row in source/ldcmods/package.d
                                       + name in compilerparse.isLdcRuntimeModule
stock file LDC patched (druntime: …)   FILE-CMP class missed-ldc
                                       → principle / splice in adapt.d
tests/codegen/new_thing.d              optional: tests/ under this package
```

## Add a new `ldc/` file (the `eh_wasm.d` pattern)

1. Create `source/ldcmods/<name>.d` with `wantX` + `renderX`.
2. Append an `LdcEmitter` row in `source/ldcmods/package.d`
   (`rel`, `name`, compiler `locus`, `want`, `render`).
3. Add the bare name to `isLdcRuntimeModule` if `ldc.<name>` appears in
   `gen/` / `driver/`.
Generate always from **this checkout**. At most one `--reference TAG`.
Ranges (`--all-versions`, `--range`) only prefetch gitignored caches.

4. From the LDC repo root:

   ```text
   dub test --root=tools/runtime-adapt --compiler=ldc2
   dub run  --root=tools/runtime-adapt --compiler=ldc2 -- --reference v1.36.0 --iterate
   ```

New **minor tag** on the ladder: append it to `consecutiveTags` in
`source/versions.d`. `--all-versions` keeps a **12-tag window** ending
at the new tip (oldest tag drops off). Prefetch that window with
`--prefetch-refs`.

5. Read `.work/generated/FILE-CMP.md`. A new `ldc-emit` row is expected.
   A `missing` row means the registry `want` predicate is false.
   A `missed-ldc` row on a *stock* file means adapt.d needs a principle.

## Diagnose before inventing

`--iterate` writes two reports next to the product:

- **FILE-CMP.md** — full-file, three-way (stock / generated / goal).
  This is the review surface, same as reading an LDC PR diff.
- **AST-DIFF.md** — missing files, struct fields, `LDC_*` pragmas on
  `ldc/*` that are not text-equal.

Do not copy goal bodies. Close a gap by teaching `compilerparse` or the
matching `ldcmods/<name>.d` emitter.

## What not to do

- Do not assign `version = DigitalMars`. LDC predefines `LDC` only
  (`dmd/target.d` is `version (IN_LLVM)` skipped). That flip would make
  `std.compiler.Vendor` digitalMars and enable DMD-only workarounds.
- Do not append `_d_*` stubs that already exist in another stock or goal
  file (`rt/dwarfeh.d`, `core/lifetime.d`, `ldc/eh_msvc.d`).
- Do not add a second list of ldc module names. The registry in
  `ldcmods/package.d` plus `isLdcRuntimeModule` are the two lists; keep
  them in lockstep.
