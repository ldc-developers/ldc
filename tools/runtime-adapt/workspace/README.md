# workspace/ — reference trees only

This directory may hold **official stock** druntime/phobos and **LDC
checkouts used as a guide**. LDC’s own `runtime/` is never an emit source.

| Path | Role |
|---|---|
| `stock/v1.N.0` | Optional official/stock druntime/phobos (preferred reference) |
| `refs/v1.30.0` … `refs/v1.42.0` | LDC checkout at that tag: compiler (`driver/`, `gen/`) to emit `ldc/*.di`, and runtime as the **goal** only |
| `overlays/` | Legacy overlay experiments (not the generate product) |
| `.tmp/` | Scratch from `--consecutive` |

Create or refresh refs:

```text
cd tools/runtime-adapt
dub run --compiler=ldc2 -- --sync-workspace
# or:
powershell -File workspace/sync-refs.ps1
```

`resolve.materializeReference` prefers `workspace/refs/<tag>` when present,
otherwise `git archive` into `.work/ref-<tag>/`.

`refs/`, `stock/`, `overlays/`, and `.tmp/` are gitignored. This README
and the `*.ps1` scripts are tracked. `--clean` removes the caches.
