# LDC — Agent Guider (repository root)

> **Scope:** entry point for agents working in this repository. Everything referenced here resolves inside this tree. Architectural notes live in `architecture/`; live state in `AGENTS-todo.md`.

```yaml
pin:              1218a472ae (v1.43.0-beta1-3)
status:           inventoried / advancing
purpose:          LLVM-based D compiler; correctness, bootstrap reproducibility, portability
languages:        D (frontend + parts of driver/gen), C++ (driver + gen + ir), CMake
build_spine:      CMake + Ninja; find_package(LLVM 18+); host ldmd2
bootstrap:        prior LDC/DMD required; LLVM is a prebuilt foreign package
green_command:    cmake -G Ninja -S . -B <build> -DCMAKE_BUILD_TYPE=Release -DLLVM_ROOT_DIR=<llvm> -DD_COMPILER=<host>/ldmd2.exe && ninja -C <build> ldc2
green_cell:       windows-x64, VS18/cl /MT, LLVM 22.1.8 (ldc-flavoured), host LDC 1.42.0
green_state:      verified 2026-08-12 — cmake + ninja ldc2 + defaultlibs + hello.exe + lit codegen/align.d
persistence:      tracked on branch tools/runtime-adapt
fingerprint:      { cmake: CMakeLists.txt, entry: driver/main.cpp, frontend: dmd/, harness: tests/ }
riscv_affinity:   latent
```

## 0. Prime directive

The checkout is authoritative. Do not invent LLVM versions or build flags; they live in `CMakeLists.txt` and `.github/`. A change is finished when the recorded green command (or a stated subset) has been run on a named cell. Prefer the form CI already uses. The public interface is `ldc2`/`ldmd2` and the defaultlib names. RISC-V is available through LLVM, not sovereign.

## 1. What carries the logic

| Tree | Role |
|---|---|
| `driver/` | Process, CLI, config, emit, link |
| `dmd/` | DMD frontend (in-tree rewrite, not a submodule) |
| `gen/` | AST → LLVM IR, ABI, optimizer |
| `ir/` | Cached IR objects |
| `runtime/druntime/` | In-tree D runtime (`ldc/` is LDC-specific) |
| `runtime/phobos/` | Submodule standard library |
| `tests/` | lit + DMD suite |
| `tools/runtime-adapt/` | Maintainer: stock → complete LDC runtime (dub, not CMake) |
| `cmake/` | FindLLVM, FindDCompiler, BuildDExecutable |

## 2. Architectural notes

See [`architecture/README.md`](architecture/README.md).

## 3. Navigate by intent

| To find / to change... | Open |
|---|---|
| driver / argument handling | `driver/main.cpp`, `driver/cl_options.cpp` |
| front-end boundary | `dmd/`, `dmd/ldcbindings.d` |
| lowering / emission | `gen/toir.cpp`, `gen/functions.cpp`, `driver/codegenerator.cpp` |
| runtime or support library | `runtime/` |
| `ldc/*` emit / stock → LDC runtime | `tools/runtime-adapt/`, `architecture/runtime-adapt.md` |
| target or platform description | `dmd/target.d`, `gen/target.cpp`, `gen/abi/` |
| test harness / one test | `tests/CMakeLists.txt`, `tests/codegen/` |

## 4. Green command

Compiler: configure + `ninja ldc2` (see CMakeLists.txt). Tool: `dub test --root=tools/runtime-adapt --compiler=ldc2`.

## 5. Invariants

- LLVM ≥ 18 via `llvm-config`; official Windows path is the LDC-flavoured 7z, not the llvm.org app installer.
- Host compiler filename must be `ldmd2` (or `dmd`/`gdmd`).
- Phobos is the only submodule. Frontend is not.
- Defaultlibs: `phobos2-ldc`, `druntime-ldc`.
- Agent notes (`AGENTS*.md`, `architecture/`) are tracked on `tools/runtime-adapt`. Runtime-adapt caches (`.work/`, `workspace/refs/`) stay gitignored; `--clean` removes them.

## 6. Target affinity

`latent`. LLVM RISCV backend may be present. No LDC RISC-V program in this pass.

## 7. Open questions

Generate is this checkout; at most one `--reference` tag. See `architecture/runtime-adapt.md`.

## 8. Refresh log

| date | pin | trigger | what changed |
|---|---|---|---|
| 2026-08-12 | 1218a472ae | first inventory | notes + guider authored |
| 2026-08-13 | 1218a472ae | runtime-adapt | generate = this checkout, one optional tag; caches gitignored; `--clean` |
