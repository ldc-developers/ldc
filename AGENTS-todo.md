# LDC — in-tree queue

**Thesis:** Widen the green cell: more lit files, then decide whether the DMD suite or runtime test-runners are the next grain.

**Current state:** Stage-1 Windows cell is green through defaultlibs, a linked `hello.exe`, and `codegen/align.d`. Debug/shared libs and full ctest not run.

**First blocked question:** Is the next pass a broader lit slice (`codegen/` or `driver/`) or `ninja ldc2-unittest`?

## Log

| date | what |
|---|---|
| 2026-08-13 | runtime-adapt: generate this checkout / one tag; `--clean`; AGENTS + architecture tracked. |
| 2026-08-12 | B0–B5 surface authored (untracked). |
| 2026-08-12 | cmake PASS; ninja ldc2 PASS with `/MT`; `ldc2 -c` PASS. |
| 2026-08-12 | `ninja druntime-ldc phobos2-ldc` PASS; `hello.exe` exit 0; lit `codegen/align.d` PASS. |
