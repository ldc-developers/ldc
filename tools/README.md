LDC – Tools
===============================

The `/tools` directory contains user tools that accompany LDC and that should be part of LDC packages.

`ldc-prune-cache` helps keeping the size of LDC's object file cache (`-cache`) in check. See [the original PR](https://github.com/ldc-developers/ldc/pull/1753) for more details.

`ldc-profdata` converts raw profiling data to a profile data format that can be used by LDC. The source is copied from LLVM (`llvm-profdata`), and is versioned for each LLVM version that we support because the version has to match exactly with LDC's LLVM version.

`timetrace2txt` converts the .timetrace output of `--ftime-trace` (which is in [Chromium's trace event JSON format](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/)) to a text file that is easier for humans to read.
