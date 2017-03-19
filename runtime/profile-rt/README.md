LDC – profile-rt
===============================

`profile-rt` is the runtime library for writing profiling instrumentation
files. It is linked with instrumented executables (`-fprofile-instr-generate`).

`profile-rt` consists of two parts: a part from LLVM (C/C++,
`profile-rt/profile-rt-3*`), and a D interface to the lib (`profile-rt/d`).
See the source files and LDC's LICENSE file for licensing details.

The sources in `profile-rt/profile-rt-3*` are exact copies of the `lib/profile`
part of LLVM's `compiler-rt` project (`compiler-rt/lib/profile/*`) and its
version has to be exactly in-sync with the LLVM version used to build LDC.
Because of this, we carry a version of compiler-rt/lib/profile for each LLVM
version supported for PGO.

LLVM's llvm-profdata tool of the corresponding LLVM version (!) can interpret
the raw profile data file output by profile-rt, and can convert it into a
stable format that can be interpreted by future LLVM version profiling data
readers (2nd compile pass).
See `ldc-profdata` in the `tools` directory.

The "d" folder contains the D bindings and helper functions to interface with
profile-rt. The code in the "d" folder should be compatible with all supported
LLVM versions.
