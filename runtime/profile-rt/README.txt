"profile-rt" is the runtime library for writing profiling instrumentation files.
This lib is part of LLVM's compiler-rt project (compiler-rt/lib/profile/*) and its version has to be in-sync with LLVM.
LLVM's llvm-profdata tool of the corresponding LLVM version (!) can interpret the raw profile data file,
and can convert it into a stable format that can be interpreted by future LLVM version profiling data readers (2nd compile pass).

Because of this, we carry a version of compiler-rt/lib/profile for each LLVM version supported for PGO.

The "d" folder contains the D bindings and helper functions to interface with profile-rt. The code in the "d" folder should be
compatible with all supported LLVM versions.
