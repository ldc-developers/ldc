LDC – Utils
===============================

The `/utils` directory contains utilities that are used in building LDC (`gen_gccbuiltins.cpp`)
and testing LDC (`not` and `FileCheck`).

`not` is copied from LLVM

`FileCheck` is copied from LLVM, and versioned for each LLVM version that we support (for example, FileCheck-3.9.cpp does not compile with LLVM 3.5).
Older versions of FileCheck contain modifications such that they contain new features/bugfixes but still compile with older LLVM versions.

How `not` and `FileCheck` are used is decribed here: [LDC Lit-based testsuite](http://wiki.dlang.org/?title=LDC_Lit-based_testsuite).
