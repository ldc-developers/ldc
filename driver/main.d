//===-- driver/main.d - General LLVM codegen helpers ----------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for driver/main.cpp
//
//===----------------------------------------------------------------------===//

module driver.main;

// In driver/main.cpp
extern(C++) int cppmain();

/+ We use this manual D main for druntime initialization via a manual
 + _d_run_main() call in the C main() in driver/main.cpp.
 +/
extern(C) int _Dmain(string[])
{
    return cppmain();
}
