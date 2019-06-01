//===-- driver/main.d - D entry point -----------------------------*- D -*-===//
//
//                         LDC – the LLVM D compiler
//
// This file is distributed under the BSD-style LDC license. See the LICENSE
// file for details.
//
//===----------------------------------------------------------------------===//
//
// D entry point for LDC/LDMD, just forwarding to cppmain().
//
//===----------------------------------------------------------------------===//

module driver.main;

// In driver/main.cpp or driver/ldmd.cpp
extern(C++) int cppmain();

/+ We use this manual D main for druntime initialization via a manual
 + _d_run_main() call in the C main() in driver/{main,ldmd}.cpp.
 +/
extern(C) int _Dmain(string[])
{
    return cppmain();
}
