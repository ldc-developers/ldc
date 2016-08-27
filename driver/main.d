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

import ddmd.globals;
import ddmd.root.file;
import ddmd.root.outbuffer;

// In driver/main.cpp
extern(C++) int cppmain(int argc, char **argv);

/+ Having a main() in D-source solves a few issues with building/linking with
 + DMD on Windows, with the extra benefit of implicitly initializing the D runtime.
 +/
int main()
{
    // For now, even just the frontend does not work with GC enabled, so we need
    // to disable it entirely.
    import core.memory;
    GC.disable();

    import core.runtime;
    auto args = Runtime.cArgs();
    return cppmain(args.argc, cast(char**)args.argv);
}
