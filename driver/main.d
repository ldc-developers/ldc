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

extern (C++) void disableGC()
{
	import core.memory;
	GC.disable();
}

extern (C++) void writeModuleDependencyFile()
{
    if (global.params.moduleDepsFile !is null)
    {
        auto deps = File(global.params.moduleDepsFile);
        OutBuffer *ob = global.params.moduleDeps;
        deps.setbuffer(cast(void*)ob.data, ob.offset);
        deps.write();
    }
}