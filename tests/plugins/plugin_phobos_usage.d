// Test usage of Phobos (used to result in dlopen missing symbols, because LDC itself did not have that Phobos symbol)

// REQUIRES: Plugins

// RUN: split-file %s %t --leading-lines
// RUN: %buildplugin %t/plugin.d -of=%t/plugin%so --buildDir=%t/build
// RUN: %ldc -wi -c -o- --plugin=%t/plugin%so %t/testcase.d 2>&1 | FileCheck %t/testcase.d

//--- plugin.d
import dmd.dmodule : Module;
import dmd.errors;
import dmd.location;

import std.experimental.allocator.mallocator;

extern(C) void runSemanticAnalysis(Module m) {
    auto buffer = Mallocator.instance.allocate(1024 * 1024 * 4);
    scope(exit) Mallocator.instance.deallocate(buffer);

    if (m.md) {
        warning(m.md.loc, "It works!");
    }
}

//--- testcase.d
// CHECK: testcase.d([[@LINE+1]]): Warning: It works!
module testcase;
int testfunction(int i) {
    return i * 2;
}
