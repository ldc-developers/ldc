// REQUIRES: Plugins

// For some reason this test fails with missing symbol linking issues (or crash) with macOS on Intel x86 (but not for all CI testers...)
// UNSUPPORTED: Darwin && host_X86

// RUN: split-file %s %t --leading-lines
// RUN: %buildplugin %t/plugin.d -of=%t/plugin%so --buildDir=%t/build
// RUN: %ldc -wi -c -o- --plugin=%t/plugin%so %t/testcase.d 2>&1 | FileCheck %t/testcase.d

//--- plugin.d
import dmd.dmodule : Module;
import dmd.errors;
import dmd.location;

extern(C) void runSemanticAnalysis(Module m) {
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
