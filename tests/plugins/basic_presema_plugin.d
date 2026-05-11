// REQUIRES: Plugins

// RUN: split-file %s %t --leading-lines
// RUN: %buildplugin %t/plugin.d -of=%t/plugin%so --buildDir=%t/build
// RUN: %ldc -wi -c -o- --plugin=%t/plugin%so %t/testcase.d 2>&1 | FileCheck %t/testcase.d

//--- plugin.d
import dmd.dmodule : Module;
import dmd.errors;
import dmd.location;

extern(C) void runPreSemanticAnalysis(Module m) {
    if (m.md) {
        warning(m.md.loc, "Pre-sema hook fired!");
    }
}

//--- testcase.d
// CHECK: testcase.d([[@LINE+1]]): Warning: Pre-sema hook fired!
module testcase;
int testfunction(int i) {
    return i * 2;
}
