// Make sure exported functions can be cross-module inlined without exporting the local function copy.

// RUN: %ldc -O -release -enable-cross-module-inlining -output-ll -of=%t.ll -I%S/inputs %s
// RUN: FileCheck %s < %t.ll

import export2;

// CHECK-NOT: _D7export23fooFZi

// CHECK: define {{.*}}_D26export_crossModuleInlining3barFZi
int bar()
{
    // CHECK-NEXT: ret i32 666
    return foo();
}

// CHECK-NOT: _D7export23fooFZi
