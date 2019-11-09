// RUN: %ldc -output-ll -of=%t.ll %s && FileCheck %s < %t.ll

import ldc.attributes: assumeUsed;

union U
{
    ubyte a;
    uint b;
}

// CHECK: @{{.*}}_D6gh32211uSQk1U{{.*}} = global { i32 } { i32 12345 }
// CHECK: @llvm.used = appending global
// CHECK-SAME: _D6gh32211uSQk1U
@assumeUsed __gshared U u = { b: 12345 };
