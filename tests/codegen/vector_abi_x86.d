// Makes sure an optimized trivial function taking and returning a vector
// takes and returns it directly in XMM0, with no memory indirections.

// REQUIRES: host_X86

// RUN: %ldc -O -output-s -of=%t.s %s && FileCheck %s < %t.s

import core.simd;

// CHECK: _D14vector_abi_x863foo
int4 foo(int4 param)
{
    // CHECK-NOT: mov
    // CHECK: paddd
    // CHECK-SAME: %xmm0
    return param + 3;
    // CHECK-NOT: mov
    // CHECK: ret
}
