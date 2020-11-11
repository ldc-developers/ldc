// Makes sure an optimized trivial function taking and returning a vector
// takes and returns it directly in XMM0, with no memory indirections.

// REQUIRES: host_X86

// RUN: %ldc -O -output-s -m32 -of=%t_32.s %s -mattr=+sse2
// RUN: FileCheck --check-prefix=COMMON %s < %t_32.s

// RUN: %ldc -O -output-s -m64 -of=%t_64.s %s
// RUN: FileCheck --check-prefix=COMMON --check-prefix=X64 %s < %t_64.s

import core.simd;

// COMMON: _D14vector_abi_x863foo
int4 foo(int4 param)
{
    // COMMON-NOT: mov
    // COMMON: paddd
    // COMMON-SAME: %xmm0
    return param + 3;
    // COMMON-NOT: mov
    // COMMON: ret
}

version (X86_64)
{
    struct Int4 { int4 v; }

    // X64: _D14vector_abi_x863barFSQw4Int4ZQj
    Int4 bar(Int4 param)
    {
        // X64-NOT: mov
        // X64: paddd
        // X64-SAME: %xmm0
        return Int4(param.v + 3);
        // X64-NOT: mov
        // X64: ret
    }
}
