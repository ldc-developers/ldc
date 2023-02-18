// REQUIRES: host_X86

// RUN: %ldc -run %s

import core.simd;
import ldc.intrinsics;

void main()
{
    const float4 f = [ 1, -2, 3, -4 ];
    const abs = llvm_fabs(f);
    assert(abs is [ 1, 2, 3, 4 ]);

    const int4 i = [ 0b0, 0b10, 0b101, 0b100011 ];
    const numOnes = llvm_ctpop(i);
    assert(numOnes is [ 0, 1, 2, 3 ]);
}
