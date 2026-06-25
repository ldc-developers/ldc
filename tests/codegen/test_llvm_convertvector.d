// RUN: %ldc -run %s

import core.simd;
import ldc.intrinsics;

void main()
{
    // Float-to-float: narrowing (double -> float)
    const double2 d2 = [ 1.5, -2.5 ];
    const f2_narrow = llvm_convertvector!(float2)(d2);
    assert(f2_narrow is [ 1.5f, -2.5f ]);

    // Float-to-float: widening (float -> double)
    const float2 f2 = [ 3.25f, -1.75f ];
    const d2_widen = llvm_convertvector!(double2)(f2);
    assert(d2_widen is [ 3.25, -1.75 ]);

    // Int-to-int: widening signed (short -> int)
    const short4 s4 = [ 1, -2, 3, -4 ];
    const i4_widen = llvm_convertvector!(int4)(s4);
    assert(i4_widen is [ 1, -2, 3, -4 ]);

    // Int-to-int: widening unsigned (ushort -> uint)
    const ushort4 us4 = [ 1, 2, 3, 4 ];
    const ui4_widen = llvm_convertvector!(uint4)(us4);
    assert(ui4_widen is [ 1, 2, 3, 4 ]);

    // Int-to-int: narrowing (long -> int)
    const long4 l4 = [ 1000000L, -2000000L, 3000000L, -4000000L ];
    const i4_narrow = llvm_convertvector!(int4)(l4);
    assert(i4_narrow is [ 1000000, -2000000, 3000000, -4000000 ]);

    // Int-to-float: signed
    const int4 i4 = [ 1, -2, 3, -4 ];
    const f4_from_i4 = llvm_convertvector!(float4)(i4);
    assert(f4_from_i4 is [ 1.0f, -2.0f, 3.0f, -4.0f ]);

    // Int-to-float: unsigned
    const uint4 ui4 = [ 1, 2, 3, 4 ];
    const f4_from_ui4 = llvm_convertvector!(float4)(ui4);
    assert(f4_from_ui4 is [ 1.0f, 2.0f, 3.0f, 4.0f ]);

    // Float-to-int: signed dest
    const float4 f4 = [ 1.5f, -2.7f, 3.2f, -4.9f ];
    const i4_from_f4 = llvm_convertvector!(int4)(f4);
    assert(i4_from_f4 is [ 1, -2, 3, -4 ]);

    // Float-to-int: unsigned dest
    const float4 f4p = [ 1.5f, 2.7f, 3.2f, 4.9f ];
    const ui4_from_f4 = llvm_convertvector!(uint4)(f4p);
    assert(ui4_from_f4 is [ 1, 2, 3, 4 ]);
}
