// RUN: %ldc -run %s

import core.simd;

void main()
{
    static void testGenericOps(T)()
    {
        const T v = [ 1, -2, 3, -4 ];

        assert(-v == [ -1, 2, -3, 4 ]);
        assert(+v == v);

        T v2 = v;
        assert(v2 == v && !(v2 != v));
        assert(v2 is v && !(v2 !is v));
        v2[0] = 0;
        assert(v2 != v && !(v2 == v));
        assert(v2 !is v && !(v2 is v));

        assert(v + v == [ 2, -4, 6, -8 ]);
        assert(v - v == T(0));
        assert(v * v == [ 1, 4, 9, 16 ]);
        assert(v / v == T(1));
        assert(v % T(3) == [ 1, -2, 0, -1 ]);
    }

    testGenericOps!float4();
    testGenericOps!int4();

    const float4 nan = float.nan;
    assert(nan != nan && !(nan == nan));
    assert(nan is nan && !(nan !is nan));

    const int4 i = [ 1, 2, 3, 4 ];
    assert(i << i == [ 2, 8, 24, 64 ]);

    const int4 a = [ 0b1, 0b10, 0b101, 0b100011 ];
    const int4 b = 0b110;
    assert((a & b) == [ 0, 0b10, 0b100, 0b10 ]);
    assert((a | b) == [ 0b111, 0b110, 0b111, 0b100111 ]);
    assert((a ^ b) == [ 0b111, 0b100, 0b11, 0b100101 ]);
    assert(~a == [ ~0b1, ~0b10, ~0b101, ~0b100011 ]);
}
