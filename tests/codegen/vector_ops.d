// RUN: %ldc -run %s

import core.simd;

void main()
{
    static void testGenericOps(T)()
    {
        const T v = [ 1, -2, 3, -4 ];

        assert(-v is [ -1, 2, -3, 4 ]);
        assert(+v is v);

        T v2 = v;
        assert(v2 is v && !(v2 !is v));
        assert((v2 == v) is [ -1, -1, -1, -1 ]);
        assert((v2 != v) is [ 0, 0, 0, 0 ]);

        v2[0] = 0;
        assert(v2 !is v && !(v2 is v));
        assert((v2 == v) is [ 0, -1, -1, -1 ]);
        assert((v2 != v) is [ -1, 0, 0, 0 ]);

        const T comparand = [ 2, -3, 3, -4 ];
        assert((v >  comparand) is [ 0, -1, 0, 0 ]);
        assert((v >= comparand) is [ 0, -1, -1, -1 ]);
        assert((v <  comparand) is [ -1, 0, 0, 0 ]);
        assert((v <= comparand) is [ -1, 0, -1, -1 ]);

        assert(v + v is [ 2, -4, 6, -8 ]);
        assert(v - v is T(0));
        assert(v * v is [ 1, 4, 9, 16 ]);
        assert(v / v is T(1));
        assert(v % T(3) is [ 1, -2, 0, -1 ]);
    }

    testGenericOps!float4();
    testGenericOps!int4();

    const float4 nan = float.nan;
    assert(nan is nan && !(nan !is nan));
    assert((nan == nan) is [ 0, 0, 0, 0 ]);
    assert((nan != nan) is [ -1, -1, -1, -1 ]);

    const int4 i = [ 1, 2, 3, 4 ];
    assert(i << i is [ 2, 8, 24, 64 ]);

    const int4 a = [ 0b1, 0b10, 0b101, 0b100011 ];
    const int4 b = 0b110;
    assert((a & b) is [ 0, 0b10, 0b100, 0b10 ]);
    assert((a | b) is [ 0b111, 0b110, 0b111, 0b100111 ]);
    assert((a ^ b) is [ 0b111, 0b100, 0b11, 0b100101 ]);
    assert(~a is [ ~0b1, ~0b10, ~0b101, ~0b100011 ]);
}
