// RUN: %ldc -enable-dynamic-compile -run %s

import std.stdio;
import std.array;
import std.string;
import ldc.simd;
import ldc.attributes;
import ldc.dynamic_compile;
import core.simd;

@dynamicCompile
ubyte[32] hex_encode_16b(ubyte[16] binary)
{
    const ubyte ascii_a = 'a' - 9 - 1;
    ubyte16 invec = loadUnaligned!ubyte16(binary.ptr);
    ubyte[32] hex;

    ubyte16 masked1 = invec & 0xF;
    ubyte16 masked2 = (invec >> 4) & 0xF;

    ubyte16 cmpmask1 = masked1 > 9;
    ubyte16 cmpmask2 = masked2 > 9;

    ubyte16 masked1_r = ((cmpmask1 & ascii_a) | (~cmpmask1 & '0')) + masked1;
    ubyte16 masked2_r = ((cmpmask2 & ascii_a) | (~cmpmask2 & '0')) + masked2;
    ubyte16 res1 = shufflevector!(ubyte16, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23)(
        masked2_r, masked1_r);
    ubyte16 res2 = shufflevector!(ubyte16, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31)(
        masked2_r, masked1_r);

    storeUnaligned!ubyte16(res1, hex.ptr);
    storeUnaligned!ubyte16(res2, hex.ptr + 16);

    return hex;
}

void main(string[] args)
{
    const ubyte[16] binary = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 0xff, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf
    ];
    const ubyte[32] expected = [
        '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6',
        '0', '7', '0', '8', 'f', 'f', '0', 'a', '0', 'b', '0', 'c', '0', 'd',
        '0', 'e', '0', 'f'
    ];

    CompilerSettings settings;
    auto dump = appender!string();
    settings.dumpHandler = (DumpStage stage, in char[] str) {
        if (DumpStage.FinalAsm == stage)
        {
            write(str);
            dump.put(str);
        }
    };
    settings.optLevel = 3;
    auto f = ldc.dynamic_compile.bind(&hex_encode_16b, binary);
    compileDynamicCode(settings);
    ubyte[32] hex = hex_encode_16b(binary);
    assert(hex == expected);

    assert(f() == expected);
}
