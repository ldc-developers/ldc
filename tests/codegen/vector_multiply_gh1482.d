// Tests integer SIMD vectorization

// RUN: %ldc -mcpu=haswell -c -output-ll %s -of=%t.ll && FileCheck %s --check-prefix LLVM < %t.ll
// RUN: %ldc -mcpu=haswell -c -output-s  %s -of=%t.s  && FileCheck %s --check-prefix ASM  < %t.s

void mulshorts(ref __vector(ushort[16]) a)
{
    // LLVM: mul <16 x i16>
    // ASM: vpmullw
    a = a * a;
}

void mulints(ref __vector(uint[8]) a)
{
    // LLVM: mul <8 x i32>
    // ASM: vpmulld
    a = a * a;
}

void mullongs(ref __vector(ulong[4]) a)
{
    // LLVM: mul <4 x i64>
    a = a * a;
}
