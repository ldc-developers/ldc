module tests.codegen.gh3606;

// https://github.com/ldc-developers/ldc/issues/3606

// RUN: %ldc -O3 -output-ll -of=%t.ll %s
// RUN: FileCheck %s < %t.ll
import ldc.simd;
import core.simd;

// CHECK-LABEL: define {{.*}} @_D5tests7codegen6gh36064passFIG16hZG16h(
ubyte[16] pass(in ubyte[16] input)
{
    ubyte16 invec = loadUnaligned!ubyte16(input.ptr);
    // CHECK: = lshr <16 x i8> %{{.*}}, <i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2, i8 2>
    ubyte16 shifted = invec >> 2;
    ubyte[16] result;
    storeUnaligned!ubyte16(shifted, result.ptr);
    return result;
}

// CHECK-LABEL: define void @_D5tests7codegen6gh360613simdConstEvalFZv(
void simdConstEval()
{
    const ubyte[16] binary = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 0xff, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf
    ];
    ubyte[16] result = pass(binary);
    // constant evaluation must have propagated to here
    // CHECK-NEXT: assertPassed:
    assert(result == [0, 0, 0, 0, 1, 1, 1, 1, 2, 63, 2, 2, 3, 3, 3, 3]);
    // CHECK-NEXT: ret void
}
