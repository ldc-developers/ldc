// Tests that static array (in)equality of unequal lengths is optimized to `false`.

// RUN: %ldc -c -O3 -output-ll -of=%t.ll %s && FileCheck %s --check-prefix=LLVM < %t.ll

// LLVM-LABEL: define{{.*}} @{{.*}}different_lengths
// ASM-LABEL: different_lengths{{.*}}:
bool different_lengths(bool[4] a, bool[3] b)
{
    // LLVM: ret i1 false
    return a == b;
}
