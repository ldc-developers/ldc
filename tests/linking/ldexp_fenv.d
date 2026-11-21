// Keep this minimal two-call runtime-input regression separate from the
// numerical matrix: extra observations can prevent the faulty LTO reuse.
// Compile the modified core.math explicitly, including its helper bodies.
// REQUIRES: LTO
// RUN: %ldc -O0 %s %S/../../runtime/druntime/src/core/math.d -of=%t-O0%exe && %t-O0%exe
// RUN: %ldc -O2 %s %S/../../runtime/druntime/src/core/math.d -of=%t-O2%exe && %t-O2%exe
// RUN: %ldc -O2 -flto=full %s %S/../../runtime/druntime/src/core/math.d -of=%t-full%exe && %t-full%exe
// RUN: %ldc -O2 -flto=thin %s %S/../../runtime/druntime/src/core/math.d -of=%t-thin%exe && %t-thin%exe
// RUN: %ldc -O2 -c %S/../../runtime/druntime/src/core/math.d -of=%t-math%obj
// RUN: %ldc -O2 -I%S/../../runtime/druntime/src %s %t-math%obj -of=%t-separate%exe && %t-separate%exe
// RUN: %ldc -O2 -flto=full -unittest -main %S/../../runtime/druntime/src/core/math.d -of=%t-unit%exe && %t-unit%exe

module fenv_runtime_assert;

import core.stdc.fenv : fegetround, fesetround, FE_TONEAREST, FE_UPWARD;
import core.stdc.stdio : printf;
import core.math : ldexp;

int main(string[] args) {
    const saved = fegetround();
    scope(exit) fesetround(saved);

    // The argument count is unavailable to compile-time evaluation or LTO.
    // Keep the exponent fixed and derive the significand at runtime.
    // The input is in [1,2), so scaling by 2^-1076 is below half a
    // minimum subnormal: nearest must give 0, upward must give 1 bit.
    ulong inputBits = (cast(ulong) args.length & 0x000fffffffffffffUL) | 0x3ff0000000000000UL;
    double input = *cast(double*) &inputBits;
    fesetround(FE_TONEAREST);
    double nearest = ldexp(input, -1076);
    fesetround(FE_UPWARD);
    double upward = ldexp(input, -1076);

    const nearestBits = *cast(ulong*) &nearest;
    const upwardBits = *cast(ulong*) &upward;
    printf("argc=%llu nearest=%016llx upward=%016llx\n",
           cast(ulong)args.length, nearestBits, upwardBits);
    return nearestBits != 0 || upwardBits != 1;
}
